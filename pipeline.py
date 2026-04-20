"""
Fundraising Data Validation & ETL Pipeline
===========================================
Ingests raw fundraising exports, enforces data quality rules,
reconciles records, and outputs a clean analytical dataset.

Mirrors the Python automation work done at UNON/UNAIDS —
extended here for a PSP fundraising context.

Author: Zipporah Mbai
Stack:  Python · pandas · great_expectations-lite (custom)
"""

import pandas as pd
import numpy as np
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data models
# ---------------------------------------------------------------------------

@dataclass
class ValidationRule:
    column: str
    check: str          # "not_null" | "range" | "regex" | "allowed_values"
    params: dict = field(default_factory=dict)
    severity: str = "error"   # "error" | "warning"


@dataclass
class ValidationResult:
    rule: ValidationRule
    passed: bool
    failed_rows: int
    failed_pct: float
    sample_failures: list


# ---------------------------------------------------------------------------
# 2. Synthetic data generator  (replace with real CSV/API ingest)
# ---------------------------------------------------------------------------

def generate_raw_donations(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulates a messy CRM export with intentional quality issues
    (nulls, out-of-range values, duplicates) to demonstrate validation.
    """
    rng = np.random.default_rng(seed)
    dates = [
        datetime(2023, 1, 1) + timedelta(days=int(d))
        for d in rng.integers(0, 730, n)
    ]

    df = pd.DataFrame({
        "donation_id":       [f"DON-{i:05d}" for i in range(n)],
        "donor_id":          rng.integers(1000, 9999, n).astype(str),
        "amount_usd":        rng.uniform(-10, 5000, n).round(2),   # neg = error
        "currency":          rng.choice(["USD", "EUR", "GBP", "KES", None], n, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
        "gift_date":         dates,
        "channel":           rng.choice(["digital", "direct_mail", "face_to_face", "telemarketing", "UNKNOWN"], n),
        "campaign_code":     rng.choice(["EMG-2023", "REG-2023", "YE-2023", None, "INVALID"], n, p=[0.4, 0.3, 0.2, 0.05, 0.05]),
        "is_recurring":      rng.choice([0, 1, None], n, p=[0.5, 0.45, 0.05]),
        "donor_country":     rng.choice(["US", "DE", "GB", "KE", "UG", None], n, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
        "email":             [f"donor{i}@example.com" if rng.random() > 0.1 else None for i in range(n)],
    })

    # Inject duplicates
    dupes = df.sample(15, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# 3. Validation engine
# ---------------------------------------------------------------------------

class DonationValidator:
    """
    Lightweight data quality engine. Rules are declarative — no hard-coded
    column logic. Add rules to RULES list to extend coverage.
    """

    RULES = [
        ValidationRule("donation_id",  "not_null",       severity="error"),
        ValidationRule("donor_id",     "not_null",       severity="error"),
        ValidationRule("amount_usd",   "range",          {"min": 0.01, "max": 50_000}, severity="error"),
        ValidationRule("currency",     "allowed_values", {"values": ["USD", "EUR", "GBP", "KES", "CHF"]}, severity="error"),
        ValidationRule("channel",      "allowed_values", {"values": ["digital", "direct_mail", "face_to_face", "telemarketing"]}, severity="warning"),
        ValidationRule("campaign_code","regex",          {"pattern": r"^[A-Z]+-\d{4}$"}, severity="warning"),
        ValidationRule("email",        "not_null",       severity="warning"),
        ValidationRule("donor_country","not_null",       severity="warning"),
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results: list[ValidationResult] = []

    def run(self) -> "DonationValidator":
        for rule in self.RULES:
            result = self._check(rule)
            self.results.append(result)
            status = "✓" if result.passed else ("✗" if rule.severity == "error" else "⚠")
            log.info(
                "%s  %-14s %-30s  %d failed (%.1f%%)",
                status, rule.severity.upper(), f"{rule.column} [{rule.check}]",
                result.failed_rows, result.failed_pct * 100,
            )
        return self

    def _check(self, rule: ValidationRule) -> ValidationResult:
        col = self.df[rule.column] if rule.column in self.df else pd.Series([], dtype=object)

        if rule.check == "not_null":
            mask = col.isna()
        elif rule.check == "range":
            mask = col.isna() | (col < rule.params["min"]) | (col > rule.params["max"])
        elif rule.check == "allowed_values":
            mask = col.isna() | ~col.isin(rule.params["values"])
        elif rule.check == "regex":
            mask = col.isna() | ~col.astype(str).str.match(rule.params["pattern"])
        else:
            mask = pd.Series(False, index=self.df.index)

        failed_idx  = self.df.index[mask]
        failed_rows = int(mask.sum())
        failed_pct  = failed_rows / max(len(self.df), 1)
        sample      = col.loc[failed_idx[:3]].tolist()

        return ValidationResult(rule, failed_rows == 0, failed_rows, failed_pct, sample)

    @property
    def has_critical_errors(self) -> bool:
        return any(
            not r.passed and r.rule.severity == "error"
            for r in self.results
        )

    def summary(self) -> dict:
        return {
            "total_rows":       len(self.df),
            "rules_passed":     sum(1 for r in self.results if r.passed),
            "rules_failed":     sum(1 for r in self.results if not r.passed),
            "critical_errors":  sum(1 for r in self.results if not r.passed and r.rule.severity == "error"),
            "warnings":         sum(1 for r in self.results if not r.passed and r.rule.severity == "warning"),
        }


# ---------------------------------------------------------------------------
# 4. Cleaning & reconciliation
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies structured remediation:
    - Remove exact duplicates
    - Drop rows with critical errors (negative amounts, null IDs)
    - Standardise categoricals
    - Fill safe defaults for warnings
    """
    before = len(df)

    # Deduplication
    df = df.drop_duplicates(subset=["donation_id"])
    log.info("Dedup: removed %d rows", before - len(df))

    # Critical: remove invalid amounts
    df = df[df["amount_usd"].between(0.01, 50_000)]

    # Critical: remove null IDs
    df = df.dropna(subset=["donation_id", "donor_id"])

    # Standardise categoricals
    df["channel"] = df["channel"].str.lower().str.strip()
    valid_channels = {"digital", "direct_mail", "face_to_face", "telemarketing"}
    df["channel"] = df["channel"].where(df["channel"].isin(valid_channels), "other")

    # Safe defaults
    df["currency"]     = df["currency"].fillna("USD")
    df["is_recurring"] = df["is_recurring"].fillna(0).astype(int)

    # Derived fields
    df["gift_year"]    = pd.to_datetime(df["gift_date"]).dt.year
    df["gift_quarter"] = pd.to_datetime(df["gift_date"]).dt.quarter

    log.info("Clean: %d → %d rows", before, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Analytical aggregates
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Produces the summary tables a PSP analyst or Tableau dashboard needs."""
    return {
        "by_channel": (
            df.groupby("channel")
              .agg(
                  total_donations=("amount_usd", "count"),
                  total_revenue=("amount_usd", "sum"),
                  avg_gift=("amount_usd", "mean"),
                  recurring_pct=("is_recurring", "mean"),
              )
              .round(2)
              .sort_values("total_revenue", ascending=False)
        ),
        "by_quarter": (
            df.groupby(["gift_year", "gift_quarter"])
              .agg(
                  donations=("amount_usd", "count"),
                  revenue=("amount_usd", "sum"),
              )
              .round(2)
        ),
        "by_campaign": (
            df.groupby("campaign_code")
              .agg(
                  donations=("amount_usd", "count"),
                  revenue=("amount_usd", "sum"),
                  unique_donors=("donor_id", "nunique"),
              )
              .round(2)
              .sort_values("revenue", ascending=False)
        ),
    }


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("UNHCR PSP — Fundraising Data Pipeline")
    print("=" * 60)

    # Ingest
    raw = generate_raw_donations(n=500)
    log.info("Ingested %d raw records", len(raw))

    # Validate
    print("\n── Validation ─────────────────────────────")
    validator = DonationValidator(raw).run()
    summary = validator.summary()
    print(f"\n  Total rows    : {summary['total_rows']:,}")
    print(f"  Rules passed  : {summary['rules_passed']}")
    print(f"  Critical errors: {summary['critical_errors']}")
    print(f"  Warnings       : {summary['warnings']}")

    if validator.has_critical_errors:
        log.warning("Critical errors found — pipeline will clean before output")

    # Clean
    print("\n── Cleaning ────────────────────────────────")
    clean_df = clean(raw)

    # Aggregate
    print("\n── Aggregates ──────────────────────────────")
    aggs = aggregate(clean_df)

    print("\nRevenue by channel:")
    print(aggs["by_channel"].to_string())

    print("\nRevenue by campaign:")
    print(aggs["by_campaign"].to_string())

    # Save outputs
    clean_df.to_csv("donations_clean.csv", index=False)
    aggs["by_channel"].to_csv("agg_by_channel.csv")
    aggs["by_quarter"].to_csv("agg_by_quarter.csv")
    aggs["by_campaign"].to_csv("agg_by_campaign.csv")
    with open("validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved:")
    print("  donations_clean.csv  |  agg_by_*.csv  |  validation_summary.json")
