"""
Donor Churn Predictor
=====================
Predicts which donors are at risk of lapsing, enabling targeted
retention campaigns. Mirrors the kind of ML work done in UNHCR's
Private Sector Partnerships (PSP) fundraising analytics team.

Author: Zipporah Mbai
Stack:  Python · scikit-learn · pandas · joblib
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib
import json
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Synthetic dataset  (replace with real CRM export in production)
# ---------------------------------------------------------------------------

def generate_donor_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Simulates a donor CRM extract with features relevant to churn prediction.
    In a real UNHCR PSP context this would come from Salesforce / Raisers Edge.
    """
    rng = np.random.default_rng(seed)

    data = pd.DataFrame({
        # Giving behaviour
        "months_since_last_gift":   rng.integers(1, 36, n),
        "total_gifts_lifetime":     rng.integers(1, 120, n),
        "avg_gift_amount_usd":      rng.uniform(5, 500, n).round(2),
        "gift_frequency_per_year":  rng.uniform(0.5, 12, n).round(1),
        "giving_streak_months":     rng.integers(0, 60, n),

        # Engagement signals
        "email_open_rate":          rng.uniform(0, 1, n).round(3),
        "email_click_rate":         rng.uniform(0, 0.5, n).round(3),
        "appeals_received_ytd":     rng.integers(1, 12, n),
        "events_attended_ytd":      rng.integers(0, 5, n),
        "num_campaigns_responded":  rng.integers(0, 10, n),

        # Donor profile
        "donor_tenure_years":       rng.uniform(0.1, 20, n).round(1),
        "is_recurring_donor":       rng.integers(0, 2, n),
        "has_upgraded_gift":        rng.integers(0, 2, n),
        "channel":                  rng.choice(
            ["digital", "direct_mail", "face_to_face", "telemarketing"], n
        ),
    })

    # Realistic churn label: donors lapse more when inactive + low engagement
    churn_score = (
        0.05 * data["months_since_last_gift"]
        - 0.02 * data["total_gifts_lifetime"]
        - 0.3  * data["email_open_rate"]
        - 0.5  * data["is_recurring_donor"]
        + 0.1  * rng.standard_normal(n)
    )
    data["churned"] = (churn_score > churn_score.quantile(0.35)).astype(int)
    return data


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Recency-Frequency-Monetary (RFM) composite
    df["rfm_score"] = (
        (1 / (df["months_since_last_gift"] + 1))
        * df["gift_frequency_per_year"]
        * np.log1p(df["avg_gift_amount_usd"])
    )

    # Engagement index
    df["engagement_index"] = (
        df["email_open_rate"] * 0.4
        + df["email_click_rate"] * 0.4
        + df["events_attended_ytd"] * 0.2
    )

    # Loyalty flag
    df["is_loyal"] = (
        (df["donor_tenure_years"] > 3) & (df["is_recurring_donor"] == 1)
    ).astype(int)

    # One-hot encode channel
    df = pd.get_dummies(df, columns=["channel"], drop_first=True)
    return df


# ---------------------------------------------------------------------------
# 3. Train / evaluate
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != "churned"]
    X = df[feature_cols]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Metrics
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")

    metrics = {
        "roc_auc_test":     round(roc_auc, 4),
        "cv_roc_auc_mean":  round(cv_scores.mean(), 4),
        "cv_roc_auc_std":   round(cv_scores.std(),  4),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
    }

    # Feature importance
    model     = pipeline.named_steps["model"]
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    metrics["top_10_features"] = importances.head(10).to_dict()

    return pipeline, metrics


# ---------------------------------------------------------------------------
# 4. Scoring / inference
# ---------------------------------------------------------------------------

def score_donors(pipeline: Pipeline, new_donors: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a risk-ranked DataFrame with churn probability and segment label.
    Designed to feed directly into a campaign targeting workflow.
    """
    new_donors = engineer_features(new_donors)
    feature_cols = pipeline.feature_names_in_  # preserved from training
    # align columns (handle any missing dummies)
    for col in feature_cols:
        if col not in new_donors.columns:
            new_donors[col] = 0
    new_donors = new_donors[feature_cols]

    proba = pipeline.predict_proba(new_donors)[:, 1]
    result = new_donors.copy()
    result["churn_probability"] = proba.round(4)
    result["risk_segment"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["low_risk", "medium_risk", "high_risk"],
    )
    return result.sort_values("churn_probability", ascending=False)


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("UNHCR PSP — Donor Churn Predictor")
    print("=" * 60)

    df = generate_donor_data(n=2000)
    print(f"\nDataset: {len(df):,} donors  |  Churn rate: {df['churned'].mean():.1%}\n")

    pipeline, metrics = train(df)

    print(f"Test  ROC-AUC : {metrics['roc_auc_test']}")
    print(f"CV    ROC-AUC : {metrics['cv_roc_auc_mean']} ± {metrics['cv_roc_auc_std']}")
    print("\nTop 10 predictive features:")
    for feat, imp in metrics["top_10_features"].items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Save model + metrics
    joblib.dump(pipeline, "donor_churn_model.pkl")
    with open("metrics.json", "w") as f:
        metrics_serializable = {
            k: v for k, v in metrics.items() if k != "classification_report"
        }
        json.dump(metrics_serializable, f, indent=2)
    print("\nModel saved → donor_churn_model.pkl")
    print("Metrics saved → metrics.json")

    # Demo: score 10 new donors
    new = generate_donor_data(n=10, seed=99)
    scored = score_donors(pipeline, new)
    print("\nSample scored donors (top 5 by churn risk):")
    print(scored[["churn_probability", "risk_segment"]].head())
