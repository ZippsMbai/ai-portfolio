"""
Fundraising Insight Engine  (NLP + LLM)
========================================
Analyses donor communication data to extract sentiment, key themes,
and actionable fundraising insights using NLP and LLM APIs.

Demonstrates:
  - Text preprocessing & feature extraction
  - Sentiment classification (rule-based + ML)
  - LLM-powered narrative insight generation
  - Vertex AI / Gemini API integration pattern

Author: Zipporah Mbai
Stack:  Python · scikit-learn · google-generativeai (Gemini) · pandas
"""

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Synthetic donor communication data
# ---------------------------------------------------------------------------

DONOR_MESSAGES = [
    # Positive sentiment
    ("I am so glad I can support UNHCR. Seeing the impact reports really motivates me.", "positive"),
    ("This organisation changes lives. I increased my monthly gift this year.", "positive"),
    ("The emergency appeal updates are incredibly moving. Keep up the vital work.", "positive"),
    ("I've been donating for 10 years and remain deeply committed to the cause.", "positive"),
    ("Just upgraded to a mid-level donor. Proud to support refugee families.", "positive"),
    ("The thank-you letter was personal and touching. Makes me want to give more.", "positive"),
    ("Excellent transparency about how funds are used. That builds my trust.", "positive"),
    ("I brought three friends to the fundraising event. We all signed up!", "positive"),

    # Neutral / informational
    ("Please update my bank details for the direct debit arrangement.", "neutral"),
    ("Can you confirm my total donations for the tax year?", "neutral"),
    ("I'd like to know which programmes my gift is currently supporting.", "neutral"),
    ("Please send me the annual report when it is available.", "neutral"),
    ("What is the breakdown between admin costs and field operations?", "neutral"),
    ("I moved address. Please update your records.", "neutral"),

    # Negative / at-risk
    ("I've received too many emails this month. Please reduce frequency.", "negative"),
    ("I'm considering stopping my donation. The situation feels hopeless.", "negative"),
    ("Why was my gift amount increased without my clear consent?", "negative"),
    ("I haven't received an update in six months. Feels like money into a void.", "negative"),
    ("I'm pausing my giving due to personal financial difficulties.", "negative"),
    ("The telemarketer was quite pushy. That put me off giving.", "negative"),
    ("I requested to be removed from the mailing list twice. Still receiving mail.", "negative"),
    ("I don't see enough evidence of impact for what I donate.", "negative"),
]

# Augment to ~200 samples
def augment(data: list[tuple], n: int = 200, seed: int = 42) -> list[tuple]:
    rng = np.random.default_rng(seed)
    aug = list(data)
    while len(aug) < n:
        text, label = data[rng.integers(0, len(data))]
        # Simple word shuffling as augmentation
        words = text.split()
        rng.shuffle(words)
        aug.append((" ".join(words), label))
    return aug[:n]


# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# 3. Sentiment classifier  (TF-IDF + Logistic Regression)
# ---------------------------------------------------------------------------

def train_sentiment_classifier(data: list[tuple]) -> Pipeline:
    texts  = [preprocess(t) for t, _ in data]
    labels = [l for _, l in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("lr",    LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")),
    ])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    log.info("Sentiment classifier trained:")
    print(classification_report(y_test, y_pred))

    return clf


# ---------------------------------------------------------------------------
# 4. Theme extraction  (keyword clusters)
# ---------------------------------------------------------------------------

THEME_KEYWORDS = {
    "donor_retention_risk":  ["stopping", "pause", "cancel", "quit", "lapse", "hopeless", "void"],
    "communication_issues":  ["email", "mailing", "frequency", "too many", "unsubscribe", "list"],
    "impact_transparency":   ["impact", "evidence", "report", "breakdown", "programmes", "funds used"],
    "positive_engagement":   ["motivates", "proud", "committed", "moving", "vital", "trust", "changed"],
    "payment_admin":         ["bank", "direct debit", "tax", "address", "details", "records"],
    "upgrade_intent":        ["upgraded", "increased", "mid-level", "higher gift"],
}

def extract_themes(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        theme for theme, keywords in THEME_KEYWORDS.items()
        if any(kw in text_lower for kw in keywords)
    ]


# ---------------------------------------------------------------------------
# 5. LLM insight generation  (Gemini / Vertex AI)
# ---------------------------------------------------------------------------

@dataclass
class InsightRequest:
    donor_messages: list[str]
    sentiment_distribution: dict[str, int]
    top_themes: list[str]


def build_prompt(req: InsightRequest) -> str:
    return f"""You are a fundraising analyst for UNHCR's Private Sector Partnerships team.

Below is a summary of recent donor communications:

Sentiment breakdown:
  Positive: {req.sentiment_distribution.get('positive', 0)} donors
  Neutral:  {req.sentiment_distribution.get('neutral', 0)} donors
  Negative (at-risk): {req.sentiment_distribution.get('negative', 0)} donors

Top themes detected: {', '.join(req.top_themes)}

Sample messages:
{chr(10).join(f'- "{m}"' for m in req.donor_messages[:5])}

Provide:
1. A 2-sentence executive summary of the donor sentiment landscape.
2. Three specific, actionable recommendations for the fundraising team.
3. Identify which donor segment needs immediate retention intervention.

Be concise, practical, and grounded in the data above.
"""


def generate_insights_gemini(req: InsightRequest, api_key: Optional[str] = None) -> str:
    """
    Calls Gemini via google-generativeai SDK.
    On Vertex AI, replace with:
        vertexai.init(project=PROJECT, location=LOCATION)
        model = GenerativeModel("gemini-1.5-pro")
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")

    if not api_key:
        log.warning("No GEMINI_API_KEY found — returning mock insight.")
        return _mock_insight(req)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-1.5-flash")
        prompt = build_prompt(req)
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        log.warning("google-generativeai not installed. pip install google-generativeai")
        return _mock_insight(req)
    except Exception as e:
        log.error("Gemini API error: %s", e)
        return _mock_insight(req)


def _mock_insight(req: InsightRequest) -> str:
    """Fallback when API key is unavailable — shows expected output format."""
    neg = req.sentiment_distribution.get("negative", 0)
    pos = req.sentiment_distribution.get("positive", 0)
    return f"""
─── MOCK INSIGHT (set GEMINI_API_KEY for live output) ───

1. Executive Summary:
   Donor sentiment shows {pos} engaged supporters and {neg} at-risk donors expressing
   concerns around communication frequency and perceived impact gaps.

2. Recommendations:
   a) Launch a personalised impact report email series for at-risk donors
      citing specific programme outcomes from their donation period.
   b) Reduce email frequency for donors who haven't opened in 90+ days —
      switch to a quarterly digest format.
   c) Create a mid-level donor upgrade journey targeting high-engagement
      positive-sentiment donors showing upgrade intent signals.

3. Priority segment for retention intervention:
   Donors citing "communication_issues" + "donor_retention_risk" themes —
   estimated {neg} accounts requiring outreach within 30 days.
"""


# ---------------------------------------------------------------------------
# 6. Vertex AI deployment note
# ---------------------------------------------------------------------------

VERTEX_AI_SNIPPET = '''
# ── Deploying this pipeline on Vertex AI ──────────────────────────────────
#
# import vertexai
# from vertexai.generative_models import GenerativeModel
#
# vertexai.init(project="unhcr-psp-prod", location="europe-west4")
# model = GenerativeModel("gemini-1.5-pro")
#
# response = model.generate_content(prompt)
#
# The sentiment classifier can be packaged as a Vertex AI custom prediction
# routine and registered in the Model Registry:
#
# from google.cloud import aiplatform
# aiplatform.Model.upload(
#     display_name="donor-sentiment-v1",
#     artifact_uri="gs://unhcr-models/sentiment/",
#     serving_container_image_uri=(
#         "europe-docker.pkg.dev/vertex-ai/prediction/"
#         "sklearn-cpu.1-2:latest"
#     ),
# )
# ──────────────────────────────────────────────────────────────────────────
'''


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("UNHCR PSP — Fundraising Insight Engine")
    print("=" * 60)

    # Build dataset
    data = augment(DONOR_MESSAGES, n=180)

    # Train sentiment model
    print("\n── Sentiment Classifier ────────────────────")
    clf = train_sentiment_classifier(data)

    # Analyse a batch of new messages
    sample_messages = [text for text, _ in DONOR_MESSAGES]
    preprocessed    = [preprocess(m) for m in sample_messages]
    sentiments      = clf.predict(preprocessed)
    probabilities   = clf.predict_proba(preprocessed)

    # Build results DataFrame
    results = pd.DataFrame({
        "message":     sample_messages,
        "sentiment":   sentiments,
        "confidence":  probabilities.max(axis=1).round(3),
        "themes":      [extract_themes(m) for m in sample_messages],
    })

    print("\n── Per-message Results ─────────────────────")
    for _, row in results.iterrows():
        icon = {"positive": "🟢", "neutral": "🔵", "negative": "🔴"}.get(row["sentiment"], "⚪")
        print(f"  {icon} [{row['sentiment']:8s}] ({row['confidence']:.0%})  {row['message'][:65]}")

    # Aggregate
    sentiment_dist = results["sentiment"].value_counts().to_dict()
    all_themes = [t for themes in results["themes"] for t in themes]
    top_themes = pd.Series(all_themes).value_counts().head(5).index.tolist()

    print(f"\n── Sentiment Distribution ──────────────────")
    for sentiment, count in sentiment_dist.items():
        bar = "█" * count
        print(f"  {sentiment:10s} {count:3d}  {bar}")

    print(f"\n── Top Themes ──────────────────────────────")
    for t in top_themes:
        print(f"  • {t}")

    # LLM insight generation
    print("\n── LLM-Generated Insights ──────────────────")
    req = InsightRequest(
        donor_messages=sample_messages,
        sentiment_distribution=sentiment_dist,
        top_themes=top_themes,
    )
    insight = generate_insights_gemini(req)
    print(insight)

    # Save outputs
    results.to_csv("donor_insights.csv", index=False)
    with open("insight_report.txt", "w") as f:
        f.write(insight)
    print("\nOutputs: donor_insights.csv  |  insight_report.txt")
    print(VERTEX_AI_SNIPPET)
