AI & ML Fundraising Analytics Portfolio
Three production-ready Python projects demonstrating AI/ML capabilities applied to humanitarian fundraising analytics.
Author: Zipporah Mbai | LinkedIn | GitHub


Projects
1. 🔴 Donor Churn Predictor
/donor_churn_predictor

Gradient Boosting classifier that identifies donors at risk of lapsing, enabling targeted retention campaigns.

Model: GradientBoostingClassifier | ROC-AUC: ~0.99
Features: RFM score, engagement index, loyalty flag, channel
Output: Risk-ranked donor list (low/medium/high_risk)
Cloud: Ready for Vertex AI Model Registry deployment


2. 🔄 Fundraising Data Validation & ETL Pipeline
/fundraising_data_pipeline

End-to-end data pipeline: ingests raw CRM exports, enforces 8 validation rules, cleans records, and outputs analytical aggregates.

Validates: Null checks, range rules, regex patterns, allowed-value lists
Cleans: Deduplication, standardisation, safe defaults
Outputs: Clean CSV + quarterly/channel/campaign aggregates
Cloud: Designed to run as a containerised step in Azure Data Factory or GCP Dataflow


3. 🧠 Fundraising Insight Engine (NLP + LLM)
/nlp_insight_engine

NLP pipeline that classifies donor communication sentiment, extracts fundraising themes, and generates actionable insights via Gemini / Vertex AI.

Classifier: TF-IDF + Logistic Regression (positive / neutral / negative)
Themes: 6 fundraising-specific keyword clusters
LLM: Gemini 1.5 Flash/Pro via google-generativeai SDK or Vertex AI
Output: Per-donor sentiment + executive insight report


Tech stack
Layer
Technologies
Language
Python 3.10+
ML
scikit-learn, pandas, numpy
LLM
Google Gemini API / Vertex AI
Cloud
Azure (AKS, Azure ML) · GCP (Vertex AI, Cloud Storage, Dataflow)
MLOps
joblib model serialisation · MLflow-ready metrics · Docker-compatible



Running locally
git clone https://github.com/ZippsMbai/unhcr-psp-ai-portfolio

cd ai-portfolio

# Project 1

cd 1_donor_churn_predictor

pip install -r requirements.txt

python donor_churn_model.py

# Project 2

cd ../2_fundraising_data_pipeline

pip install -r requirements.txt

python pipeline.py

# Project 3

cd ../3_nlp_insight_engine

pip install -r requirements.txt

export GEMINI_API_KEY=your_key   # optional — falls back to mock

python insight_engine.py


Background
These projects were built to demonstrate AI/ML capabilities relevant to  fundraising analytics mission. They reflect 6+ years of Python development and data engineering experience at several orgs., now applied directly to the PSP domain.
