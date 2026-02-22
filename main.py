import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  PHARMACOGENOMIC RISK PREDICTION — ADVANCED TRAINING")
print("=" * 60)

# ──────────────── 1. LOAD DATA ────────────────
print("\n[1/7] Loading datasets...")
train = pd.read_csv("data/drugsComTrain_raw.csv")
test = pd.read_csv("data/drugsComTest_raw.csv")

print(f"  Train: {train.shape[0]:,} rows  |  Test: {test.shape[0]:,} rows")

# ──────────────── 2. CLEAN DATA ────────────────
print("[2/7] Cleaning data...")

# Drop rows with missing values in key columns
for df in [train, test]:
    df.dropna(subset=["drugName", "condition", "review", "rating"], inplace=True)

# Clean HTML entities in reviews
def clean_review(text):
    text = str(text)
    text = re.sub(r"&#\d+;", " ", text)      # HTML entities
    text = re.sub(r"<.*?>", " ", text)         # HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Keep only letters
    text = re.sub(r"\s+", " ", text).strip()   # Multiple spaces
    return text.lower()

train["clean_review"] = train["review"].apply(clean_review)
test["clean_review"] = test["review"].apply(clean_review)

# ──────────────── 3. CREATE 5-TIER RISK LABELS ────────────────
print("[3/7] Creating 5-tier risk classification...")

def create_risk_tier(rating):
    """
    5-tier risk classification:
      Rating 1-2  → 4 (Critical Risk)
      Rating 3-4  → 3 (High Risk)
      Rating 5-6  → 2 (Moderate Risk)
      Rating 7-8  → 1 (Low Risk)
      Rating 9-10 → 0 (Minimal Risk)
    """
    if rating <= 2:
        return 4
    elif rating <= 4:
        return 3
    elif rating <= 6:
        return 2
    elif rating <= 8:
        return 1
    else:
        return 0

RISK_LABELS = {
    0: "Minimal Risk",
    1: "Low Risk",
    2: "Moderate Risk",
    3: "High Risk",
    4: "Critical Risk"
}

train["risk_tier"] = train["rating"].apply(create_risk_tier)
test["risk_tier"] = test["rating"].apply(create_risk_tier)

print("  Risk distribution (train):")
for tier, label in sorted(RISK_LABELS.items()):
    count = (train["risk_tier"] == tier).sum()
    print(f"    {label}: {count:,}")

# ──────────────── 4. FEATURE ENGINEERING ────────────────
print("[4/7] Engineering features...")

# --- Text features (TF-IDF) ---
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=5,
    max_df=0.95,
    sublinear_tf=True
)

X_text = vectorizer.fit_transform(train["clean_review"])
print(f"  TF-IDF shape: {X_text.shape}")

# --- Numeric features ---
train["review_length"] = train["clean_review"].apply(len)
train["word_count"] = train["clean_review"].apply(lambda x: len(x.split()))

# Sentiment keyword counts
NEGATIVE_WORDS = {"pain", "horrible", "terrible", "worst", "bleeding", "nausea",
                  "vomiting", "death", "died", "emergency", "allergic", "rash",
                  "seizure", "depression", "suicidal", "anxiety", "insomnia",
                  "headache", "dizziness", "fatigue", "weight", "swelling"}
POSITIVE_WORDS = {"great", "excellent", "amazing", "wonderful", "perfect",
                  "love", "best", "effective", "helped", "relief", "recommend",
                  "fantastic", "improved", "works", "miracle", "happy"}

def count_keywords(text, keywords):
    words = text.lower().split()
    return sum(1 for w in words if w in keywords)

train["neg_word_count"] = train["clean_review"].apply(lambda x: count_keywords(x, NEGATIVE_WORDS))
train["pos_word_count"] = train["clean_review"].apply(lambda x: count_keywords(x, POSITIVE_WORDS))
train["useful_count_log"] = np.log1p(train["usefulCount"].fillna(0))

numeric_features = train[["review_length", "word_count", "neg_word_count",
                           "pos_word_count", "useful_count_log"]].values

# --- Drug name encoding (top 200 drugs) ---
top_drugs = train["drugName"].value_counts().head(200).index.tolist()
drug_encoder = LabelEncoder()
drug_encoder.fit(top_drugs + ["_OTHER_"])

def encode_drug(name):
    return drug_encoder.transform([name if name in top_drugs else "_OTHER_"])[0]

train["drug_encoded"] = train["drugName"].apply(encode_drug)

# --- Condition encoding (top 100 conditions) ---
top_conditions = train["condition"].value_counts().head(100).index.tolist()
condition_encoder = LabelEncoder()
condition_encoder.fit(top_conditions + ["_OTHER_"])

def encode_condition(name):
    return condition_encoder.transform([str(name) if str(name) in top_conditions else "_OTHER_"])[0]

train["condition_encoded"] = train["condition"].apply(encode_condition)

categorical_features = train[["drug_encoded", "condition_encoded"]].values

# --- Combine all features ---
from scipy.sparse import hstack, csr_matrix

X_combined = hstack([
    X_text,
    csr_matrix(numeric_features),
    csr_matrix(categorical_features)
])

y = train["risk_tier"]
print(f"  Combined feature matrix: {X_combined.shape}")

# ──────────────── 5. TRAIN MODEL ────────────────
print("[5/7] Training Random Forest model...")

X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

# ──────────────── 6. EVALUATE ────────────────
print("[6/7] Evaluating model...")

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\n  Validation Accuracy: {accuracy:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_val, y_pred, target_names=[RISK_LABELS[i] for i in sorted(RISK_LABELS.keys())]))
print("  Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ──────────────── 7. SAVE ARTIFACTS ────────────────
print("\n[7/7] Saving model artifacts...")

artifacts = {
    "model": model,
    "vectorizer": vectorizer,
    "drug_encoder": drug_encoder,
    "condition_encoder": condition_encoder,
    "top_drugs": top_drugs,
    "top_conditions": top_conditions,
    "risk_labels": RISK_LABELS,
    "negative_words": NEGATIVE_WORDS,
    "positive_words": POSITIVE_WORDS,
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

# Also save the processed datasets for the dashboard
dashboard_data = {
    "drug_risk_stats": train.groupby("drugName").agg(
        avg_rating=("rating", "mean"),
        review_count=("rating", "count"),
        risk_distribution=("risk_tier", lambda x: x.value_counts().to_dict())
    ).reset_index(),
    "condition_risk_stats": train.groupby("condition").agg(
        avg_rating=("rating", "mean"),
        review_count=("rating", "count"),
        risk_distribution=("risk_tier", lambda x: x.value_counts().to_dict())
    ).reset_index(),
    "overall_risk_distribution": train["risk_tier"].value_counts().to_dict(),
    "risk_labels": RISK_LABELS,
    "total_reviews": len(train),
    "total_drugs": train["drugName"].nunique(),
    "total_conditions": train["condition"].nunique(),
}

with open("dashboard_data.pkl", "wb") as f:
    pickle.dump(dashboard_data, f)

print("\n✅ All artifacts saved successfully!")
print(f"  → model_artifacts.pkl  (model + encoders)")
print(f"  → dashboard_data.pkl   (pre-computed analytics)")
print("=" * 60)