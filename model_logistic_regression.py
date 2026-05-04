import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score,
                                     confusion_matrix, classification_report)

# Settings
INPUT_CSV    = "flights_with_weather.csv"
SAMPLE_SIZE  = 500_000  # set to None to use all 3M rows
RANDOM_STATE = 42

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"  Full dataset shape: {df.shape}")

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"  Using random sample of {SAMPLE_SIZE:,} rows for speed.")

# Non-leaky features only — all knowable before the flight departs
CATEGORICAL_FEATURES = ["AIRLINE_CODE", "ORIGIN", "DEST"]
NUMERIC_FEATURES = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "IS_WEEKEND",
    "SCHEDULED_DEP_HOUR", "DEP_TIME_BLOCK",
    "CRS_ELAPSED_TIME", "DISTANCE",
    "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "wind_speed_10m_max",
]
TARGET = "DELAYED"

all_needed = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]
df = df[[c for c in all_needed if c in df.columns]].copy()
df = df.dropna(subset=[TARGET])

# Convert string categories to integers (required by the model)
encoders = {}
for col in CATEGORICAL_FEATURES:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

df = df.fillna(df.median(numeric_only=True))

feature_cols = [c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c in df.columns]
X = df[feature_cols]
y = df[TARGET].astype(int)

print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")
print(f"Class balance — Not delayed: {(y==0).sum():,}  Delayed: {(y==1).sum():,}")

# 80/20 train/test split, stratified to keep the same class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# Scale features so no single column dominates due to its units
# Fit only on training data to avoid leaking test set statistics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nTraining Logistic Regression...")
model = LogisticRegression(
    solver="saga",
    max_iter=1000,
    C=1.0,
    class_weight="balanced",  # prevents the model from ignoring delayed flights
    random_state=RANDOM_STATE,
)
model.fit(X_train_scaled, y_train)
print("Training complete.")

y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print("\n=== Logistic Regression Results ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Not Delayed", "Delayed"]))

cm = confusion_matrix(y_test, y_pred)
print("--- Confusion Matrix ---")
print(f"               Predicted Not Delayed   Predicted Delayed")
print(f"Actual Not Del : {cm[0,0]:>20,}   {cm[0,1]:>17,}")
print(f"Actual Delayed : {cm[1,0]:>20,}   {cm[1,1]:>17,}")

# Positive coefficient means the feature increases delay probability
# Negative means it decreases it
coef_df = pd.DataFrame({
    "feature":     feature_cols,
    "coefficient": model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)

print("\n--- Top 10 Most Influential Features ---")
print(coef_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Delayed", "Delayed"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Delayed", "Delayed"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Logistic Regression — Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("lr_confusion_matrix.png", dpi=120)
print("\nConfusion matrix saved to lr_confusion_matrix.png")
