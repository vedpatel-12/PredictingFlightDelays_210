import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import OrdinalEncoder
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score,
                                     confusion_matrix, classification_report)
from sklearn.inspection      import permutation_importance

# Settings
INPUT_CSV    = "flights_with_weather.csv"
SAMPLE_SIZE  = None  # HistGradientBoosting handles 3M rows efficiently
RANDOM_STATE = 42

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"  Full dataset shape: {df.shape}")

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"  Using random sample of {SAMPLE_SIZE:,} rows.")

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

# OrdinalEncoder is required by HistGradientBoostingClassifier
# unknown_value=-1 handles any new categories seen at test time
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
cat_cols_in_df = [c for c in CATEGORICAL_FEATURES if c in df.columns]
df[cat_cols_in_df] = enc.fit_transform(df[cat_cols_in_df].astype(str))

numeric_df = df.select_dtypes(include="number")
df[numeric_df.columns] = numeric_df.fillna(numeric_df.median())

feature_cols = [c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c in df.columns]
X = df[feature_cols]
y = df[TARGET].astype(int)

print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")
print(f"Class balance — Not delayed: {(y==0).sum():,}  Delayed: {(y==1).sum():,}")

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")

print("\nTraining HistGradientBoostingClassifier...")
print("  (Early stopping enabled — will stop when validation score stops improving)")
model = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    class_weight="balanced",  # prevents the model from ignoring delayed flights
    random_state=RANDOM_STATE,
    verbose=1,
)
model.fit(X_train, y_train)
print(f"\nTraining stopped at {model.n_iter_} trees (out of max {model.max_iter}).")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print("\n=== Gradient Boosting Results ===")
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

# HistGradientBoosting has no .feature_importances_, so we use permutation
# importance instead — shuffles each feature and measures the ROC-AUC drop
print("\nComputing permutation importance on a 20,000-row sample (may take ~30s)...")
rng        = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(len(X_test), size=min(20_000, len(X_test)), replace=False)
X_sample   = X_test.iloc[sample_idx]
y_sample   = y_test.iloc[sample_idx]

perm = permutation_importance(
    model, X_sample, y_sample,
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring="roc_auc",
)

importance_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": perm.importances_mean,
    "std":        perm.importances_std,
}).sort_values("importance", ascending=False)

print("\n--- Feature Importance (mean ROC-AUC drop per feature) ---")
print(importance_df.to_string(index=False))

top15 = importance_df.head(15)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top15["feature"][::-1], top15["importance"][::-1],
        xerr=top15["std"][::-1], color="darkorange", ecolor="gray", capsize=3)
ax.set_xlabel("Mean ROC-AUC drop (higher = more important)")
ax.set_title("Gradient Boosting — Permutation Feature Importance")
plt.tight_layout()
plt.savefig("gb_feature_importance.png", dpi=120)
print("\nFeature importance chart saved to gb_feature_importance.png")

# Learning curve - shows how the score improved as more trees were added
train_scores = getattr(model, "train_score_", None)
val_scores   = getattr(model, "validation_score_", None)
if train_scores is not None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_scores, label="Train score")
    if val_scores is not None:
        ax.plot(val_scores, label="Validation score")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Score")
    ax.set_title("Gradient Boosting — Learning Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig("gb_learning_curve.png", dpi=120)
    print("Learning curve saved to gb_learning_curve.png")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Oranges")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Delayed", "Delayed"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Delayed", "Delayed"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Gradient Boosting — Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("gb_confusion_matrix.png", dpi=120)
print("Confusion matrix saved to gb_confusion_matrix.png")
