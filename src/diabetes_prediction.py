"""
Diabetes Prediction using Machine Learning (Logistic Regression)
End-to-end pipeline: EDA -> preprocessing -> training -> evaluation -> analysis

Run from the repository root:
    python src/diabetes_prediction.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             roc_auc_score, RocCurveDisplay)

RANDOM_STATE = 42
DATA_PATH = "data/diabetes.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ----------------------------------------------------------------------
# PART 1 - DATA PREPARATION
# ----------------------------------------------------------------------
section("PART 1: DATA PREPARATION")

# 1. Load
df = pd.read_csv(DATA_PATH)

# 2. Explore
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nMissing values (NaN):\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe().T.round(2))
print("\nTarget distribution:\n", df["Outcome"].value_counts())
print("\nTarget distribution (%):\n", (df["Outcome"].value_counts(normalize=True) * 100).round(2))

# Hidden missing values: physiologically impossible zeros
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_counts = (df[zero_cols] == 0).sum()
print("\nImpossible zeros (hidden missing values):\n",
      pd.DataFrame({"zeros": zero_counts, "percent": (zero_counts / len(df) * 100).round(2)}))

# EDA plots
df.hist(figsize=(12, 9), bins=25, edgecolor="black")
plt.suptitle("Feature distributions (raw data)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_feature_distributions.png", dpi=120); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation matrix")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png", dpi=120); plt.close()

plt.figure(figsize=(4.5, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Class balance (0 = No Diabetes, 1 = Diabetes)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/03_class_balance.png", dpi=120); plt.close()

# Replace impossible zeros with NaN so they can be imputed
df_clean = df.copy()
df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)
print("\nMissing values after converting impossible zeros to NaN:\n", df_clean.isnull().sum())

# 3. Features / target
X = df_clean.drop(columns="Outcome")
y = df_clean["Outcome"]
print("\nFeatures (X):", list(X.columns), "| shape:", X.shape)
print("Target (y): Outcome | shape:", y.shape)
print("Categorical variables: none (all features are numeric)")

# 4. Train-test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"\nTraining set: {X_train.shape[0]} rows ({X_train.shape[0]/len(X):.0%})")
print(f"Testing set : {X_test.shape[0]} rows ({X_test.shape[0]/len(X):.0%})")
print("Train class ratio:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Test  class ratio:", y_test.value_counts(normalize=True).round(3).to_dict())

# 5. Preprocessing: median imputation + standardisation.
#    Wrapped in a Pipeline so statistics are learned ONLY from training data (no leakage).
preprocess_steps = [("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())]

# ----------------------------------------------------------------------
# PART 2 - MODEL BUILDING
# ----------------------------------------------------------------------
section("PART 2: MODEL BUILDING")
model = Pipeline(preprocess_steps + [
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
print(model)

# ----------------------------------------------------------------------
# PART 3 - MODEL TRAINING & EVALUATION
# ----------------------------------------------------------------------
section("PART 3: MODEL TRAINING AND EVALUATION")
model.fit(X_train, y_train)
print("Model trained successfully.")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nTraining accuracy: {train_acc:.4f}")
print(f"Testing  accuracy: {test_acc:.4f}")

prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_proba)
print(f"Test precision: {prec:.4f}\nTest recall   : {rec:.4f}\nTest F1-score : {f1:.4f}\nTest ROC-AUC  : {auc:.4f}")
print("\nClassification report (test):\n",
      classification_report(y_test, y_test_pred, target_names=["No Diabetes", "Diabetes"]))

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix (test):\n", cm)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(cmap="Blues")
plt.title("Confusion matrix - test set")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_confusion_matrix.png", dpi=120); plt.close()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.title("ROC curve - test set"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/05_roc_curve.png", dpi=120); plt.close()

comparison = pd.DataFrame({
    "Accuracy": [train_acc, test_acc],
    "Precision": [precision_score(y_train, y_train_pred), prec],
    "Recall": [recall_score(y_train, y_train_pred), rec],
    "F1-score": [f1_score(y_train, y_train_pred), f1]}, index=["Train", "Test"]).round(4)
print("\nTrain vs Test comparison:\n", comparison)
print(f"Accuracy gap (train - test): {train_acc - test_acc:+.4f}")
comparison.to_csv(f"{OUT_DIR}/train_vs_test_metrics.csv")

# ----------------------------------------------------------------------
# PART 4 - PERFORMANCE ANALYSIS
# ----------------------------------------------------------------------
section("PART 4: MODEL PERFORMANCE ANALYSIS")

# Cross-validation for a more reliable estimate than a single split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"5-fold CV accuracy: {cv_scores.round(3)} -> mean {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Baseline: always predict majority class
baseline = (y_test == 0).mean()
print(f"Majority-class baseline accuracy: {baseline:.4f}")

# Feature importance via standardised coefficients (log-odds per 1 SD increase)
coefs = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.named_steps["clf"].coef_[0]})
coefs["Odds_Ratio_per_SD"] = np.exp(coefs["Coefficient"])
coefs = coefs.sort_values("Coefficient", ascending=False).round(3)
print("\nFeature coefficients (standardised):\n", coefs.to_string(index=False))
coefs.to_csv(f"{OUT_DIR}/feature_coefficients.csv", index=False)

plt.figure(figsize=(7, 4.5))
c = coefs.sort_values("Coefficient")
plt.barh(c["Feature"], c["Coefficient"], color=np.where(c["Coefficient"] > 0, "tomato", "steelblue"))
plt.axvline(0, color="k", lw=0.8)
plt.title("Logistic Regression coefficients (standardised features)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/06_feature_importance.png", dpi=120); plt.close()

print("\nAll plots and tables saved to the 'outputs/' folder.")
