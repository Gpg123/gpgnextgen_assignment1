# Diabetes Prediction using Machine Learning

End-to-end binary classification pipeline (EDA, preprocessing, training, evaluation, analysis) that predicts whether a patient has diabetes from medical attributes, using **Python, Pandas and Scikit-learn**.

## Repository structure
```
diabetes-prediction/
├── data/diabetes.csv                 # Dataset (768 patients, 8 features + target)
├── src/diabetes_prediction.py        # Complete pipeline (Parts 1-4)
├── outputs/                          # Generated plots, metric tables, run log
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python src/diabetes_prediction.py      # run from the repository root
```
All figures and tables are written to `outputs/`; the console output is saved in `outputs/run_log.txt`. A fixed `random_state=42` makes results reproducible.

---

## Part 1 - Data Preparation

### Dataset exploration summary
| Item | Finding |
|---|---|
| Shape | 768 rows x 9 columns |
| Columns | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome |
| Data types | 6 integer columns, 2 float columns (BMI, DiabetesPedigreeFunction), target is integer. No categorical columns |
| Missing values (NaN) | None |
| Class balance | 500 No Diabetes (65.1%), 268 Diabetes (34.9%) - moderately imbalanced |

**Hidden missing values.** Although there are no NaNs, several columns contain `0`, which is physiologically impossible (a person cannot have 0 blood pressure or 0 BMI):

| Column | Zeros | % of rows |
|---|---|---|
| Glucose | 5 | 0.65% |
| BloodPressure | 35 | 4.56% |
| SkinThickness | 227 | 29.56% |
| Insulin | 374 | 48.70% |
| BMI | 11 | 1.43% |

These zeros are treated as missing values. (`Pregnancies = 0` is valid and left untouched.) Insulin and SkinThickness have very high missingness, so imputation there is approximate.

Basic statistics (mean): Glucose 120.9, BloodPressure 69.1, BMI 32.0, Age 33.2, Pregnancies 3.85. Insulin is highly right-skewed (mean 79.8, max 846). See `outputs/01_feature_distributions.png` and `outputs/02_correlation_heatmap.png`.

### Features and target
- **Features (X):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age (8 numeric features)
- **Target (y):** `Outcome` (0 = No Diabetes, 1 = Diabetes)

### Preprocessing steps
1. Replaced impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI with `NaN`.
2. **Median imputation** (`SimpleImputer`) - median is robust to the skew/outliers in these columns.
3. **Standardisation** (`StandardScaler`) - needed because features are on very different scales (e.g. Insulin up to 846 vs DiabetesPedigreeFunction ~0.5), and it makes logistic regression coefficients comparable.
4. No categorical encoding required (all features numeric).
5. Imputer and scaler are inside a scikit-learn `Pipeline`, so they are fitted **only on training data**, which prevents data leakage into the test set.

### Train-test split
Stratified 80/20 split (`random_state=42`), preserving the class ratio:

| Set | Rows | Share | Diabetes share |
|---|---|---|---|
| Training | 614 | 80% | 34.9% |
| Testing | 154 | 20% | 35.1% |

---

## Part 2 - Model Building

**Model:** Logistic Regression (`sklearn.linear_model.LogisticRegression`) inside a Pipeline: `SimpleImputer -> StandardScaler -> LogisticRegression`.

**Why Logistic Regression is suitable**
- The task is **binary classification**, which is exactly what logistic regression is designed for; it outputs a probability of diabetes via the sigmoid function.
- It is **interpretable**: each coefficient shows how a feature shifts the log-odds of diabetes. This matters in a healthcare setting where predictions must be explainable.
- It is a **strong, fast baseline** that works well on small tabular datasets (768 rows) and is less prone to overfitting than complex models.
- It provides **probabilities**, so the decision threshold can be tuned (e.g. to catch more diabetic patients).

---

## Part 3 - Model Training and Evaluation

### Results
| Metric | Training | Testing |
|---|---|---|
| **Accuracy** | **0.7964** | **0.7078** |
| Precision | 0.7697 | 0.6000 |
| Recall | 0.5935 | 0.5000 |
| F1-score | 0.6702 | 0.5455 |

Test ROC-AUC: **0.813**

**Confusion matrix (test set, 154 patients)** - see `outputs/04_confusion_matrix.png`

| | Predicted No Diabetes | Predicted Diabetes |
|---|---|---|
| **Actual No Diabetes** | 82 (TN) | 18 (FP) |
| **Actual Diabetes** | 27 (FN) | 27 (TP) |

Class-wise (test): No Diabetes precision 0.75 / recall 0.82 / F1 0.78; Diabetes precision 0.60 / recall 0.50 / F1 0.55.

### Training vs testing comparison
Training accuracy (79.6%) is about 8.9 points higher than testing accuracy (70.8%). Some gap is expected, but it is moderate rather than severe. Logistic regression is a low-complexity model, so this is not classic overfitting; much of the drop reflects the small test set (154 rows, only 54 diabetic cases), where a handful of cases moves accuracy noticeably.

### Interpretation
- The model beats the majority-class baseline (64.9% on the test set), so it has learned useful signal, and the ROC-AUC of 0.81 shows good ability to rank diabetic patients above non-diabetic ones.
- The model is better at identifying healthy patients (recall 0.82) than diabetic ones (recall 0.50): it **missed 27 of 54 diabetic patients** (false negatives).

---

## Part 4 - Model Performance Analysis

**Reliability of the estimate.** 5-fold stratified cross-validation on the full dataset gives accuracy of **77.2% (+/- 1.7%)**, which is higher than the 70.8% from the single split. This suggests the single test split was somewhat pessimistic, and **~77% is a more trustworthy estimate** of real-world accuracy.

**Key drivers of diabetes (standardised coefficients, `outputs/06_feature_importance.png`)**

| Feature | Coefficient | Odds ratio per +1 SD |
|---|---|---|
| Glucose | 1.183 | 3.26 |
| BMI | 0.689 | 1.99 |
| Pregnancies | 0.377 | 1.46 |
| DiabetesPedigreeFunction | 0.233 | 1.26 |
| Age | 0.148 | 1.16 |
| SkinThickness | 0.028 | 1.03 |
| BloodPressure | -0.044 | 0.96 |
| Insulin | -0.066 | 0.94 |

- **Glucose is by far the strongest predictor**, followed by BMI - consistent with medical knowledge.
- Pregnancies, family history (pedigree function) and age contribute modestly.
- SkinThickness, BloodPressure and Insulin have near-zero effect. Their heavy missingness/imputation likely weakens their signal, so this should not be read as "insulin is irrelevant to diabetes".

**Business / analytical insights**
- Glucose and BMI screening alone gives a strong first-pass risk signal; patients with high values of both should be prioritised.
- In medical screening, a **false negative (missed diabetic) is costlier than a false positive**. With the default 0.5 threshold, recall for diabetes is only 50%. Lowering the threshold would catch more cases at the cost of more false alarms.

**Limitations**
- Small dataset; the test set holds only 54 positive cases, so metrics have wide uncertainty.
- Heavy imputation for Insulin (49%) and SkinThickness (30%).
- Moderate class imbalance (35% positive) pulls the model towards the majority class.
- The dataset is the Pima Indians dataset (adult women of Pima heritage), so results may not generalise to other populations.

**Suggested improvements**
1. Tune the decision threshold or use `class_weight="balanced"` to improve recall.
2. Try Random Forest / Gradient Boosting and compare via cross-validation.
3. Hyper-parameter tuning with `GridSearchCV`.
4. Feature engineering (e.g. BMI categories, Glucose x BMI interaction, age groups).
5. Evaluate with Precision-Recall curves, which suit imbalanced data.

## Conclusion
A Logistic Regression pipeline achieves about **71% test accuracy (77% cross-validated)** and **ROC-AUC 0.81**, clearly better than the 65% baseline. Glucose and BMI are the dominant predictors. The main weakness is low recall for diabetic patients (50%), which should be addressed before any screening use.
