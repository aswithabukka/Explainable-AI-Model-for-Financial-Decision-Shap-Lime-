# Explainable AI Model for Financial Decision Making (SHAP & LIME)

**Objective**: Build an interpretable machine learning model to predict credit loan approval/rejection and explain decisions using SHAP and LIME
**Dataset**: Credit Risk Dataset (Kaggle) — 32,581 records
**Deployment**: Flask web application with real-time SHAP-based explanations

---

## 📊 Project Overview

This project implements an Explainable AI (XAI) pipeline for credit risk assessment. A Random Forest classifier predicts whether a loan application will be **approved** or **rejected**, and SHAP (SHapley Additive exPlanations) provides transparent, human-readable explanations for each decision — including reasons for rejection and actionable improvement suggestions.

### Key Highlights
- **9 ML models** benchmarked; Random Forest selected as best performer
- **93.3% accuracy** and **96.7% precision** on the test set
- **SHAP & LIME** used for both global and local explainability
- **Flask web app** for real-time loan prediction and explanation

---

## 🗂️ Repository Structure

```
├── Creditrsik_final.ipynb       # Main notebook: EDA, model training, SHAP & LIME analysis
├── app.py                       # Flask web application backend
├── Index.html                   # Frontend loan application form
├── results.html                 # Prediction results and explanation page
├── Dataset/
│   └── credit_risk_dataset.csv  # Raw dataset (32,581 records)
├── Results/                     # Visualization outputs
│   ├── Comparison of SHAP and LIME Local.png
│   ├── Global Mean Shap Value.png
│   ├── LIME BAr plot.png
│   ├── ML model analysis table.png
│   ├── RF CM and ROC.png
│   ├── SHAP Force plot for sample input.png
│   ├── SHAP Waterfall plot.png
│   └── SHAP values table for sample input.png
├── rf_model.zip                 # Trained Random Forest model (pickle)
├── scaler.pkl                   # StandardScaler for numerical features
├── feature_names.pkl            # Encoded feature names used by model
├── shap_explainer.zip           # SHAP TreeExplainer (pickle)
├── Instructions to run code.txt # Setup and run instructions
├── Requirments.txt              # Python dependencies
└── README.md                   # Project overview (this file)
```

---

## 📋 Dataset Details

- **Source**: Credit Risk Dataset (Kaggle)
- **Records**: 32,581 rows × 12 columns
- **Target Variable**: `loan_status` — `0` (Non-default / Approved), `1` (Default / Rejected)
- **Class Distribution**: 78.2% Approved (0), 21.8% Rejected (1)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `person_age` | Numerical | Age of the loan applicant |
| `person_income` | Numerical | Annual income |
| `person_home_ownership` | Categorical | RENT / MORTGAGE / OWN / OTHER |
| `person_emp_length` | Numerical | Employment length (years) |
| `loan_intent` | Categorical | EDUCATION / MEDICAL / VENTURE / PERSONAL / DEBTCONSOLIDATION / HOMEIMPROVEMENT |
| `loan_grade` | Categorical | Credit grade A–G (A = lowest risk) |
| `loan_amnt` | Numerical | Loan amount requested |
| `loan_int_rate` | Numerical | Interest rate on the loan |
| `loan_percent_income` | Numerical | Loan amount as % of income |
| `cb_person_default_on_file` | Categorical | Historical default on record (Y/N) |
| `cb_person_cred_hist_length` | Numerical | Length of credit history (years) |

---

## 🧹 Data Preprocessing

| Step | Details | Records Remaining |
|------|---------|-------------------|
| Raw dataset | Initial load | 32,581 |
| Remove duplicates | 165 duplicate rows dropped | 32,416 |
| Handle missing values | Dropped rows with nulls (`loan_int_rate`: 3,116 missing; `person_emp_length`: 895 missing) | 28,501 |
| Remove outliers | `person_age > 80` (5 rows), `person_emp_length > 80` (2 rows) | **28,493** |

### Feature Engineering
- **One-Hot Encoding**: `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file` (drop_first=True) → 22 features total
- **Standard Scaling**: Applied to 7 numerical columns (`person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`)
- **Train-Test Split**: 80% train / 20% test (random_state=42)

---

## 🤖 Model Benchmarking

9 models were trained and evaluated on the test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** ✅ | **93.3%** | **96.7%** | 72.3% | **82.7%** | **93.2%** |
| Bagging Classifier | 92.9% | 94.1% | 72.4% | 81.9% | 91.5% |
| Gradient Boosting | 92.5% | 93.9% | 70.6% | 80.6% | 92.8% |
| Extra Trees | 91.9% | 92.8% | 68.5% | 78.8% | 92.0% |
| Support Vector Machine | 91.5% | 94.3% | 65.4% | 77.2% | 90.3% |
| K-Nearest Neighbors | 89.2% | 83.0% | 64.1% | 72.4% | 86.7% |
| Decision Tree | 88.6% | 73.3% | 76.5% | 74.8% | 84.3% |
| AdaBoost | 88.3% | 78.9% | 64.4% | 70.9% | 89.9% |
| Logistic Regression | 87.0% | 79.0% | 56.0% | 65.6% | 87.9% |

**Selected Model**: Random Forest Classifier (`n_estimators=100`, `random_state=42`)
**Reason**: Best overall accuracy, highest precision (96.7%), and strong ROC AUC — minimising false approvals in credit risk context.

---

## 🔍 Explainability: SHAP vs LIME

### SHAP (SHapley Additive exPlanations)
- **Method**: `TreeExplainer` for Random Forest
- **Global Explanation**: Mean absolute SHAP values across all test samples
- **Local Explanation**: Per-prediction feature contributions (waterfall plot, force plot)
- **Top influential features** (globally): `loan_percent_income`, `loan_int_rate`, `person_income`, `person_emp_length`, `cb_person_cred_hist_length`

### LIME (Local Interpretable Model-agnostic Explanations)
- **Method**: `LimeTabularExplainer` with `discretize_continuous=False`
- **Scope**: Local explanations only (per prediction)

### SHAP vs LIME Comparison
- **Correlation between SHAP and LIME values**: **-0.23** (low agreement on feature rankings)
- **Conclusion**: SHAP selected as the primary explainability method due to:
  - Consistent global + local explanations in one framework
  - Theoretically grounded (Shapley values from game theory)
  - Better alignment with domain knowledge for credit risk
  - More stable across repeated explanations (LIME can vary)

---

## 🌐 Flask Web Application

The app (`app.py`) provides a real-time loan decision interface:

### How It Works
1. User submits loan application details via the web form (`Index.html`)
2. App encodes inputs (one-hot encoding + standard scaling)
3. Random Forest model predicts **Approved** or **Rejected**
4. SHAP TreeExplainer generates feature-level explanation
5. Results page (`results.html`) displays:
   - **Approved**: Top 3 positive factors supporting the decision
   - **Rejected**: Top 3 negative factors + actionable improvement suggestions

### Required Files for the App
```
app.py
rf_model.pkl          (extracted from rf_model.zip)
scaler.pkl
feature_names.pkl
shap_explainer.pkl    (extracted from shap_explainer.zip)
templates/
  ├── Index.html
  └── results.html
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- All dependencies listed in `Requirments.txt`

### 1. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn shap lime imbalanced-learn matplotlib seaborn
```

### 2. Extract Model Files
```bash
# Extract the zipped model files
unzip rf_model.zip
unzip shap_explainer.zip
```

### 3. Run the Web Application
```bash
python app.py
```

### 4. Access the App
Open your browser and go to: `http://127.0.0.1:5000`

### 5. Run the Notebook
Open `Creditrsik_final.ipynb` in Jupyter Notebook to explore:
- Full data preprocessing pipeline
- Model training and benchmarking
- SHAP global and local explanations
- LIME explanations and SHAP vs LIME comparison

---

## 📈 Results & Visualizations

All output plots are saved in the `Results/` folder:

| File | Description |
|------|-------------|
| `ML model analysis table.png` | Comparison of all 9 models |
| `RF CM and ROC.png` | Random Forest confusion matrix and ROC curve |
| `Global Mean Shap Value.png` | SHAP global feature importance |
| `SHAP Waterfall plot.png` | Local SHAP explanation for a sample input |
| `SHAP Force plot for sample input.png` | SHAP force plot for a sample |
| `SHAP values table for sample input.png` | Feature-level SHAP contributions table |
| `LIME BAr plot.png` | LIME local explanation bar chart |
| `Comparison of SHAP and LIME Local.png` | Side-by-side SHAP vs LIME comparison |

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **ML Framework** | scikit-learn |
| **Explainability** | SHAP, LIME |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Web Framework** | Flask |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Frontend** | HTML |

---

## ⚠️ Disclaimer

This project is for **educational purposes only** (Kaggle Credit Risk dataset). Predictions are based on a publicly available dataset and should **not** be used for real-world credit decisions at financial institutions. Credit risk data is highly sensitive — interpret results cautiously.

---

## 📝 Key Insights

- **`loan_percent_income`** and **`loan_int_rate`** are the strongest predictors of default risk
- **`person_income`** and **`cb_person_cred_hist_length`** heavily influence approvals
- High loan-to-income ratios and elevated interest rates are the most common rejection drivers
- SHAP and LIME show low correlation (-0.23), highlighting they capture different aspects of model behavior
- Random Forest's high precision (96.7%) makes it well-suited for credit risk where false approvals are costly

---

**Last Updated**: November 29, 2025
