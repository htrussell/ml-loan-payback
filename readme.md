# 💳 Kaggle Playground Series s5e11: Loan Repayment Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground%20s5e11-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s5e11)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20Local-orange.svg)](#environment--setup)

An end-to-end machine learning and data science pipeline designed to predict loan repayment outcomes (`loan_paid_back`) using tabular borrower financial, demographic, and credit risk indicators.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset Architecture](#-dataset-architecture)
- [Key Exploratory Insights](#-key-exploratory-insights)
- [Repository Structure](#-repository-structure)
- [Environment & Setup](#-environment--setup)
- [Pipeline & Modeling Workflow](#-pipeline--modeling-workflow)
- [Feature Engineering Roadmap](#-feature-engineering-roadmap)
- [Contributing & License](#-contributing--license)

---

## 📖 Overview

The goal of this competition is **binary classification**: forecasting whether an applicant will successfully repay their loan (`1.0`) or default (`0.0`). 

* **Competition**: [Kaggle Playground Series - Season 5, Episode 11](https://www.kaggle.com/competitions/playground-series-s5e11)
* **Training Instances**: 593,994 tabular records
* **Test Instances**: Evaluation partition indexed from ID `593994`
* **Target Metric**: Binary ROC-AUC / F1-Score
* **Target Distribution**: ~79.88% Repaid (`1.0`) vs. ~20.12% Defaulted (`0.0`)

---

## 📊 Dataset Architecture

| Column Name | Type | Description | Observed Domain Range |
| :--- | :--- | :--- | :--- |
| `id` | Integer | Unique applicant identifier | `0` – `N` |
| `annual_income` | Float | Gross annual income | `$6,002.43` – `$393,381.74` |
| `debt_to_income_ratio` | Float | Total monthly debt burden over gross income | `0.011` – `0.627` |
| `credit_score` | Integer | Credit bureau score (FICO equivalent) | `395` – `849` |
| `loan_amount` | Float | Requested principal amount | `$500.09` – `$48,959.95` |
| `interest_rate` | Float | Loan interest rate percentage | `3.20%` – `20.99%` |
| `gender` | Categorical | Borrower gender | `Male`, `Female` |
| `marital_status` | Categorical | Legal marital classification | `Single`, `Married` |
| `education_level` | Categorical | Highest educational achievement | `High School`, `Bachelor's`, `Master's`, `PhD` |
| `employment_status` | Categorical | Employment arrangement | `Employed`, `Self-employed` |
| `loan_purpose` | Categorical | Declared intent of the borrowed capital | `Debt consolidation`, `Business`, `Other`, etc. |
| `grade_subgrade` | Categorical | Granular internal credit quality tier | `A1` through `F5` |
| `loan_paid_back` | Float/Int | **Target Variable**: Repaid (`1.0`) or Default (`0.0`) | `0.0`, `1.0` |

---

## 🔍 Key Exploratory Insights

1. **Credit Score Dominance**: `credit_score` is the single strongest positive predictor of repayment integrity. Higher scores align directly with lower default probabilities.
2. **Leverage & Cost Headwinds**: `debt_to_income_ratio` and `interest_rate` show strong negative correlations with successful repayment. High interest rates compound monthly debt obligations, driving higher default frequencies.
3. **Target Imbalance Considerations**: With an approximate 80:20 class split, metric evaluation relies on stratified validation schemes (e.g., Stratified 5-Fold CV) and post-training probability calibration/threshold optimization.

---

## 📂 Repository Structure

```text
├── data/
│   ├── train.csv                     # Training records (593,994 rows)
│   ├── test.csv                      # Test records for inference
│   └── sample_submission.csv         # Baseline submission format
├── notebooks/
│   └── s5e11_loan_prediction.ipynb   # End-to-end exploratory analysis & modeling
├── src/
│   ├── __init__.py
│   ├── features.py                   # Feature transformation & engineering logic
│   └── models.py                     # Cross-validation loops and estimators
├── .gitignore
├── requirements.txt                  # Python dependencies
├── submission.csv                    # Final predictions output
└── README.md
```

---

## ⚙️ Environment & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/kaggle-s5e11-loan-prediction.git
cd kaggle-s5e11-loan-prediction
```

### 2. Configure Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup Kaggle API Credentials

Place your `kaggle.json` key in your default directory or root folder:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Download and extract the competition dataset:

```bash
kaggle competitions download -c playground-series-s5e11
unzip -q playground-series-s5e11.zip -d data/
rm playground-series-s5e11.zip
```

---

## 🚀 Pipeline & Modeling Workflow

1. **Data Ingestion & Integrity Validation**:
   * Verification of dataset dimensions, variable data types, and potential null/missing values across both sets.
2. **Feature Preprocessing & Encoding**:
   * Continuous variables: Log-transformations (`np.log1p`) applied to right-skewed variables (`annual_income`, `loan_amount`).
   * Ordinal variables: Explicit mapping of `grade_subgrade` to sequential numeric values (`A1` = 1, ..., `F5` = 30).
   * Nominal categoricals: Frequency and one-hot encoding across `loan_purpose`, `education_level`, and `employment_status`.
3. **Cross-Validation Scheme**:
   * Out-of-fold (OOF) evaluation leveraging **5-Fold Stratified K-Fold** with seed averaging to prevent data leakage.
4. **Model Architecture**:
   * Gradient-boosted decision trees (LightGBM, XGBoost, CatBoost).
   * Soft-voting and rank-averaged ensembling for final prediction generation.

---

## 🛠 Feature Engineering Roadmap

- [ ] **Debt Service Burden**:
  $$\text{installment\_to\_income} = \frac{\text{loan\_amount} \times (\text{interest\_rate} / 100)}{\text{annual\_income}}$$
- [ ] **Residual Liquidity**:
  $$\text{disposable\_income} = \text{annual\_income} \times (1 - \text{debt\_to\_income\_ratio})$$
- [ ] **Risk Multiplier**:
  $$\text{risk\_burden\_index} = \frac{\text{debt\_to\_income\_ratio} \times \text{interest\_rate}}{\text{credit\_score}}$$
- [ ] **Hyperparameter Optimization**: Automated Optuna sweeps for learning rate, regularization terms, and tree depth.
- [ ] **Threshold Tuning**: Optimization of binary classification decision thresholds against validation F1-scores.

---

## 📄 License & Acknowledgments

This project is open-source under the [MIT License](LICENSE). Dataset provided by Kaggle as part of the [Playground Series Season 5](https://www.kaggle.com/competitions/playground-series-s5e11).