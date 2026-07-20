# Credit Risk Analysis Model and Web App
An end-to-end Machine Learning solution designed to evaluate individual credit applicants and predict whether their credit risk is **GOOD** or **BAD**. This repository contains the complete workflow, from data preprocessing and model training to a live interactive web app deployed on Streamlit Cloud.

**[Live Demo](https://credit-risk-analysis-model.streamlit.app/)**

##  Project Overview
Evaluating credit risk accurately is essential for financial institutions to minimize default rates while continuing to offer loans to qualified candidates. This project utilizes an **XGBoost Classifier** to analyze applicant demographics, financial habits, and loan parameters to assess creditworthiness.

####  Key Features
* **Interactive UI:** Simple form inputs for key financial and demographic metrics.
* **Pre-trained Encoders:** Saved scikit-learn Label Encoders to handle categorical variables on the fly.
* **Real-time Predictions:** Instant inference powering a clear risk assessment output.
* **Cloud Deployed:** Publicly accessible via Streamlit Community Cloud.

####   Tech Stack & Tools

* **Language:** Python 3.12+
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Joblib
* **Deployment & Hosting:** Streamlit, Streamlit Community Cloud, Git/GitHub
* **Development Environment:** Google Colab

####   Dataset & Features
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **Age** | Numeric | Age of applicant (18–80) |
| **Sex** | Categorical | Gender (`male`, `female`) |
| **Job** | Categorical | Skill/job classification level (0–3) |
| **Housing** | Categorical | Housing status (`own`, `rent`, `free`) |
| **Saving accounts**| Categorical | Savings balance (`little`, `moderate`, `rich`, `quite rich`) |
| **Checking account**| Categorical | Checking balance (`little`, `moderate`, `rich`) |
| **Credit amount** | Numeric | Requested credit/loan amount |
| **Duration** | Numeric | Loan duration in months |

##  Exploratory Data Analysis (EDA)

Before building the predictive model, an Exploratory Data Analysis was conducted to identify key risk drivers and understand how demographic and financial features correlate with credit default.

### Key Insights & Findings

1. **Checking Account Status is the Strongest Risk Indicator**
   * Applicants with **little or no checking account balance** exhibited a significantly higher proportion of high-risk classifications compared to those with moderate or rich balances.
   * *Takeaway:* Liquid account balance serves as an immediate proxy for short-term financial stability.

2. **Loan Duration & Amount vs. Risk**
   * **Higher Credit Amounts & Longer Terms:** Risk increases noticeably as loan duration exceeds 24 months and credit amounts rise.
   * **Distribution:** "Bad" credit risks generally had a higher median loan duration (approx. 24–36 months) compared to "Good" credit risks (approx. 12–18 months).

3. **Age & Demographics**
   * Younger applicants (under 25) showed a slightly higher probability of being classified as bad credit risk, likely tied to shorter credit histories and lower savings balances.
   * Credit risk stabilized significantly for applicants in middle-to-older age brackets with established housing (`own`).

4. **Housing & Stability**
   * Applicants who **owned their homes** had a higher proportion of "Good" credit predictions compared to those living in rented or free housing, making housing status a reliable marker for financial backing.

---

### Summary Visualizations (Key Feature Distributions)

| Feature | Pattern Observed in EDA | Impact on Model Risk |
| :--- | :--- | :--- |
| **Duration** | Long duration $\rightarrow$ Increased risk | High |
| **Checking Account** | Low/No balance $\rightarrow$ High default rate | High |
| **Credit Amount** | Skewed right; higher amounts correlate with bad risk | Medium-High |
| **Age** | Right-skewed; younger borrowers carry slightly higher risk | Medium |
| **Housing Status** | Homeowners exhibit lower default rates | Medium |

## Data Preprocessing & Feature Engineering

To prepare the **[Dataset](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)** for the XGBoost classification model, the raw data went through a structured preprocessing pipeline to ensure high data quality and correct encoding for machine learning inference.

### 1. Handling Missing Values (Data Cleaning)
* **Strategy:** Complete Case Analysis (Dropping Nulls).
* **Rationale:** Missing values in critical financial attributes (such as `Checking account` and `Saving accounts`) were removed to prevent data distortion and ensure the model was trained strictly on complete applicant profiles.

### 2. Feature Encoding & Transformation
XGBoost requires all numerical input features. Categorical variables were processed using `LabelEncoder` from `scikit-learn`:

* **Categorical Features Encoded:**
  * `Sex` $\rightarrow$ Encoded to numerical values (`female`: 0, `male`: 1).
  * `Housing` $\rightarrow$ Mapped categories (`free`, `own`, `rent`).
  * `Saving accounts` $\rightarrow$ Mapped categories (`little`, `moderate`, `quite rich`, `rich`).
  * `Checking account` $\rightarrow$ Mapped categories (`little`, `moderate`, `rich`).

> **Production Note:** Each `LabelEncoder` object was individually serialized using `joblib` into dedicated `.pkl` files (e.g., `Sex_encoder.pkl`, `Housing_encoder.pkl`). This ensures that user inputs from the Streamlit UI are transformed using the exact same mapping parameters used during training.

### 3. Final Feature Set
The processed dataset supplied to the model consists of 8 key predictor variables:

| Feature Name | Feature Type | Preprocessing Applied |
| :--- | :--- | :--- |
| **Age** | Numeric | Pass-through (no transformation required) |
| **Sex** | Categorical | Encoded via `Sex_encoder.pkl` |
| **Job** | Numeric/Ordinal | Pass-through (0–3 scale) |
| **Housing** | Categorical | Encoded via `Housing_encoder.pkl` |
| **Saving accounts** | Categorical | Encoded via `Saving accounts_encoder.pkl` |
| **Checking account** | Categorical | Encoded via `Checking account_encoder.pkl` |
| **Credit amount** | Numeric | Pass-through |
| **Duration** | Numeric | Pass-through (months) |

## 📊 Model Evaluation & Performance

The **XGBoost Classifier** was evaluated on a held-out test set to assess its ability to differentiate between "Good" and "Bad" credit risks. Because predicting credit risk involves balancing loan approval rates against potential defaults, the model was evaluated across multiple classification metrics.

### Performance Summary

| Metric | Score | Business Meaning |
| :--- | :--- | :--- |
| **Accuracy** | ~78% | The overall percentage of credit risk predictions the model got right. |















