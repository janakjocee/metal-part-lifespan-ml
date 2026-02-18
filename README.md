# 🚀 Turbine Component Lifespan Prediction using Ensemble Learning & NLP

## 📌 Project Overview
This project presents a complete end-to-end **Machine Learning solution** for predicting and classifying turbine component lifespan using synthetic manufacturing data.

The objective was to:
* **Predict continuous lifespan** (Regression)
* **Classify components into quality tiers** (Binary & Multi-Class Classification)
* **Compare models rigorously** and provide a deployment recommendation

The project strictly follows a structured ML pipeline including data exploration, preprocessing, feature engineering, hyperparameter tuning, model evaluation, and interpretability.

---

## 🏭 Business Context
In industrial manufacturing, destructive lifespan testing is expensive and time-consuming. This project demonstrates how Machine Learning can:
* **Reduce** destructive testing costs
* **Improve** predictive maintenance
* **Identify** key manufacturing drivers
* **Support** data-driven production optimization

---

## 📊 Dataset Description
* **Size:** 1,000 rows × 16 columns (Synthetic turbine manufacturing dataset)
* **Target Variable:** `Lifespan` (continuous, hours)

| Category | Features |
| :--- | :--- |
| **Numerical** | coolingRate, quenchTime, forgeTime, HeatTreatTime, Nickel%, Iron%, Cobalt%, Chromium%, smallDefects, largeDefects, sliverDefects |
| **Categorical** | partType, microstructure, seedLocation, castType |

---

## 🔎 Data Exploration & Preprocessing
* **✔ Outlier Handling:** Interquartile Range (IQR) capping to prevent tree overfitting while preserving all samples.
* **✔ Encoding & Scaling:** One-Hot Encoding for categorical variables and `StandardScaler` for numeric features.
* **✔ Data Splitting:** 80/20 Train-Test split (`random_state=42`) with stratified split for classification.
* **✔ Class Imbalance Handling:** Binary class imbalance (29% ≥1500 hours) addressed using **SMOTE** (applied on training data only).

---

## 🧠 Regression Modelling

### **Models Implemented**
1.  Linear Regression (Baseline)
2.  Random Forest Regressor
3.  Gradient Boosting Regressor
4.  **NLP + Numeric Hybrid Model** (TF-IDF + Gradient Boosting)

### **Best Numeric Model**
> **Tuned Gradient Boosting Regressor**
> * **R²:** ≈ 0.984
> * **RMSE:** ≈ 43.5
> * **MAE:** ≈ 35.0

### **NLP-Enhanced Regression**
* **Method:** Concatenated categorical columns into a text feature $\rightarrow$ Applied TF-IDF vectorization (100 terms) $\rightarrow$ Combined with scaled numeric features using `ColumnTransformer`.
* **Result:** **R² ≈ 0.990** | **RMSE ≈ 35.0** | **MAE ≈ 26.8**
* **Insight:** This hybrid approach improved predictive power by capturing latent relationships between categorical combinations.

---

## 🔍 Regression Interpretability
Using **SHAP (SHapley Additive exPlanations)**, the key drivers identified were:
1.  **coolingRate** (High rate increases lifespan)
2.  **Nickel%** (High percentage increases lifespan)
3.  **forgeTime / HeatTreatTime**
4.  **largeDefects** (Higher defect counts significantly reduce lifespan)

---

## 🎯 Classification Modelling

### **🔹 Binary Classification (≥ 1500 Hours)**
* **Label 1:** Lifespan ≥ 1500 | **Label 0:** Lifespan < 1500
* **Best Model:** **Tuned AdaBoost + SMOTE**
* **Metrics:** Accuracy ≈ 0.95 | Macro F1 ≈ 0.95 | ROC-AUC ≈ 0.97

### **🔹 Three-Class Classification**
Two strategies were implemented:
1.  **Quantile-Based (33/33/33 Split):** CatBoost Macro F1 ≈ 0.87
2.  **KMeans-Based Grouping (k=3):** Unsupervised clustering on numeric data.
    * **Best Performance:** **CatBoost + KMeans Labels**
    * **Macro F1:** ≈ 0.985 | **ROC-AUC:** ≈ 1.00

---

## 📈 Model Comparison Summary

| Task | Best Model | Key Metric | Performance |
| :--- | :--- | :--- | :--- |
| **Regression (Numeric)** | Gradient Boosting | $R^2$ | 0.984 |
| **Regression (Hybrid NLP)**| GB + TF-IDF | $R^2$ | 0.990 |
| **Binary Classification** | AdaBoost + SMOTE | Macro F1 | 0.95 |
| **Three-Class Classification**| CatBoost + KMeans | Macro F1 | 0.985 |

---

## ⚖️ Critical Evaluation

**Strengths:**
* Rigorous hyperparameter tuning and SMOTE-based imbalance correction.
* Innovative Hybrid NLP + Numeric modelling.
* SHAP-based explainability for "Black Box" models.

**Limitations:**
* Synthetic dataset lacks real-world noise/stochasticity.
* TF-IDF ignores word order semantics (could be improved with Embeddings).
* KMeans boundaries may shift as new manufacturing data is collected.

---

## 🏁 Final Recommendations
1.  **Deploy Gradient Boosting + NLP hybrid** for lifespan prediction ($R^2 \approx 0.990$).
2.  **Use AdaBoost + SMOTE** for binary safety classification.
3.  **Monitor primary production drivers:** `coolingRate`, `Nickel%`, `forgeTime`, `HeatTreatTime`, and `defect counts`.

---

## 🛠 Tech Stack
* **Data:** Pandas, NumPy
* **ML:** Scikit-learn, XGBoost, CatBoost
* **Viz/Interpret:** Matplotlib, Seaborn, SHAP
* **Imbalance:** imbalanced-learn (SMOTE)

---

## 📂 Repository Structure
```text
turbine-lifespan-ml/
│
├── notebook/
│   ├── turbine_lifespan_model.ipynb
│   └── turbine_lifespan_model.pdf
│
├── report/
│   └── COMP1801_ML_Report.pdf
│
├── images/
│   └── (plots, confusion matrices, SHAP visuals)
│
├── requirements.txt
│
└── README.md
