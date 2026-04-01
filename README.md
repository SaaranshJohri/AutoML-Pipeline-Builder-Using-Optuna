# 🚀 AutoML Pipeline Builder using Optuna

An end-to-end **Automated Machine Learning (AutoML)** system that intelligently optimizes machine learning pipelines using **Optuna**, including preprocessing, model selection, and hyperparameter tuning — all through an interactive **Streamlit UI**.

---

## 🧠 Overview

This project eliminates manual trial-and-error in ML workflows by automatically discovering the **best-performing pipeline** for a given dataset.

It handles:
- Data preprocessing (missing values, encoding, scaling)
- Model selection (Random Forest, XGBoost, Logistic Regression)
- Hyperparameter optimization (Optuna)
- Visualization and analysis
- Interactive user interface

---

## ⚙️ Key Features

### 🔹 Automated Pipeline Optimization
End-to-end pipeline:

Imputation → Encoding → Scaling → Model → Evaluation

- Dynamically constructed using Optuna

---

### 🔹 Hyperparameter Tuning
- Uses Optuna’s efficient search strategies (TPE sampler)
- Supports:
  - Random Forest
  - XGBoost
  - Logistic Regression

---

### 🔹 Intelligent Preprocessing
- Numerical features:
  - Mean imputation
  - Optional scaling
- Categorical features:
  - Mode imputation
  - One-hot encoding
- Target encoding (LabelEncoder for classification)

---

### 🔹 Interactive Streamlit UI
- Upload dataset (CSV)
- Select target column
- Run AutoML with one click
- View results instantly

---

### 🔹 Visualization Dashboard
- Optimization history
- Parameter importance
- Parallel coordinate plots

---

### 🔹 Robust Data Handling
Handles:
- Missing values
- Mixed data types
- Categorical targets

---

## 📊 Tech Stack

| Category | Tools |
|--------|------|
| ML Framework | scikit-learn |
| Optimization | Optuna |
| Models | RandomForest, XGBoost, Logistic Regression |
| UI | Streamlit |
| Visualization | Plotly |
| Data | Pandas, NumPy |

---

## 📁 Project Structure
AutoML-Pipeline-Builder/
│
├── src/
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── models.py
│ ├── objective.py
│ ├── optimizer.py
│
├── results/
│
├── app.py
├── requirements.txt
├── README.md

---

## 🚀 How It Works

1. Upload dataset via UI  
2. Select target column  
3. System:
   - Detects feature types
   - Builds preprocessing pipeline  
4. Optuna:
   - Explores models & hyperparameters  
   - Evaluates using cross-validation  
5. Best pipeline is selected and displayed  

---
