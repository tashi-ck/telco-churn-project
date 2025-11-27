# 📞 Telco Customer Churn Prediction

### Machine Learning + Streamlit Web App

A complete end-to-end Data Science project that predicts whether a
telecom customer will **churn** using machine learning.\
This project includes **data cleaning, EDA, feature engineering, model
training, evaluation, and a deployed Streamlit web app**.

------------------------------------------------------------------------

## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by telecom
companies. In this project, we:

-   Clean and preprocess the **Telco Customer Churn dataset**\
-   Perform **Exploratory Data Analysis (EDA)**\
-   Build & evaluate multiple ML models\
-   Choose the best-performing model (Logistic Regression / Random
    Forest / XGBoost)\
-   Save model using `pickle`\
-   Develop a **Streamlit Web App** for real-time churn prediction\
-   Prepare the repository for deployment (requirements, structure,
    documentation)

------------------------------------------------------------------------

## 📂 Project Structure

    telco-churn-project/
    │── app/
    │   └── app.py                 # Streamlit UI
    │
    │── models/
    │   ├── churn_model.pkl        # Trained ML model
    │   ├── columns.pkl
    │   └── scaler.pkl             # Scaler (if used)
    │
    │── notebooks/
    │   ├── 01_data_cleaning.ipynb    # Notebook 1: Data cleaning + EDA
    │   ├── 02_feature_engineering.ipynb 
    │   └── 03_model_training.ipynb   # Notebook 2: Model training + evaluation
    │
    │── data/
    │   ├── telco_churn.csv
    │   ├── telco_churn_cleaned.csv
    │   ├── X_test.csv
    │   ├── X_train.csv
    │   ├── y_test.csv
    │   └── y_train.csv
    │
    │── requirements.txt           
    │── README.md

------------------------------------------------------------------------

## 🧼 1. Data Cleaning & Preprocessing

Handled using **Notebook 1 (data_cleaning.ipynb)**:

✔ Handle missing values\
✔ Remove incorrect values\
✔ Convert categorical → numeric\
✔ Feature engineering\
✔ Balance dataset (SMOTE optional)\
✔ Export cleaned dataset

------------------------------------------------------------------------

## 📊 2. Exploratory Data Analysis (EDA)

Performed using **seaborn**, **matplotlib**, **pandas profiling**:

-   Churn distribution\
-   Contract type vs Churn\
-   Tenure distribution\
-   Monthly Charges comparison\
-   Service usage patterns\
-   Correlation heatmap

------------------------------------------------------------------------

## 🤖 3. Model Training

Built and compared:

-   Logistic Regression\
-   Random Forest\
-   Gradient Boosting\

Final chosen model exported as:

    models/churn_model.pkl
    models/scaler.pkl

------------------------------------------------------------------------

## 🌐 4. Streamlit Web App

Run Streamlit App:

``` bash
cd app
streamlit run app.py
```

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/tashi-ck/telco-churn-project.git
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd app
streamlit run app.py
```

------------------------------------------------------------------------

## 📁 Dataset

Telco Customer Churn Dataset (IBM)\
Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn
