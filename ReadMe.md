# 📉 Customer Churn Prediction Dashboard

An end-to-end machine learning application that predicts customer churn risk and provides actionable insights through an interactive dashboard.

---

## 🚀 Overview

Customer churn is a critical business problem where companies lose customers over time. This project builds a complete ML system to:

* Predict which customers are likely to churn
* Quantify risk using probability scores
* Help businesses take preventive actions
* Explain *why* customers are likely to leave

---

## 🧠 Key Features

### 🔹 Batch Prediction

* Upload customer data (CSV)
* Get churn probability for each customer
* Automatically ranked by risk level

### 🔹 Real-Time Prediction

* Simulate individual customer profiles
* Instantly see churn probability

### 🔹 Risk Segmentation

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

---

### 🔹 Explainable AI (SHAP)

* Identifies key drivers of churn
* Shows feature impact on predictions

---

### 🔹 Model Evaluation Metrics

* **AUC Score:** ~0.82
* **Precision / Recall / F1:** Computed dynamically in the dashboard (based on uploaded labeled data and selected threshold)

---

## ⚙️ How It Works

```id="l8o3c3"
Customer Data
   ↓
Preprocessing & Feature Engineering
   ↓
Scaling
   ↓
XGBoost Model
   ↓
Churn Probability
   ↓
Risk Classification + Visualization
```

---

## 🧩 Feature Engineering Highlights

* Charges per tenure (spending behavior)
* Total-to-monthly ratio (billing consistency)
* Number of services (customer engagement)
* Contract type (churn risk indicator)

---

## 🛠️ Tech Stack

* Python
* XGBoost
* Scikit-learn
* Streamlit
* SHAP
* Pandas, NumPy
* Plotly, Matplotlib

---

## 📂 Project Structure

```id="jwj3b5"
churn-project/
│
├── app/
│   └── dashboard.py
│
├── models/
│   ├── xgboost_churn.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── metrics.json
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

```id="4ox7rf"
git clone https://github.com/YOUR_USERNAME/churn-prediction-dashboard.git
cd churn-prediction-dashboard
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## 📌 Important Note

* Model is trained on **Telco Customer Churn dataset**
* Works best with similar feature schema
* Performance may vary for different data distributions
* Retraining is required for other domains

---

## 🎯 Business Value

* Identify high-risk customers early
* Enable targeted retention strategies
* Reduce customer churn and revenue loss
* Support explainable, data-driven decisions

---

## 📈 Future Improvements

* API deployment (FastAPI)
* Cloud hosting
* Model monitoring and drift detection
* Automated retraining pipeline

---

## 👨‍💻 Author

RANGANATH RANGAM

* B.Tech in DATA SCIENCE (2025)
* Interested in Machine Learning, Deep Learning, and Generative AI
* Skilled in Python, Data Science, and building real-world AI applications
* Focused on developing end-to-end intelligent systems
  

🔗 GitHub: https://github.com/ranganath18/Customer_Churn_Prediction_DashBoard/
🔗 LinkedIn: www.linkedin.com/in/ranganath-rangam-49a2a324a

---

## ⭐ Summary

A complete machine learning system that combines prediction, explainability, and business decision support in a single interactive application.
