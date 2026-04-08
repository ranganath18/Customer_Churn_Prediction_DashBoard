# app/dashboard.py
# Streamlit dashboard for Customer Churn Prediction
# Loads pre-trained model from Colab — no retraining happens here

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Streamlit + matplotlib
from sklearn.metrics import precision_score, recall_score, f1_score
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the very first Streamlit command in the script — nothing can come before it
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# We use @st.cache_resource so these heavy files are loaded ONCE when the app
# starts and then cached in memory. Without this, every time a user moves a
# slider or clicks a button, Streamlit would reload the entire 800KB model
# from disk — making the app feel very slow.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("models/xgboost_churn.pkl")
    scaler       = joblib.load("models/scaler.pkl")
    
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    
    return model, scaler, feature_names, metrics

model, scaler, feature_names, metrics = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — USER CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")
    
    st.subheader("Decision Threshold")
    threshold = st.slider(
        label="Churn probability cutoff",
        min_value=0.10,
        max_value=0.90,
        value=0.40,         # Default 0.40 is better than 0.50 for imbalanced churn data
        step=0.05,
        help="Lower = catch more churners but flag more false alarms. "
             "Higher = more precise but miss some real churners."
    )
    
    st.markdown("---")
    st.subheader("Model Info")
    
    # Display the metrics saved from Colab training
    # This way the app shows real training results without needing to retrain
    st.metric("AUC-ROC (Test Set)", f"{metrics['auc']:.4f}")
    st.metric("Features Used",      metrics['n_features'])
    st.metric("Test Set Size",      metrics['n_test'])
    st.caption("Model trained in Google Colab with XGBoost + SMOTE.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown(
    "This dashboard uses a pre-trained **XGBoost model** to predict which customers "
    "are likely to churn. Predictions are explained using **SHAP values** so you can "
    "see *why* the model flagged each customer, not just *that* it flagged them."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TAB LAYOUT
# Splitting the app into tabs keeps it organised and loads each section
# only when the user clicks on it — better performance than showing everything at once
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Batch Prediction (CSV Upload)",
    "👤 Single Customer Prediction",
    "🔍 SHAP Explainability"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — BATCH PREDICTION
# User uploads a CSV of customers and gets churn probabilities for all of them
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload Customer Data for Batch Scoring")
    st.markdown(
        "Upload the raw Telco CSV file. The app will clean, encode, and score "
        "every customer automatically."
    )
    
    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=["csv"],
        help="Upload the IBM Telco Customer Churn CSV"
    )
    
    if uploaded_file:
        
        # ── Data Loading and Preprocessing ──────────────────────────────────
        @st.cache_data  # Cache the processed data so re-running doesn't redo this work
        def preprocess_uploaded(file):
            df = pd.read_csv(file)
            
            # Save customerID separately to display in results table
            customer_ids = df["customerID"] if "customerID" in df.columns else pd.Series(range(len(df)))
            
            # Store the actual churn labels if present (for evaluation)
            actual_churn = None
            if "Churn" in df.columns:
                actual_churn = (df["Churn"] == "Yes").astype(int)
            
            # ── Same cleaning pipeline as Colab ─────────────────────────────
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(0)
            df = df.drop(columns=["customerID"], errors="ignore")
            df = df.drop(columns=["Churn"],      errors="ignore")
            
            # ── Feature engineering (same as Colab) ─────────────────────────
            service_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                            "TechSupport","StreamingTV","StreamingMovies"]
            df["Charges_per_tenure"]     = df["MonthlyCharges"] / (df["tenure"] + 1)
            df["Total_to_monthly_ratio"] = df["TotalCharges"]   / (df["MonthlyCharges"] + 1)
            df["Num_services"]           = df[service_cols].apply(
                lambda row: sum(v == "Yes" for v in row), axis=1
            )
            df["Is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
            
            # ── Encoding (same as Colab) ─────────────────────────────────────
            from sklearn.preprocessing import LabelEncoder
            binary_cols = ["gender","Partner","Dependents","PhoneService",
                           "PaperlessBilling","MultipleLines"]
            for col in binary_cols:
                if col in df.columns:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
            multi_cols = ["InternetService","OnlineSecurity","OnlineBackup",
                          "DeviceProtection","TechSupport","StreamingTV",
                          "StreamingMovies","Contract","PaymentMethod"]
            df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
            
            # ── Critical step: align columns to training feature order ───────
            # This is why we saved feature_names.json in Colab.
            # reindex adds any missing columns as 0 and drops any extra columns,
            # ensuring the DataFrame perfectly matches what the model expects.
            df = df.reindex(columns=feature_names, fill_value=0)
            
            return df, customer_ids, actual_churn
        
        df_processed, customer_ids, actual_churn = preprocess_uploaded(uploaded_file)
        
        # ── Scale and Predict ────────────────────────────────────────────────
        # Pass as DataFrame (not numpy array) to avoid the feature name warning
        df_scaled        = pd.DataFrame(
            scaler.transform(df_processed),
            columns=feature_names   # This eliminates the warning from test_load.py
        )
        churn_proba      = model.predict_proba(df_scaled)[:, 1]
        churn_prediction = (churn_proba >= threshold).astype(int)
        
        # ── Results Table ────────────────────────────────────────────────────
        results_df = pd.DataFrame({
            "Customer ID":        customer_ids.values,
            "Churn Probability":  churn_proba.round(3),
            "Risk Level":         pd.cut(
                churn_proba,
                bins=[0, 0.35, 0.65, 1.0],
                labels=["🟢 Low", "🟡 Medium", "🔴 High"]
            ),
            "Flagged as Churner": churn_prediction.astype(bool)
        })
        
        if actual_churn is not None:
            results_df["Actual Churn"] = actual_churn.values.astype(bool)
        
        results_df = results_df.sort_values("Churn Probability", ascending=False)
        # ── Performance Metrics (only if actual labels available) ──
        precision = recall = f1 = None

        if actual_churn is not None:
            precision = precision_score(actual_churn, churn_prediction)
            recall    = recall_score(actual_churn, churn_prediction)
            f1        = f1_score(actual_churn, churn_prediction)
        # ── Summary Metrics ──────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers",    len(results_df))
        col2.metric("Flagged At-Risk",    churn_prediction.sum())
        col3.metric("High Risk (>65%)",   (churn_proba >= 0.65).sum())
        col4.metric("Avg Churn Prob",     f"{churn_proba.mean():.1%}")
        # ── Show evaluation metrics only if actual labels exist ──
        if actual_churn is not None:
            st.markdown("### 📊 Model Performance (Based on Uploaded Data)")
            
            col5, col6, col7 = st.columns(3)
            col5.metric("Precision", f"{precision:.2f}")
            col6.metric("Recall",    f"{recall:.2f}")
            col7.metric("F1 Score",  f"{f1:.2f}")
        st.markdown("---")
        
        # ── Probability Distribution Chart ───────────────────────────────────
        st.subheader("Churn Probability Distribution")
        fig_dist = px.histogram(
            x=churn_proba,
            nbins=40,
            labels={"x": "Churn Probability", "y": "Number of Customers"},
            color_discrete_sequence=["#3B8BD4"]
        )
        fig_dist.add_vline(
            x=threshold, line_dash="dash", line_color="red",
            annotation_text=f"Threshold = {threshold}"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # ── Sortable Results Table ───────────────────────────────────────────
        st.subheader("Customer Risk Rankings")
        st.dataframe(results_df, use_container_width=True, height=400)
        
        # ── Download Button ──────────────────────────────────────────────────
        csv_output = results_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_output,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE CUSTOMER PREDICTION
# User manually fills in one customer's details using sliders and dropdowns
# This is great for demos — you can show a recruiter live predictions
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Predict Churn for a Single Customer")
    st.markdown("Adjust the customer profile below and see the churn probability update instantly.")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**Account Info**")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges   = st.number_input("Total Charges ($)",
                                           min_value=0.0,
                                           value=float(tenure * monthly_charges))
        contract        = st.selectbox("Contract Type",
                                        ["Month-to-month", "One year", "Two year"])
        payment_method  = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col_b:
        st.markdown("**Demographics**")
        gender      = st.selectbox("Gender",      ["Male", "Female"])
        senior      = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner     = st.selectbox("Partner",     ["Yes", "No"])
        dependents  = st.selectbox("Dependents",  ["No", "Yes"])
    
    with col_c:
        st.markdown("**Services**")
        phone_service   = st.selectbox("Phone Service",    ["Yes", "No"])
        internet        = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        online_backup   = st.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
        device_prot     = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
        streaming_movies= st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        multiple_lines  = st.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])
        tech_support    = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        streaming_tv    = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
        paperless       = st.selectbox("Paperless Billing",["Yes", "No"])
    
    # ── Build a single-row DataFrame matching the training pipeline ──────────
    # This manually reconstructs what your Colab preprocessing pipeline did
    single_customer = pd.DataFrame([{
        "tenure":          tenure,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    total_charges,
        "SeniorCitizen":   1 if senior == "Yes" else 0,
        "gender":          1 if gender == "Male" else 0,
        "Partner":         1 if partner == "Yes" else 0,
        "Dependents":      1 if dependents == "Yes" else 0,
        "PhoneService":    1 if phone_service == "Yes" else 0,
        "PaperlessBilling":1 if paperless == "Yes" else 0,
        "MultipleLines":   1 if multiple_lines == "Yes" else 0,
        # Engineered features
        "Charges_per_tenure":     monthly_charges / (tenure + 1),
        "Total_to_monthly_ratio": total_charges   / (monthly_charges + 1),
        "Num_services":           sum([
            online_security == "Yes", tech_support == "Yes", streaming_tv == "Yes"
        ]),
        "Is_month_to_month": 1 if contract == "Month-to-month" else 0,
        # One-hot encoded columns — set to 1 if this customer matches, else 0
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No":          1 if internet == "No" else 0,
        "Contract_One year":           1 if contract == "One year" else 0,
        "Contract_Two year":           1 if contract == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)":    1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check":           1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check":               1 if payment_method == "Mailed check" else 0,
        "OnlineSecurity_No internet service":       1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes":                       1 if online_security == "Yes" else 0,
        "TechSupport_No internet service":          1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes":                          1 if tech_support == "Yes" else 0,
        "StreamingTV_No internet service":          1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes":                          1 if streaming_tv == "Yes" else 0,
        # Missing columns from feature_names.json — filled here to avoid silent 0-fill
        "OnlineBackup_No internet service":         1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes":                         1 if online_backup == "Yes" else 0,
        "DeviceProtection_No internet service":     1 if device_prot == "No internet service" else 0,
        "DeviceProtection_Yes":                     1 if device_prot == "Yes" else 0,
        "StreamingMovies_No internet service":      1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes":                      1 if streaming_movies == "Yes" else 0,
    }])
    
    # Align to exact feature order from training
    single_customer = single_customer.reindex(columns=feature_names, fill_value=0)
    
    # Scale using the fitted scaler (as DataFrame to avoid warning)
    single_scaled = pd.DataFrame(
        scaler.transform(single_customer),
        columns=feature_names
    )
    
    prob = model.predict_proba(single_scaled)[0][1]
    
    # ── Display Result ───────────────────────────────────────────────────────
    st.markdown("---")
    
    if prob >= 0.65:
        st.error(f"🔴 High Churn Risk — {prob:.1%} probability")
    elif prob >= 0.35:
        st.warning(f"🟡 Medium Churn Risk — {prob:.1%} probability")
    else:
        st.success(f"🟢 Low Churn Risk — {prob:.1%} probability")
    
    # Gauge chart — visually satisfying for demos
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        title={"text": "Churn Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#E24B4A" if prob >= 0.65 else
                              "#EF9F27" if prob >= 0.35 else "#639922"},
            "steps": [
                {"range": [0, 35],  "color": "#EAF3DE"},
                {"range": [35, 65], "color": "#FAEEDA"},
                {"range": [65, 100],"color": "#FCEBEB"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": threshold * 100
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP EXPLAINABILITY
# Shows which features are driving the model's predictions globally
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("SHAP Feature Importance — What Drives Churn?")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows how much each feature "
        "contributed to the model's predictions. Features on the right push "
        "the prediction **toward churn**. Features on the left push it **away from churn**."
    )
    
    shap_file = st.file_uploader(
        "Upload the same CSV again to generate SHAP plots",
        type=["csv"],
        key="shap_uploader"
    )
    
    if shap_file:
        with st.spinner("Computing SHAP values — this takes about 30 seconds..."):
            
            df_shap = pd.read_csv(shap_file)
            
            # Same preprocessing as Tab 1
            df_shap["TotalCharges"] = pd.to_numeric(df_shap["TotalCharges"], errors="coerce")
            df_shap["TotalCharges"] = df_shap["TotalCharges"].fillna(0)
            df_shap = df_shap.drop(columns=["customerID", "Churn"], errors="ignore")
            
            service_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                            "TechSupport","StreamingTV","StreamingMovies"]
            df_shap["Charges_per_tenure"]     = df_shap["MonthlyCharges"] / (df_shap["tenure"] + 1)
            df_shap["Total_to_monthly_ratio"] = df_shap["TotalCharges"]   / (df_shap["MonthlyCharges"] + 1)
            df_shap["Num_services"]           = df_shap[service_cols].apply(
                lambda row: sum(v == "Yes" for v in row), axis=1
            )
            df_shap["Is_month_to_month"] = (df_shap["Contract"] == "Month-to-month").astype(int)
            
            from sklearn.preprocessing import LabelEncoder
            for col in ["gender","Partner","Dependents","PhoneService","PaperlessBilling","MultipleLines"]:
                if col in df_shap.columns:
                    df_shap[col] = LabelEncoder().fit_transform(df_shap[col].astype(str))
            
            multi_cols = ["InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
                          "TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]
            df_shap = pd.get_dummies(df_shap, columns=multi_cols, drop_first=True)
            df_shap = df_shap.reindex(columns=feature_names, fill_value=0)
            
            # Use a sample of 200 rows for speed — SHAP on all 7000 rows is slow
            sample     = df_shap.sample(min(200, len(df_shap)), random_state=42)
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(sample)
            
            # Global importance bar chart
            # NOTE: shap.summary_plot() does NOT accept an `ax` parameter —
            # it creates its own figure internally. We call plt.gcf() AFTER
            # the plot is drawn to grab the figure shap created, then pass
            # that to st.pyplot(). Using matplotlib_figure_size via
            # plt.rcParams is the correct way to control size here.
            st.subheader("Top 15 Most Influential Features")
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                shap_vals, sample,
                plot_type="bar",
                max_display=15,
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
            
            # Dot plot — shows direction of each feature's effect
            st.subheader("Feature Impact Direction")
            st.markdown(
                "Red dots = high feature value. Blue dots = low feature value. "
                "Right of center = pushes toward churn."
            )
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                shap_vals, sample,
                plot_type="dot",
                max_display=15,
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()