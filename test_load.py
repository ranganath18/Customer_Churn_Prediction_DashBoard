# test_load.py
# This script verifies all your Colab-trained files loaded correctly
# before you start building the Streamlit dashboard

import joblib
import json
import numpy as np

print("Step 1: Loading model...")
model = joblib.load("models/xgboost_churn.pkl")
print(f"  Done. Model type: {type(model).__name__}")

print("\nStep 2: Loading scaler...")
scaler = joblib.load("models/scaler.pkl")
print(f"  Done. Expects {scaler.n_features_in_} features as input.")

print("\nStep 3: Loading feature names...")
with open("models/feature_names.json") as f:
    feature_names = json.load(f)
print(f"  Done. Found {len(feature_names)} feature names.")

print("\nStep 4: Loading saved metrics...")
with open("models/metrics.json") as f:
    metrics = json.load(f)
print(f"  Done. AUC-ROC from training was: {metrics['auc']}")

print("\nStep 5: Running a dummy prediction to confirm end-to-end...")
dummy_input  = np.zeros((1, len(feature_names)))
dummy_scaled = scaler.transform(dummy_input)
pred_proba   = model.predict_proba(dummy_scaled)[0][1]
print(f"  Done. Dummy prediction returned: {pred_proba:.4f}")

print("\n All files loaded successfully. You are ready to build Streamlit!")