import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = "model/churn_model.pkl"
SCALER_PATH = "model/scaler.pkl"
SHAP_EXPLAINER_PATH = "model/shap_explainer.pkl"
ENCODER_PATH = "model/label_encoders.pkl"
COLUMNS_PATH = "model/model_columns.pkl"

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_model.py first.")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    explainer = joblib.load(SHAP_EXPLAINER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    
    return model, scaler, explainer, encoders, model_columns

def predict_churn(data_dict):
    """
    Predict churn from a dictionary of raw input values.
    """
    model, scaler, _, encoders, model_columns = load_artifacts()

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([data_dict])

    # Preprocess the input data
    # 1. Numerics
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

    # 2. Categoricals
    for col, encoder in encoders.items():
        if col in input_df.columns:
            # Handle unknown labels gracefully (though ideally UI validation prevents this)
            input_df[col] = input_df[col].apply(lambda x: transform_label(x, encoder))

    # 3. Ensure columns match training order using model_columns
    # Add missing columns with 0 if any (fragile safeguard)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0 # Default value
            
    input_df = input_df[model_columns]

    # Scale
    data_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]
    
    return pred, prob

def transform_label(value, encoder):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Fallback for unseen labels: use most frequent (mode) or 0
        return 0
