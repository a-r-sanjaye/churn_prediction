import predict
import traceback

data = {
    'gender': 'Female',
    'SeniorCitizen': '0',
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': '1',
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': '29.85',
    'TotalCharges': '29.85'
}

print("Testing prediction with mock data...")
try:
    pred, prob = predict.predict_churn(data)
    print(f"✅ Prediction success! Result: {pred}, Probability: {prob:.4f}")
except Exception:
    print("❌ Prediction failed.")
    traceback.print_exc()
