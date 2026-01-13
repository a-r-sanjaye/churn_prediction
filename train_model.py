import pandas as pd
import joblib
import os
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode categorical variables and save encoders
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare X and y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save feature names for prediction order alignment
model_columns = list(X.columns)
joblib.dump(model_columns, "model/model_columns.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, _, y_train, _ = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

# Explain Model
explainer = shap.TreeExplainer(model)

# Save Artifacts
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(explainer, "model/shap_explainer.pkl")

print("✅ Model, Scaler, Encoders & SHAP Explainer saved")
