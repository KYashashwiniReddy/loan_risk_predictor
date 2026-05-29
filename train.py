import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load Dataset
df = pd.read_csv("loan_data.csv")

# Fill Missing Values
df = df.fillna(df.mode().iloc[0])

# Remove LoanID (not useful for prediction)
if "LoanID" in df.columns:
    df = df.drop("LoanID", axis=1)

# Encode Categorical Columns
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Target Column
target = "Default"

# Features and Target
X = df.drop(target, axis=1)
y = df[target]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for Feature Reduction
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression (Loan Approval / Rejection)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Random Forest (Risk Prediction)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

# K-Means (Customer Segmentation)
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
kmeans.fit(X_pca)

# Save Models
#joblib.dump(log_model, "loan_model.pkl")
#joblib.dump(rf_model, "risk_model.pkl")
#joblib.dump(kmeans, "kmeans.pkl")
#joblib.dump(scaler, "scaler.pkl")
#joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "encoders.pkl")
print("All Models Saved Successfully!")