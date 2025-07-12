import zipfile
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data from ZIP file
with zipfile.ZipFile("mobile_phone_pricing.zip", "r") as z:
    with z.open("Mobile Phone Pricing/dataset.csv") as f:
        data = pd.read_csv(f)

# Check column names 
print("Columns in dataset:")
print(data.columns.tolist())

print(data.head())
print(data.info())
print(data['price_range'].value_counts())

# Check for nulls
print("\nMissing values:\n", data.isnull().sum())

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    data.drop_duplicates(inplace=True)

# Features and target
X = data.drop('price_range', axis=1)
y = data['price_range']

# EDA Top 10 Correlated features
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "mobile_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Prediciton & model evaluation
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', color='skyblue')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()