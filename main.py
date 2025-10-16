import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib


dataset = pd.read_csv("dataset_37_diabetes.csv")

# These medical values cannot be 0 (glucose, blood pressure, BMI)
dataset = dataset[(dataset['plas'] > 0) & (dataset['pres'] > 0) & (dataset['mass'] > 0)]

# KEEP: plas, mass, age, preg, pedi 
dataset.drop(columns=['pres', 'skin', 'insu'], inplace=True)

dataset_encoded = pd.get_dummies(dataset, columns=['class'], drop_first=True)

y = dataset_encoded['class_tested_positive']
X = dataset_encoded.drop(columns=['class_tested_positive'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Accuracy: {acc:.4f}")

# Save the trained model and scaler
joblib.dump(knn, 'diabetes_knn_model.pkl')
joblib.dump(scaler, 'diabetes_scaler.pkl')
print("\nModel and scaler saved successfully!")
print("  - diabetes_knn_model.pkl")
print("  - diabetes_scaler.pkl")

