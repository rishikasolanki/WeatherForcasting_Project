# 🧠 RAIN PREDICTION USING MACHINE LEARNING

# 📦 Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🚀 Load the Dataset
data = pd.DataFrame({
    'Temperature': [23.72, 27.88, 25.06, 23.62, 20.59, 32.29, 34.09, 19.58, 25.79, 26.23,
                    24.20, 33.13, 11.77, 12.17, 20.50, 30.81, 25.93, 31.75, 29.46, 29.97,
                    23.25, 21.57, 35.13, 26.98, 33.66],
    'Humidity': [89.59, 46.48, 83.07, 74.36, 96.88, 51.84, 48.05, 82.97, 81.37, 76.87,
                 45.15, 90.32, 96.96, 67.12, 75.45, 65.04, 49.84, 50.03, 51.93, 43.35,
                 55.87, 47.86, 55.73, 62.31, 45.56],
    'Wind_Speed': [7.33, 5.95, 1.37, 7.00, 4.64, 2.87, 5.57, 5.76, 1.65, 15.82,
                   11.57, 5.77, 6.37, 11.84, 14.79, 7.68, 1.91, 17.76, 12.99, 10.71,
                   0.16, 3.52, 12.46, 4.01, 0.57],
    'Cloud_Cov': [50.50, 4.99, 14.85, 67.25, 47.67, 92.55, 82.52, 98.01, 93.92, 72.86,
                  5.25, 99.91, 47.13, 6.09, 99.14, 57.03, 45.90, 1.97, 42.84, 18.32,
                  11.47, 0.78, 31.47, 30.52, 95.30],
    'Pressure': [1032.37, 921.42, 1007.23, 982.63, 980.85, 1006.04, 993.73, 1036.50, 1029.40, 980.10,
                 1033.98, 987.80, 1046.40, 1003.29, 1011.57, 980.34, 1037.70, 1001.57, 1018.95, 1044.86,
                 1005.17, 1040.71, 1044.71, 1011.06, 1011.00],
    'Rain': ['rain', 'no rain', 'no rain', 'rain', 'no rain', 'rain', 'no rain', 'rain', 'rain', 'rain',
             'no rain', 'no rain', 'no rain', 'no rain', 'rain', 'no rain', 'no rain', 'rain', 'no rain', 'no rain',
             'rain', 'no rain', 'rain', 'no rain', 'no rain']
})

# ✅ Convert 'Rain' to binary
data['Rain'] = data['Rain'].map({'no rain': 0, 'rain': 1})

# 🧪 Split Features and Labels
X = data[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cov', 'Pressure']]
y = data['Rain']

# 📊 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📈 Predictions
y_pred = model.predict(X_test)

# 🧾 Evaluation
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 🔍 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🔮 Predict on new sample
new_sample = np.array([[25.5, 78.0, 5.5, 60.0, 1010.0]])  # [Temperature, Humidity, Wind_Speed, Cloud_Cov, Pressure]
prediction = model.predict(new_sample)
print("\n🌦 Prediction for sample weather:", "Rain" if prediction[0] == 1 else "No Rain")
