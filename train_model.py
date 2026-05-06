import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Features
X = data[['signal_strength', 'fiber_loss', 'temperature', 'traffic_load']]

# Target
y = data['fault']

# Train model
model = RandomForestClassifier()

model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ ML Model Trained Successfully")
