from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load ML model
model_path = "model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print("⚠️ model.pkl not found")

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        # Input Features
        signal_strength = int(data['signal_strength'])
        fiber_loss = float(data['fiber_loss'])

        temperature = int(data.get('temperature', 35))
        traffic_load = int(data.get('traffic_load', 60))
        power_status = int(data.get('power_status', 1))
        equipment_health = int(data.get('equipment_health', 80))
        humidity = int(data.get('humidity', 50))

        # Create feature array
        features = np.array([[

            signal_strength,
            fiber_loss,
            temperature,
            traffic_load,
            power_status,
            equipment_health,
            humidity

        ]])

        # Prediction
        if model is not None:

            prediction = model.predict(features)[0]

            probability = model.predict_proba(features)[0][1]

        else:

            prediction = 0
            probability = 0

        # Result
        if prediction == 1:
            result = "Fault Detected ⚠️"
        else:
            result = "No Fault ✅"

        # Severity Logic
        if probability > 0.8:
            severity = "High"
        elif probability > 0.5:
            severity = "Medium"
        else:
            severity = "Low"

        # Response
        return jsonify({

            "prediction": result,
            "fault_probability": f"{probability*100:.2f}%",
            "severity": severity,

            "input_data": {

                "signal_strength": signal_strength,
                "fiber_loss": fiber_loss,
                "temperature": temperature,
                "traffic_load": traffic_load,
                "power_status": power_status,
                "equipment_health": equipment_health,
                "humidity": humidity

            }

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        })

# Health Check API
@app.route('/health')
def health():

    return jsonify({

        "status": "Running",
        "model_loaded": model is not None

    })

# Run App
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000,
        debug=True
    )
