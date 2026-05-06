from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    signal = int(data['signal_strength'])
    fiber = float(data['fiber_loss'])

    if signal < 40 or fiber > 3:
        result = "Fault Detected ⚠️"
    else:
        result = "No Fault ✅"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
