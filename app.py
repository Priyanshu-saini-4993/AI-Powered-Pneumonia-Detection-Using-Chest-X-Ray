from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

# Load trained model
try:
    model = tf.keras.models.load_model("../model/medical_ai_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.route("/")
def home():
    return "Medical AI Backend Running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Check if file is in request
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    file = request.files["image"]

    try:
        # Convert file to grayscale image
        image_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Preprocess image
        image = cv2.resize(image, (256, 256)) / 255.0
        image = image.reshape(1, 256, 256, 1)

        # Predict
        prediction = model.predict(image)
        prediction_value = float(prediction[0][0])
        print("Prediction value:", prediction_value)  # Debug in terminal

        # Threshold logic
        threshold = 0.5  # increase to 0.6 if too many false positives
        result = "PNEUMONIA DETECTED" if prediction_value > threshold else "NORMAL"

        return jsonify({"prediction": result})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Error processing image"}), 500


if __name__ == "__main__":
    app.run(debug=True)