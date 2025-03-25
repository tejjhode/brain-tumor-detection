from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image 
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_cnn.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess th uploaded image
def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]  # Get prediction score
    return "🚨 Tumor Detected" if prediction > 0.5 else " No Tumor Detected"

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            print(f"Image saved at: {file_path}")  # Debugging

            # Make prediction
            result = predict_tumor(file_path)
            print(f"Prediction result: {result}")  # Debugging

            return render_template("index.html", prediction=result, image=file_path)

    return render_template("index.html", prediction=None, image=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  
