import os
import cv2
import torch
from flask import Flask, render_template, request, send_file
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO("best.pt")  # Ensure best.pt is in the project folder

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded file
        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return "No file selected!"

        # Save uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(image_path)

        # Run YOLOv8 model on the image
        results = model(image_path)

        # Save output image
        output_path = os.path.join(RESULT_FOLDER, "output.jpg")
        for result in results:
            annotated_image = result.plot()
            cv2.imwrite(output_path, annotated_image)

        return render_template("result.html", image_url=output_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
