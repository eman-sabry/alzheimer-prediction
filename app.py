from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

#app initialization
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#load the trained model
model = load_model("best_model.keras")
#image classes
classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
#preprocess the image
def preprocess_image(img_path, target_size=(128,128), padding=4):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        img = img[y1:y2, x1:x2]
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1) 
    img = np.expand_dims(img, axis=0)   
    return img

#route for home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        if img is None:
            return "Error reading image"
        pred = model.predict(img)
        cls_idx = np.argmax(pred)
        prediction = classes[cls_idx]
        return render_template("index.html", prediction=prediction, filename=filename)
    # not image uploaded, just render the page
    return render_template("index.html", prediction=None)

#run the app
if __name__ == "__main__":
    app.run(debug=True)
