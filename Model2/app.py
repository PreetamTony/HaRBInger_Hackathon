from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)
model = load_model("keras_Model.h5", compile=False)

class_names = open("labels.txt", "r").readlines()

def model_predict(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:], confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            class_name, confidence_score = model_predict(file_path)
            return render_template('index.html', class_name=class_name, confidence_score=confidence_score, image_path=file_path)
    return render_template('index.html')

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
