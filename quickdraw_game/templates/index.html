from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import base64
import re
from PIL import Image
import io
import random

app = Flask(__name__)

import os
model_path = os.path.join(os.path.dirname(__file__), "model_final.h5")
model = tf.keras.models.load_model(model_path)

class_names = ['gazebo', 'kalachuchi', 'Siklab', 'Tomorrow logo']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_prompt', methods=['POST'])
def get_prompt():
    prompts = random.sample(class_names, len(class_names))
    return jsonify({"prompts": prompts})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    expected_class = request.json['expected_class']
    data = re.sub('^data:image/.+;base64,', '', data)
    image_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index])

    if confidence >= 0.95:
        if predicted_class == expected_class:
            result = f"The AI guessed: {predicted_class} ({confidence*100:.1f}%)\nCorrect!"
            correct = True
        else:
            result = f"The AI guessed: {predicted_class} ({confidence*100:.1f}%)\nWrong!"
            correct = False
    else:
        result = "The AI doesn't know what you're drawing..."
        correct = False

    return jsonify({"result": result, "correct": correct})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7272, debug=True)
