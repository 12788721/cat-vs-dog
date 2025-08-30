from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model_path = 'best_mobilenetv2.keras'
model = load_model(model_path)

# Define your class labels (update these to match your model's classes)
class_labels = ['cat', 'dog']  # 0=cat, 1=dog

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction_text='No image uploaded.')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction_text='No image selected.')
    if file:
        # Save the image temporarily
        img_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(img_path)

        # Preprocess the image for your model
        img = image.load_img(img_path, target_size=(160, 160))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # No normalization, matches your Colab code

        # Predict (binary classification)
        prob = float(model.predict(img_array)[0][0])
        pred_class = class_labels[int(prob >= 0.5)]

        # Remove the temp image
        os.remove(img_path)

        return render_template('index.html', prediction_text=f'Prediction: {pred_class}')
    return render_template('index.html', prediction_text='Error processing image.')

if __name__ == "__main__":
    app.run(debug=True)