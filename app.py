from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = load_model('model/mobilenet_glaucoma_model.h5', compile=False)

# Preprocess image to match model input
def preprocess_image(img):
    img = img.resize((224, 224))  # adjust to your model's input shape
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Root route for sanity check
@app.route('/')
def home():
    return 'Glaucoma Detection App is running!'

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = preprocess_image(img)
        prediction = model.predict(img)[0]

        # Example: Binary classification - adjust logic based on your training
        result = 'Glaucoma' if prediction[0] < 0.5 else 'Normal'
        return jsonify({
            'prediction': result,
            'confidence': float(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # for Render or local
    app.run(host='0.0.0.0', port=port, debug=True)
