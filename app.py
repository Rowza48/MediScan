from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json
import os

app = Flask(__name__)

# Load your trained model and class names
MODEL_PATH = 'model/multiscan_cnn_model.h5'
model = load_model(MODEL_PATH)

# Load class names from JSON
with open("model/classes.json") as f:
    classes_dict = json.load(f)
classes = [classes_dict[str(i)] for i in range(len(classes_dict))]

def preprocess_image(file_stream):
    """
    CORRECTED preprocessing to match EXACTLY how ImageDataGenerator processes images
    """
    try:
        # 1. Read the file bytes and decode as COLOR image (BGR format)
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode the image")
        
        # 2. Convert BGR to RGB (CRITICAL: OpenCV uses BGR, model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Resize to 224x224 (same as training)
        img = cv2.resize(img, (224, 224))
        
        # 4. Normalize pixel values (same as rescale=1./255 in ImageDataGenerator)
        img = img.astype('float32') / 255.0
        
        # 5. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    try:
        # DEBUG: Reset file pointer
        file.stream.seek(0)
        
        # Preprocess the image
        processed_image = preprocess_image(file)
        
        # DEBUG: Print input shape
        print(f"Input shape: {processed_image.shape}")
        print(f"Input range: {processed_image.min():.3f} to {processed_image.max():.3f}")
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # DEBUG: Print raw predictions
        print(f"Raw predictions: {prediction}")
        
        # Get results
        predicted_class_index = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        result = classes[predicted_class_index]
        
        print(f"Predicted: {result} (index: {predicted_class_index}), Confidence: {confidence:.2f}%")

        return render_template('index.html',
                               prediction=f'Prediction: {result}',
                               confidence=f'Confidence: {confidence:.2f}%')

    except Exception as e:
        print(f"ERROR during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)