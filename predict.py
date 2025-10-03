# predict.py
import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Paths - update these to be relative
model_path = "model/lung_disease_resnet50.h5"
class_path = "model/class_names.pkl"

# Load model and class names
try:
    model = load_model(model_path)
    with open(class_path, "rb") as f:
        class_names = pickle.load(f)
    print("✅ Model and classes loaded:", class_names)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    class_names = []

# Function to preprocess single image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Could not read image at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict single image
def predict_single(img_path):
    if model is None:
        raise ValueError("Model not loaded properly")
    
    img = preprocess_image(img_path)
    pred_probs = model.predict(img)
    pred_class = class_names[np.argmax(pred_probs)]
    confidence = float(np.max(pred_probs)) * 100
    
    print(f"Image: {os.path.basename(img_path)} -> Prediction: {pred_class} (Confidence: {confidence:.2f}%)")
    return pred_class

# Example usage:
if __name__ == "__main__":
    test_image = "C:/Users/hp/Pictures/test/Normal (452).jpg"  # replace with your image path
    result = predict_single(test_image)
    print(f"Result: {result}")