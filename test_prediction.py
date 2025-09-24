import numpy as np
import json
import cv2
from tensorflow.keras.models import load_model

# Load model + classes
model = load_model("model/multiscan_cnn_model.h5")
with open("model/classes.json") as f:
    classes_dict = json.load(f)
classes = [classes_dict[str(i)] for i in range(len(classes_dict))]

def preprocess_xray(img_path):
    # Load as COLOR and convert BGR to RGB
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

img_path = "C:/Users/hp/Pictures/test/Emphysema_24.jpg"

x = preprocess_xray(img_path)
preds = model.predict(x, verbose=0)[0]

idx = int(np.argmax(preds))
print("Prediction:", classes[idx])
print("Confidence:", round(100*preds[idx], 2), "%")
print("All probabilities:", {cls: round(100*float(p),2) for cls,p in zip(classes, preds)})