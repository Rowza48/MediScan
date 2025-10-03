# evaluate_resnet50.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Paths
val_dir = "dataset/val"
model_path = "model/lung_disease_resnet50.h5"
history_path = "model/history_resnet50.pkl"
class_path = "model/class_names.pkl"

# Load model and class names
model = load_model(model_path)
with open(class_path, "rb") as f:
    class_names = pickle.load(f)
print("✅ Model loaded with classes:", class_names)

# Validation generator
val_gen = ImageDataGenerator(preprocessing_function=None).flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
loss, val_acc = model.evaluate(val_gen)
print(f"📊 Validation Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Predictions
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\n📑 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot training history if available
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    plt.figure(figsize=(12,5))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # Loss
    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
else:
    print("⚠️ Training history not found. Accuracy/Loss graphs won't be shown.")
