import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ======================
# 1. Load Model & Classes
# ======================
model = load_model("model/multiscan_cnn_model.h5")

with open("model/classes.json", "r") as f:
    classes = json.load(f)

print("✅ Loaded Classes:", classes)

# ======================
# 2. Test Data
# ======================
test_dir = "Dataset/Dataset_5_Class_Crop/test"
img_size = 224
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    color_mode="rgb"
)

# ======================
# 3. Evaluate
# ======================
predictions = model.predict(test_gen)
y_pred = predictions.argmax(axis=1)
y_true = test_gen.classes

# ======================
# 4. Report
# ======================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(classes.values())))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(classes.values()), 
            yticklabels=list(classes.values()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
