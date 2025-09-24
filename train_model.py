import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ======================
# 1. Dataset Paths
# ======================
train_dir = "Dataset/Dataset_5_Class_Crop/train"
val_dir = "Dataset/Dataset_5_Class_Crop/val"
test_dir = "Dataset/Dataset_5_Class_Crop/test"

img_size = 224
batch_size = 32
epochs = 20

# ======================
# 2. Data Generators
# ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="rgb"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="rgb"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    color_mode="rgb"
)

# ======================
# 3. Save Class Labels
# ======================
class_indices = train_gen.class_indices
classes = {v: k for k, v in class_indices.items()}

os.makedirs("model", exist_ok=True)
with open("model/classes.json", "w") as f:
    json.dump(classes, f)

print("✅ Classes saved at model/classes.json:", classes)

# ======================
# 4. Build Custom CNN Model
# ======================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# ======================
# 5. Compile Model
# ======================
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ======================
# 6. Train Model
# ======================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# ======================
# 7. Save Model
# ======================
model.save("model/multiscan_cnn_model.h5")
print("🎉 CNN Model saved at model/multiscan_cnn_model.h5")
