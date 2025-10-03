# train_resnet50.py
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/val"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)  # ensure model folder exists

# ImageDataGenerator with augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # important for ResNet
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Flow generators
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Save class names
class_names = list(train_gen.class_indices.keys())
with open(os.path.join(model_dir, "class_names.pkl"), "wb") as f:
    pickle.dump(class_names, f)
print("Class order:", class_names)

# Compute class weights (handles imbalance)
labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Load ResNet50 base
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
rlp = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
mc = ModelCheckpoint(os.path.join(model_dir, "lung_disease_resnet50.h5"), monitor="val_accuracy", save_best_only=True)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    class_weight=class_weights,
    callbacks=[es, rlp, mc]
)

# Save training history for plotting later
with open(os.path.join(model_dir, "history_resnet50.pkl"), "wb") as f:
    pickle.dump(history.history, f)

print("✅ Training finished. Model saved as model/lung_disease_resnet50.h5")
print("✅ Training history saved as model/history_resnet50.pkl")
