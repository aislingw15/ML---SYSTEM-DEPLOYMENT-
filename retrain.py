import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import json
import os

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\famim\.project\data"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")
IMG_SIZE   = (150, 150)
BATCH_SIZE = 32
EPOCHS     = 20
OUTPUT_DIR = r"C:\Users\famim\.project"

# ── Data generators ──────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ── Class counts & weights ───────────────────────────────────────────────────
class_indices = train_data.class_indices
print("\nClass indices:", class_indices)

# Count images per class
class_counts = {}
for cls, idx in class_indices.items():
    path = os.path.join(TRAIN_DIR, cls)
    count = len([f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    class_counts[cls] = count
    print(f"  {cls}: {count} images")

total = sum(class_counts.values())
n_classes = len(class_indices)

# Compute balanced class weights
class_weights = {}
for cls, idx in class_indices.items():
    weight = total / (n_classes * class_counts[cls])
    class_weights[idx] = weight
    print(f"  Weight for {cls} (idx {idx}): {weight:.3f}")

# ── Model — MobileNetV2 transfer learning ────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base first

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'disease_model_best.h5'),
        monitor='val_accuracy', save_best_only=True
    )
]

# ── Phase 1 — train top layers only ─────────────────────────────────────────
print("\n── Phase 1: Training top layers ──")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# ── Phase 2 — fine-tune top 30 layers of base ───────────────────────────────
print("\n── Phase 2: Fine-tuning ──")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ── Save final model & class indices ────────────────────────────────────────
model.save(os.path.join(OUTPUT_DIR, 'disease_model.h5'))
with open(os.path.join(OUTPUT_DIR, 'class_indices.json'), 'w') as f:
    json.dump(class_indices, f)

print("\n✅ Training complete.")
print(f"   Model saved to: {OUTPUT_DIR}\\disease_model.h5")
print(f"   Class indices:  {OUTPUT_DIR}\\class_indices.json")
