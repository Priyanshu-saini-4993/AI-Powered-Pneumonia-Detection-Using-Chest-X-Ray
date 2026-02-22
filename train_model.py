import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==============================
# 1. Set Base Directory
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(BASE_DIR, "..", "dataset", "chest_xray", "train")
val_dir   = os.path.join(BASE_DIR, "..", "dataset", "chest_xray", "val")
model_dir = os.path.join(BASE_DIR, "..", "model")

# Create model directory if not exists
os.makedirs(model_dir, exist_ok=True)

print("Train Directory:", train_dir)
print("Validation Directory:", val_dir)

# ==============================
# 2. Image Data Generator
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.0  # We are using separate val folder
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary"
)

# ==============================
# 3. Build Improved CNN Model
# ==============================
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:\n")
model.summary()

# ==============================
# 4. Callbacks (Early Stopping + Best Model Save)
# ==============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "best_model.h5"),
    monitor='val_accuracy',
    save_best_only=True
)

# ==============================
# 5. Train Model
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)

# ==============================
# 6. Save Final Model
# ==============================
model.save(os.path.join(model_dir, "medical_ai_model.keras"))
model.save(os.path.join(model_dir, "medical_ai_model.h5"))

print("\nModel saved successfully in /model folder!")