import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =========================
# SETTINGS
# =========================
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 15

# =========================
# LOAD TRAINING DATA
# =========================
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=123
)

# =========================
# LOAD VALIDATION DATA
# =========================
validation_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset="validation",
    seed=123
)

# =========================
# CLASS NAMES
# =========================
class_names = train_data.class_names

print("\nClasses:")
print(class_names)

# =========================
# PERFORMANCE
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)

# =========================
# DATA AUGMENTATION
# =========================
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05)
])

# =========================
# CNN MODEL
# =========================
model = models.Sequential([
    
    layers.Input(shape=(48, 48, 1)),
    
    layers.Rescaling(1.0 / 255),

    augmentation,

    # Block 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.20),

    # Block 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

    # Block 4
    layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

    # Classification
    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.50),

    layers.Dense(7, activation="softmax")
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# SHOW MODEL
# =========================
model.summary()

# =========================
# CALLBACKS
# =========================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)

model.save("models/emotion_model.keras")

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")
print("Model saved to:")
print("models/emotion_model.keras")