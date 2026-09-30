import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout
)

# Emotion order used by the pretrained model
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Build the pretrained emotion model
model = Sequential([
    Conv2D(64, (5, 5), activation="relu",
           input_shape=(48, 48, 1)),

    MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),

    AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    Conv2D(128, (3, 3), activation="relu"),

    AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),

    Flatten(),

    Dense(1024, activation="relu"),
    Dropout(0.2),

    Dense(1024, activation="relu"),
    Dropout(0.2),

    Dense(7, activation="softmax")
])

# Load pretrained weights
model.load_weights(
    "models/facial_expression_model_weights.h5"
)

print("Emotion model loaded successfully!")

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print()
print("==============================")
print("REAL-TIME EMOTION DETECTOR")
print("==============================")
print("Smile -> Happy")
print("Sad expression -> Sad")
print("Angry expression -> Angry")
print("Surprised expression -> Surprise")
print("Press Q to close")
print("==============================")
print()

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, width, height) in faces:

        # Crop face
        face = gray[
            y:y + height,
            x:x + width
        ]

        # Resize to 48x48
        face = cv2.resize(
            face,
            (48, 48)
        )

        # Normalize
        face = face.astype(
            "float32"
        ) / 255.0

        # Shape: (1, 48, 48, 1)
        face = face.reshape(
            1, 48, 48, 1
        )

        # Predict
        probabilities = model.predict(
            face,
            verbose=0
        )[0]

        # Highest probability
        index = np.argmax(probabilities)

        emotion = EMOTIONS[index]

        confidence = (
            probabilities[index] * 100
        )

        # Face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # Emotion text
        label = f"{emotion} - {confidence:.1f}%"

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Window title
    cv2.imshow(
        "Real-Time Emotion Detector",
        frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()

cv2.destroyAllWindows()

print("Camera closed.")