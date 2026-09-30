import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = tf.keras.models.load_model("models/emotion_model.keras")

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    shuffle=False
)

class_names = test_data.class_names

# Get predictions
y_true = []
y_pred = []

for images, labels in test_data:
    predictions = model.predict(images, verbose=0)
    
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Accuracy
accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print("\n==============================")
print("MODEL TEST RESULTS")
print("==============================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Detailed results
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))