
import tensorflow as tf
from tensorflow.keras import layers, models

# Create our CNN model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    # Normalize pixel values from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    # First feature extraction block
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second feature extraction block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Third feature extraction block
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into a vector
    layers.Flatten(),

    # Learn patterns for classification
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Four output classes
    layers.Dense(4, activation="softmax")
])

# Configure the model for training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display the model architecture
model.summary()

print("\nCNN MODEL CREATED SUCCESSFULLY, HOMIE!")
