
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

PROJECT_PATH = Path(__file__).parent
DATASET_PATH = PROJECT_PATH / "datasets"
MODEL_PATH = PROJECT_PATH / "models" / "brain_tumor_cnn.keras"

def find_split(split_name):
    for folder in DATASET_PATH.iterdir():
        candidate = folder / split_name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find dataset split: {split_name}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    find_split("train"),
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    seed=42
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    find_split("valid"),
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=callbacks
)

# Save the final model state as well
model.save(MODEL_PATH)

print(f"\nTraining finished! Model saved to: {MODEL_PATH}")
