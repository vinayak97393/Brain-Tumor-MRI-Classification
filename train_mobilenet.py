
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MobileNetV2 pretrained on ImageNet
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers for our first experiment
base_model.trainable = False

# Build our transfer-learning model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nMOBILENETV2 MODEL CREATED, HOMIE!")


from pathlib import Path

PROJECT_PATH = Path(__file__).parent
DATASET_PATH = PROJECT_PATH / "datasets"
MODEL_PATH = PROJECT_PATH / "models" / "mobilenetv2_brain_tumor.keras"

def find_split(split_name):
    for folder in DATASET_PATH.iterdir():
        candidate = folder / split_name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find {split_name} dataset")

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

model.save(MODEL_PATH)
print(f"\nMobileNetV2 saved to: {MODEL_PATH}")
