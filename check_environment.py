
from pathlib import Path
import tensorflow as tf

project_path = Path(__file__).parent
datasets_path = project_path / "datasets"

def find_split(split_name):
    for outer_folder in datasets_path.iterdir():
        candidate = outer_folder / split_name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find {split_name} dataset")

train_path = find_split("train")
valid_path = find_split("valid")
test_path = find_split("test")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

print("\nCLASS NAMES:", train_ds.class_names)
print("Training batches:", len(train_ds))
print("Validation batches:", len(valid_ds))
print("Testing batches:", len(test_ds))

images, labels = next(iter(train_ds))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("\nDATA LOADING SUCCESSFUL, HOMIE!")
