
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

PROJECT_PATH = Path(__file__).parent
DATASET_PATH = PROJECT_PATH / "datasets"
MODEL_PATH = PROJECT_PATH / "models" / "mobilenetv2_brain_tumor.keras"

def find_split(split_name):
    for folder in DATASET_PATH.iterdir():
        candidate = folder / split_name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find {split_name} dataset")

test_ds = tf.keras.utils.image_dataset_from_directory(
    find_split("test"),
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probabilities = model.predict(test_ds)
y_pred = np.argmax(probabilities, axis=1)

class_names = test_ds.class_names

print("\nMOBILENETV2 TEST ACCURACY:",
      round(accuracy_score(y_true, y_pred), 4))

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
))
