
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Find the training dataset
project_path = Path(__file__).parent
train_path = project_path / "datasets"

for folder in train_path.iterdir():
    candidate = folder / "train"
    if candidate.exists():
        train_path = candidate
        break

# Display one MRI image from each class
classes = sorted(folder.name for folder in train_path.iterdir() if folder.is_dir())

fig, axes = plt.subplots(1, len(classes), figsize=(14, 4))

for ax, class_name in zip(axes, classes):
    class_folder = train_path / class_name
    image_path = next(
        file for file in class_folder.rglob("*")
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.set_title(class_name)
    ax.axis("off")

plt.tight_layout()
plt.show()
