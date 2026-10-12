
from pathlib import Path

project_path = Path(__file__).parent
datasets_path = project_path / "datasets"

split_names = ["train", "valid", "test"]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split_name in split_names:
    split_path = None

    # Find the actual split folder inside the downloaded wrapper folders
    for outer_folder in datasets_path.iterdir():
        candidate = outer_folder / split_name
        if candidate.is_dir():
            split_path = candidate
            break

    if split_path is None:
        print(f"Could not find the {split_name} folder!")
        continue

    total = 0
    print(f"\n{split_name.upper()} DATASET")

    for class_folder in sorted(split_path.iterdir()):
        if not class_folder.is_dir():
            continue

        count = sum(
            1 for file in class_folder.rglob("*")
            if file.is_file() and file.suffix.lower() in image_extensions
        )

        total += count
        print(f"{class_folder.name}: {count} images")

    print(f"Total images: {total}")
