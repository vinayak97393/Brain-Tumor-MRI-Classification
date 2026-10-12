
from pathlib import Path
import pandas as pd

project_path = Path(__file__).parent
output_path = project_path / "outputs"
output_path.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame({
    "Model": ["Custom CNN", "MobileNetV2"],
    "Test Accuracy": [220 / 246, 212 / 246],
    "Macro F1-Score": [0.89, 0.86],
    "Parameters": [11169476, 2263108],
})

results["Test Accuracy"] = results["Test Accuracy"].round(4)

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

csv_path = output_path / "model_comparison.csv"
results.to_csv(csv_path, index=False)

print(f"\nComparison saved to: {csv_path}")
