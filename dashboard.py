
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_PATH = Path(__file__).parent
OUTPUTS_PATH = PROJECT_PATH / "outputs"

st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Brain Tumor MRI Classification")
st.subheader("Model Performance Dashboard")

st.caption(
    "Educational machine-learning project — not a medical diagnostic tool."
)

# Model comparison
comparison_file = OUTPUTS_PATH / "model_comparison.csv"

if comparison_file.exists():
    results = pd.read_csv(comparison_file)

    col1, col2, col3 = st.columns(3)

    cnn_accuracy = float(
        results.loc[results["Model"] == "Custom CNN", "Test Accuracy"].iloc[0]
    )
    mobile_accuracy = float(
        results.loc[results["Model"] == "MobileNetV2", "Test Accuracy"].iloc[0]
    )

    with col1:
        st.metric("Custom CNN Accuracy", f"{cnn_accuracy:.2%}")

    with col2:
        st.metric("MobileNetV2 Accuracy", f"{mobile_accuracy:.2%}")

    with col3:
        st.metric(
            "Accuracy Difference",
            f"{(cnn_accuracy - mobile_accuracy) * 100:.2f} percentage points",
        )

    st.markdown("### Model Comparison")
    st.dataframe(results, hide_index=True, use_container_width=True)

    chart_data = results.set_index("Model")["Test Accuracy"]
    st.bar_chart(chart_data, y_label="Test Accuracy")

else:
    st.warning("Run model_comparison.py first to generate the comparison CSV.")

# Confusion matrix
st.markdown("### Custom CNN Confusion Matrix")

matrix_file = OUTPUTS_PATH / "confusion_matrix.png"

if matrix_file.exists():
    st.image(
        Image.open(matrix_file),
        caption="Custom CNN — test dataset confusion matrix",
        width="stretch",
    )
else:
    st.info(
        "Confusion matrix image not found. Run evaluate_cnn.py "
        "and make sure it saves outputs/confusion_matrix.png."
    )

st.markdown("### Project Summary")

st.write(
    """
    - **Task:** Classify brain MRI images into four dataset categories.
    - **Classes:** Glioma, meningioma, no_tumor, and pituitary.
    - **Models:** Custom Convolutional Neural Network (CNN) and MobileNetV2.
    - **Evaluation:** Test accuracy, classification metrics, and confusion matrix.
    """
)

st.warning(
    "Accuracy and confidence scores do not establish clinical reliability. "
    "Predictions must not be used to make medical decisions."
)
