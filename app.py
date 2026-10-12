
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_PATH = Path(__file__).parent
MODEL_PATH = PROJECT_PATH / "models" / "brain_tumor_cnn.keras"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]

st.set_page_config(
    page_title="Brain MRI Image Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Brain MRI Image Classifier")
st.write(
    "An educational deep-learning project that classifies "
    "MRI images into four dataset categories."
)

st.warning(
    "Educational demonstration only. This model is not a "
    "medical diagnostic tool and must not be used to make "
    "healthcare decisions."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI image", width="stretch")

    if st.button("Classify Image", type="primary"):
        with st.spinner("Processing image..."):
            model = load_model()

            image = image.resize((224, 224))
            image_array = np.asarray(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.subheader("Model Output")
        st.write(f"**Predicted dataset class:** {predicted_class}")
        st.write(f"**Model confidence score:** {confidence:.2f}%")

        st.caption(
            "The score reflects the model's output, not the "
            "probability that a person has or does not have a tumor."
        )

        st.write("Scores for all dataset classes:")
        for class_name, score in zip(CLASS_NAMES, predictions):
            st.write(f"**{class_name}:** {score * 100:.2f}%")
            st.progress(float(score))
