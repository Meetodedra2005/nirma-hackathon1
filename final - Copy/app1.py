    
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import importlib

# Load main cancer models
cancer_models = {
    "Brain Tumor": tf.keras.models.load_model("models/brain_tumor_model.h5"),
    "Lung Cancer": tf.keras.models.load_model("models/lung_cancer_classifier.h5"),
    "Retinoblastoma": tf.keras.models.load_model("models/retinoblastoma_model.h5")
}

# Define skin cancer models with module names
skin_models = {
    "Actinic Keratosis": "skin_models.actinic_keratosis",
    "Dermatofibroma": "skin_models.dermatofibroma",
    "Melanoma": "skin_models.melanoma",
    "Nevus": "skin_models.nevus",
    "Pigmented Benign Keratosis": "skin_models.pigmented_benign_keratosis",
    "Seborrheic Keratosis": "skin_models.seborrheic_keratosis",
}

st.title("Cancer Classification System")

# Dropdown for cancer type selection
selected_cancer_type = st.selectbox("Select the Cancer Type", list(cancer_models.keys()) + ["Skin Disease"])

# File upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to array
    img_array = np.array(image)
    
    if selected_cancer_type in ["Brain Tumor", "Lung Cancer"]:
        img_array = np.array(image)  # Convert to NumPy array

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Check if it's RGB
            st.error("Please upload a **black and white (grayscale)** image for Brain Tumor or Lung Cancer classification.")
        else:
            img_array = cv2.resize(img_array, (224, 224))  # Resize to model input size
            img_array = img_array / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = np.expand_dims(img_array, axis=-1)  # Convert (1, 224, 224) → (1, 224, 224, 1)

            # Convert single channel grayscale to 3 channels (needed for model)
            img_array = np.repeat(img_array, 3, axis=-1)  # (1, 224, 224, 1) → (1, 224, 224, 3)

            model = cancer_models[selected_cancer_type]
            prediction = model.predict(img_array)
            max_confidence = np.max(prediction)
            confidence_threshold = 0.6

            if max_confidence < confidence_threshold:
                st.write("No Cancer Detected (-1)")
            else:
                st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

    # Skin Disease Classification (Calls each model separately)
    elif selected_cancer_type == "Skin Disease":
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        highest_confidence = 0
        best_match = "Unknown"

        for skin_type, module_name in skin_models.items():
            module = importlib.import_module(module_name)  # Dynamic import
            result = module.predict_skin_disease(img_array)  # Expecting ("Found"/"Not Found", confidence)

            if result == "Found":
                # highest_confidence = confidence
                best_match = skin_type
                break

        confidence_threshold = 0.6
        if highest_confidence < confidence_threshold:
            st.write("No Specific Skin Cancer Detected (-1)")
        else:
            st.write(f"Predicted Skin Disease: {best_match} (Confidence: {highest_confidence:.2f})")

    # Other cancer types (Retinoblastoma, etc.) accept RGB images
    else:
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = cancer_models[selected_cancer_type]
        prediction = model.predict(img_array)
        max_confidence = np.max(prediction)
        confidence_threshold = 0.6

        if max_confidence < confidence_threshold:
            st.write("No Cancer Detected (-1)")
        else:
            st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")
