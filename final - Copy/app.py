# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from PIL import Image
# import os

# def load_model_safe(path):
#     try:
#         if os.path.exists(path):
#             return tf.keras.models.load_model(path)
#         else:
#             st.error(f"Model file not found: {path}")
#             return None
#     except Exception as e:
#         st.error(f"Error loading model {path}: {str(e)}")
#         return None

# # Load general cancer models
# cancer_models = {
#     "Brain Tumor": load_model_safe("model/brain_tumor_model.h5"),
#     "Lung Cancer": load_model_safe("model/lung_cancer_classifier.h5"),
#     "Retinoblastoma": load_model_safe("model/retinoblastoma_model.h5"),
#     "Skin Disease": load_model_safe("model/skin_disease_model2.h5")
# }

# cancer_models = {k: v for k, v in cancer_models.items() if v is not None}

# # Load skin cancer models
# skin_cancer_models = {
#     "Actinic Keratosis": load_model_safe(r"model\skin_cancer_ak_model.h5"),
#     "Dermatofibroma": load_model_safe(r"model\skin_cancer_dermato_model.h5"),
#     "Melanoma": load_model_safe(r"model\skin_cancer_melonoma_model.h5"),
#     "Nevus": load_model_safe(r"model\skin_cancer_melonoma_model.h5"),
#     "Pigmented Benign Keratosis": load_model_safe(r"model\skin_disease_classifier.h5"),
#     # "Seborrheic Keratosis": load_model_safe(r"m
#                                             # odel\seborrheic_keratosis_classifier.h5")
# }

# skin_cancer_models = {k: v for k, v in skin_cancer_models.items() if v is not None}

# st.title("Cancer Classification System")
# st.write("Upload an image to identify the type of cancer.")

# if not cancer_models:
#     st.error("No general cancer models loaded successfully. Please check model paths.")
#     st.stop()

# # User selects type of cancer
# selected_cancer_type = st.selectbox("Select the Cancer Type", list(cancer_models.keys()))

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     image = image.convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     img_array = np.array(image)
#     img_array = cv2.resize(img_array, (224, 224))
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Get the selected model
#     selected_model = cancer_models[selected_cancer_type]
#     confidence_threshold = 0.6
#     if selected_model=="Skin Disease":
#         # If the selected type is skin cancer, classify further
#             skin_cancer_type = None
#             highest_skin_confidence = 0
            
#             for skin_type, model in skin_cancer_models.items():
#                 prediction = model.predict(img_array)
#                 confidence = np.max(prediction)

#                 if confidence > highest_skin_confidence:
#                     highest_skin_confidence = confidence
#                     skin_cancer_type = skin_type

#             if highest_skin_confidence < confidence_threshold:
#                 st.write("No Specific Skin Cancer Detected (-1)")
#             else:
#                 st.write(f"Predicted Skin Cancer Type: {skin_cancer_type} (Confidence: {highest_skin_confidence:.2f})")
#     else: 

#         prediction = selected_model.predict(img_array)
#         max_confidence = np.max(prediction)
#         if max_confidence < confidence_threshold:
#             st.write("No Cancer Detected (-1)")
#         else:
#             st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from PIL import Image
# import os

# def load_model_safe(path):
#     """Loads a model safely and returns None if the model is not found or has an error."""
#     try:
#         if os.path.exists(path):
#             return tf.keras.models.load_model(path)
#         else:
#             st.error(f"Model file not found: {path}")
#             return None
#     except Exception as e:
#         st.error(f"Error loading model {path}: {str(e)}")
#         return None

# # Load general cancer models
# cancer_models = {
#     "Brain Tumor": load_model_safe("model/brain_tumor_model.h5"),
#     "Lung Cancer": load_model_safe("model/lung_cancer_classifier (1).h5"),
#     "Retinoblastoma": load_model_safe("model/retinoblastoma_model.h5"),
#     "Skin Disease": load_model_safe("model/skin_disease_model2.h5")
# }

# # Remove None values (failed model loads)
# cancer_models = {k: v for k, v in cancer_models.items() if v is not None}

# # Load skin cancer models
# skin_cancer_models = {
#     "Actinic Keratosis": load_model_safe(r"model\skin_cancer_ak_model.h5"),
#     "Dermatofibroma": load_model_safe(r"model\skin_cancer_dermato_model.h5"),
#     "Melanoma": load_model_safe(r"model\skin_cancer_melonoma_model.h5"),
#     "Nevus": load_model_safe(r"model\skin_cancer_nevus_model.h5"),
#     "Pigmented Benign Keratosis": load_model_safe(r"model\skin_disease_classifier.h5"),
#     "Seborrheic Keratosis": load_model_safe(r"model\seborrheic_keratosis_classifier.h5")
# }

# skin_cancer_models = {k: v for k, v in skin_cancer_models.items() if v is not None}

# # Streamlit UI
# st.title("Cancer Classification System")
# st.write("Upload an image to identify the type of cancer.")

# # If no models are loaded, stop execution
# if not cancer_models:
#     st.error("No general cancer models loaded successfully. Please check model paths.")
#     st.stop()

# # User selects type of cancer
# selected_cancer_type = st.selectbox("Select the Cancer Type", list(cancer_models.keys()))

# # Upload image
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert image to numpy array
#     img_array = np.array(image)
#     img_array = cv2.resize(img_array, (224, 224))
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Get the selected model
#     selected_model = cancer_models[selected_cancer_type]
#     confidence_threshold = 0.5

#     # **Check Image Format Based on Cancer Type**
#     if selected_cancer_type in ["Brain Tumor", "Lung Cancer"]:
#         # Convert image to grayscale if it's not already
#         # gray_image = image.convert("L")  # Convert to grayscale
#         # if image.mode != "L":  # If uploaded image was not grayscale
#             # st.error("Please upload a **black and white (grayscale)** image for Brain Tumor or Lung Cancer classification.")
#         # else:
#             # Proceed with prediction
#             prediction = selected_model.predict(img_array)
#             max_confidence = np.max(prediction)
#             if max_confidence < confidence_threshold:
#                 st.write("No Cancer Detected (-1)")
#             else:
#                 st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

#     elif selected_cancer_type == "Skin Disease":
#         # Classify skin cancer type
#         skin_cancer_type = None
#         highest_skin_confidence = 0
        
#         for skin_type, model in skin_cancer_models.items():
#             prediction = model.predict(img_array)
#             confidence = np.max(prediction)

#             if confidence > highest_skin_confidence:
#                 highest_skin_confidence = confidence
#                 skin_cancer_type = skin_type

#         if highest_skin_confidence < confidence_threshold:
#             st.write("No Specific Skin Cancer Detected (-1)")
#         else:
#             st.write(f"Predicted Skin Cancer Type: {skin_cancer_type} (Confidence: {highest_skin_confidence:.2f})")

#     else:
#         # Other cancer types (Retinoblastoma, etc.) accept RGB images
#         prediction = selected_model.predict(img_array)
#         max_confidence = np.max(prediction)
#         if max_confidence < confidence_threshold:
#             st.write("No Cancer Detected (-1)")
#         else:
#             st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

# # import streamlit as st
# # import numpy as np
# # import cv2
# # import tensorflow as tf
# # from PIL import Image
# # import os

# # # Load brain tumor and lung cancer models
# # cancer_models = {
# #     "Brain Tumor": tf.keras.models.load_model("model/brain_tumor_model.h5"),
# #     "Lung Cancer": tf.keras.models.load_model("model/lung_cancer_classifier.h5"),
# #     "Retinoblastoma": tf.keras.models.load_model(r"model/retinoblastoma_model.h5")
# # }

# # # Dynamic import for skin models
# # skin_models = {
# #     "Actinic Keratosis": "skin_models.actinic_keratosis",
# #     "Dermatofibroma": "skin_models.dermatofibroma",
# #     "Melanoma": "skin_models.melanoma",
# #     "Nevus": "skin_models.nevus",
# #     "Pigmented Benign Keratosis": "skin_models.benign_keratosis",
# #     "Seborrheic Keratosis": "skin_models.seborrheic_keratosis",
# # }

# # st.title("Cancer Classification System")

# # selected_cancer_type = st.selectbox("Select the Cancer Type", list(cancer_models.keys()) + ["Skin Disease"])

# # uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
# #     st.image(image, caption="Uploaded Image", use_column_width=True)

# #     img_array = np.array(image)
# #     img_array = cv2.resize(img_array, (224, 224))
# #     img_array = img_array / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)

# #     if selected_cancer_type in ["Brain Tumor", "Lung Cancer"]:
# #         # Convert image to grayscale if it's not already
# #         gray_image = image.convert("L")  # Convert to grayscale
# #         if image.mode != "L":  # If uploaded image was not grayscale
# #             st.error("Please upload a **black and white (grayscale)** image for Brain Tumor or Lung Cancer classification.")
# #         else:
# #             # Proceed with prediction
# #             prediction = selected_model.predict(img_array)
# #             max_confidence = np.max(prediction)
# #             if max_confidence < confidence_threshold:
# #                 st.write("No Cancer Detected (-1)")
# #             else:
# #                 st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

# #     elif selected_cancer_type == "Skin Disease":
# #         highest_confidence = 0
# #         best_match = "Unknown"

# #         for skin_type, module_name in skin_models.items():
# #             module = __import__(module_name, fromlist=["predict_skin_disease"])
# #             confidence = module.predict_skin_disease(img_array)

# #             if confidence > highest_confidence:
# #                 highest_confidence = confidence
# #                 best_match = skin_type

# #             st.write(f"Predicted Skin Disease: {best_match} (Confidence: {highest_confidence:.2f})")
            
# #     else:
# #         # Other cancer types (Retinoblastoma, etc.) accept RGB images
# #         prediction = selected_model.predict(img_array)
# #         max_confidence = np.max(prediction)
# #         if max_confidence < confidence_threshold:
# #             st.write("No Cancer Detected (-1)")
# #         else:
# #             st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")


# # import streamlit as st
# # import numpy as np
# # import cv2
# # import tensorflow as tf
# # from PIL import Image
# # import os

# # # Load main cancer models
# # cancer_models = {
# #     "Brain Tumor": tf.keras.models.load_model("models/brain_tumor_model.h5"),
# #     "Lung Cancer": tf.keras.models.load_model("models/lung_cancer_classifier.h5"),
# #     "Retinoblastoma": tf.keras.models.load_model("models/retinoblastoma_model.h5")
# # }

# # # Define skin cancer models with module names
# # skin_models = {
# #     "Actinic Keratosis": "skin_models.actinic_keratosis",
# #     "Dermatofibroma": "skin_models.dermatofibroma",
# #     "Melanoma": "skin_models.melanoma",
# #     "Nevus": "skin_models.nevus",
# #     "Pigmented Benign Keratosis": "skin_models.pigmented_benign_keratosis",
# #     "Seborrheic Keratosis": "skin_models.seborrheic_keratosis",
# # }

# # st.title("Cancer Classification System")

# # # Dropdown for cancer type selection
# # selected_cancer_type = st.selectbox("Select the Cancer Type", list(cancer_models.keys()) + ["Skin Disease"])

# # # File upload
# # uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
# #     st.image(image, caption="Uploaded Image", use_column_width=True)

# #     # Convert to array and preprocess
# #     img_array = np.array(image)
# #     img_array = cv2.resize(img_array, (224, 224))
# #     img_array = img_array / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)

# #     # Grayscale check for Brain Tumor & Lung Cancer
# #     if selected_cancer_type in ["Brain Tumor", "Lung Cancer"]:
# #         if image.mode != "L":
# #             st.error("Please upload a **black and white (grayscale)** image for Brain Tumor or Lung Cancer classification.")
# #         else:
# #             model = cancer_models[selected_cancer_type]
# #             prediction = model.predict(img_array)
# #             max_confidence = np.max(prediction)
# #             confidence_threshold = 0.6

# #             if max_confidence < confidence_threshold:
# #                 st.write("No Cancer Detected (-1)")
# #             else:
# #                 st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

# #     # Skin Disease Classification (Call Each Model)
# #     elif selected_cancer_type == "Skin Disease":
# #         highest_confidence = 0
# #         best_match = "Unknown"

# #         for skin_type, module_name in skin_models.items():
# #             module = __import__(module_name, fromlist=["predict_skin_disease"])
# #             confidence = module.predict_skin_disease(img_array)

# #             if confidence > highest_confidence:
# #                 highest_confidence = confidence
# #                 best_match = skin_type

# #         confidence_threshold = 0.6
# #         if highest_confidence < confidence_threshold:
# #             st.write("No Specific Skin Cancer Detected (-1)")
# #         else:
# #             st.write(f"Predicted Skin Disease: {best_match} (Confidence: {highest_confidence:.2f})")

# #     # Other cancer types (Retinoblastoma, etc.) accept RGB images
# #     else:
# #         model = cancer_models[selected_cancer_type]
# #         prediction = model.predict(img_array)
# #         max_confidence = np.max(prediction)
# #         confidence_threshold = 0.6

# #         if max_confidence < confidence_threshold:
# #             st.write("No Cancer Detected (-1)")
# #         else:
# #             st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

# # import streamlit as st
# # import numpy as np
# # import cv2
# # import tensorflow as tf
# # from PIL import Image
# # import importlib
# # import os

# # # Load main cancer models
# # cancer_models = {
# #     "Brain Tumor": tf.keras.models.load_model("models/brain_tumor_model.h5"),
# #     "Lung Cancer": tf.keras.models.load_model("models/lung_cancer_classifier.h5"),
# #     "Retinoblastoma": tf.keras.models.load_model("models/retinoblastoma_model.h5")
# # }

# # # Define skin cancer models with module names
# # skin_models = {
# #     "Actinic Keratosis": "skin_models.actinic_keratosis",
# #     "Dermatofibroma": "skin_models.dermatofibroma",
# #     "Melanoma": "skin_models.melanoma",
# #     "Nevus": "skin_models.nevus",
# #     "Pigmented Benign Keratosis": "skin_models.pigmented_benign_keratosis",
# #     "Seborrheic Keratosis": "skin_models.seborrheic_keratosis",
# # }

# # st.title("Cancer Classification System")

# # # Dropdown for cancer type selection
# # selected_cancer_type = st.selectbox("Select the Cancer Type", list(cancer_models.keys()) + ["Skin Disease"])

# # # File upload
# # uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
# #     st.image(image, caption="Uploaded Image", use_column_width=True)

# #     # Convert to array and preprocess
# #     img_array = np.array(image)
# #     img_array = cv2.resize(img_array, (224, 224))
# #     img_array = img_array / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)

# #     # Grayscale check for Brain Tumor & Lung Cancer
# #     if selected_cancer_type in ["Brain Tumor", "Lung Cancer"]:
# #         if image.mode != "L":
# #             st.error("Please upload a **black and white (grayscale)** image for Brain Tumor or Lung Cancer classification.")
# #         else:
# #             model = cancer_models[selected_cancer_type]
# #             prediction = model.predict(img_array)
# #             max_confidence = np.max(prediction)
# #             confidence_threshold = 0.6

# #             if max_confidence < confidence_threshold:
# #                 st.write("No Cancer Detected (-1)")
# #             else:
# #                 st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

# #     # Skin Disease Classification (Call Each Model)
# #     elif selected_cancer_type == "Skin Disease":
# #         highest_confidence = 0
# #         best_match = "Unknown"

# #         for skin_type, module_name in skin_models.items():
# #             module = importlib.import_module(module_name)  # ✅ FIXED dynamic import
# #             confidence = module.predict_skin_disease(img_array)

# #             if confidence > highest_confidence:
# #                 highest_confidence = confidence
# #                 best_match = skin_type

# #         confidence_threshold = 0.6
# #         if highest_confidence < confidence_threshold:
# #             st.write("No Specific Skin Cancer Detected (-1)")
# #         else:
# #             st.write(f"Predicted Skin Disease: {best_match} (Confidence: {highest_confidence:.2f})")

# #     # Other cancer types (Retinoblastoma, etc.) accept RGB images
# #     else:
# #         model = cancer_models[selected_cancer_type]
# #         prediction = model.predict(img_array)
# #         max_confidence = np.max(prediction)
# #         confidence_threshold = 0.6

# #         if max_confidence < confidence_threshold:
# #             st.write("No Cancer Detected (-1)")
# #         else:
# #             st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

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

    # # Grayscale check for Brain Tumor & Lung Cancer
    # if selected_cancer_type in ["Brain Tumor", "Lung Cancer"]:
    #     # Convert image to grayscale if it's not already
    #     gray_image = image.convert("L")  # Convert to grayscale
    #     if image.mode != "L":  # If uploaded image was not grayscale
    #         st.error("Please upload a **black and white (grayscale)** image for Brain Tumor or Lung Cancer classification.")
    #     else:
    #         # Proceed with prediction
    #         prediction = selected_model.predict(selected_cancer_type)
    #         max_confidence = np.max(prediction)
    #         if max_confidence < confidence_threshold:
    #             st.write("No Cancer Detected (-1)")
    #         else:
    #             st.write(f"Predicted Cancer Type: {selected_cancer_type} (Confidence: {max_confidence:.2f})")

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
