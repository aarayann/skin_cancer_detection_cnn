import streamlit as st
import numpy as np
import gdown
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Google Drive File ID & Model File Name
file_id = "1YA20yqCGAvE2ZEV12d_OJSd0pknW3YeB"
model_filename = "skin_cancer_cnn.h5"

# Download Model from Google Drive (if not already present)
if not os.path.exists(model_filename):
    with st.spinner("Downloading model... Please wait."):
        model_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(model_url, model_filename, quiet=False)

# Load the trained model
model = load_model(model_filename)

# Function to preprocess and predict the image
def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, img


# Streamlit App UI
st.title("🩺 Skin Cancer Detection App")
st.markdown("""
    Upload an image, and the model will predict whether the skin lesion is **Malignant** or **Benign**.
""")

# File uploader
uploaded_image = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Predict and display results
    class_label, img = predict_skin_cancer(uploaded_image, model)

    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write(f"🎯 **Prediction: {class_label}**")

# About the model
st.markdown("""
    ### ℹ️ About the Model:
    This model is a CNN-based classifier trained to distinguish between **Benign** and **Malignant** skin lesions.

    **How to use:**
    1. 📤 Upload an image of a skin lesion.
    2. 🏥 The model will predict if it's **Benign** or **Malignant**.
""")
