import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Hugging Face Model URL (Direct Download Link)
MODEL_URL = "https://huggingface.co/aarayann/skin_cancer_cnn.h5/resolve/main/skin_cancer_cnn.h5"
MODEL_PATH = "skin_cancer_cnn.h5"

# Function to download the model if not present
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait ⏳"):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()  # Raise error if download failed
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192  # 8 KB
                progress_bar = st.progress(0)
                
                downloaded_size = 0
                with open(MODEL_PATH, "wb") as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        downloaded_size += len(data)
                        if total_size > 0:
                            progress_percent = downloaded_size / total_size
                            progress_bar.progress(min(progress_percent, 1.0))
                
                progress_bar.empty()  # Remove the progress bar when done
                st.success("Model downloaded successfully! ✅")
            
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                raise e

    return load_model(MODEL_PATH)

# Load the trained model
model = download_model()

# Function to preprocess and predict the image
def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label

# Streamlit App UI
st.title("🩺 Skin Cancer Detection App")
st.markdown("""
    Upload an image, and the model will predict whether the skin lesion is **Malignant** or **Benign**.
""")

# File uploader
uploaded_image = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    class_label = predict_skin_cancer(uploaded_image, model)

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
