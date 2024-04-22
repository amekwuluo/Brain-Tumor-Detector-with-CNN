import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("mri.h5")

# Define function to preprocess input image
def prepare(image):
    # Convert the BytesIO object to numpy array
    img_array = np.array(Image.open(image))
    img_array = img_array / 255.0
    resized_array = cv2.resize(img_array, (145, 145))
    return resized_array.reshape(-1, 145, 145, 3)

# Define function to check for tumor presence
def check(image):
    # Load and preprocess the image
    img = prepare(image)
    # Predict tumor presence
    output = model.predict(img)
    # Determine tumor presence
    tumor_present = output[0][0].round() == 0
    return tumor_present

# Streamlit app
st.title("Brain Tumor Detection")
st.write("Upload an MRI scanned image to check for tumor presence.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
    
    # Check for tumor presence
    tumor_present = check(uploaded_file)
    
    # Display the prediction
    if tumor_present:
        st.write("Prediction: Tumor Detected")
    else:
        st.write("Prediction: No Tumor Detected")
       

