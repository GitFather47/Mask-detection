import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# Load the trained model
model = keras.models.load_model('mask.keras')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    image_resized = cv2.resize(image, (128, 128))  # Resize to match model input
    image_scaled = image_resized / 255.0  # Scale pixel values
    image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])  # Reshape for model input
    return image_reshaped

# Streamlit app
st.title("Face Mask Detection")
st.write("Upload an image to check if the person is wearing a mask.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    input_image_reshaped = preprocess_image(image)

    # Make prediction
    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    # Display result
    if input_pred_label == 1:
        st.write("The person in the image is **not wearing a mask**.")
    else:
        st.write("The person in the image is **wearing a mask**.")