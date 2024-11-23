import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained Keras model (replace with the correct .keras file)
model = load_model('mask.keras')  # Load the Keras model

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the model's expected input shape (e.g., 224x224)
    target_size = (128, 128)  # Change this to the input size expected by your model
    image = image.resize(target_size)
    
    # Define the transformation
    preprocess = keras_image.img_to_array(image)  # Convert image to array
    preprocess = np.expand_dims(preprocess, axis=0)  # Add an extra dimension (for batch size)
    preprocess = preprocess / 255.0  # Normalize to [0, 1]
    return preprocess

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
    prediction = model.predict(input_image_reshaped)  # Predict using the Keras model
    input_pred_label = np.argmax(prediction, axis=1).item()  # Get the predicted class index

    # Display result
    if input_pred_label == 1:
        st.write("The person in the image is **not wearing a mask**.")
    else:
        st.write("The person in the image is **wearing a mask**.")
