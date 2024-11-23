import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('mask.keras')

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to numpy array and resize
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0  # Normalize pixel values
    # Add an extra dimension (for batch size) and ensure shape is [1, 128, 128, 3]
    image_reshaped = np.expand_dims(image_array, axis=0)  # Adds a batch dimension
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
