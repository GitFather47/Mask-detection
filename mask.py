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

# Streamlit app styling using markdown and custom CSS
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f0f0;
            color: #333;
        }
        .header {
            text-align: center;
            color: #4CAF50;
        }
        .nav-bar {
            background-color: #4CAF50;
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
        }
        .container {
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .uploaded-image {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .prediction {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .prediction h4 {
            color: #333;
            font-weight: bold;
        }
        .prediction p {
            font-size: 18px;
            color: #4CAF50;
        }
        .upload-container {
            margin-top: 20px;
            text-align: center;
        }
        .about-section {
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .about-section h3 {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation Bar (Sidebar for easy navigation)
st.sidebar.title("Navigation")
nav_option = st.sidebar.radio("Go to:", ["Home", "About"])

# Home Page: Face Mask Detection
if nav_option == "Home":
    st.markdown("<div style='text-align:center'><h1 class='header' style='font-family:Ink Free;'>MaskMate - Face Mask Detection😷</h1></div>", unsafe_allow_html=True)
    st.write("Upload an image to check if the person is wearing a mask.")
    
    # Upload image section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"], key="image_uploader")

    if uploaded_file is not None:
        # Display the uploaded image with custom styling using HTML
        image = Image.open(uploaded_file)
        st.markdown(f"""
            <div style='text-align:center'>
                <img src="data:image/png;base64,{uploaded_file.getvalue().decode('utf-8')}" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-top: 20px;" width="80%"/>
            </div>
        """, unsafe_allow_html=True)
        
        # Preprocess the image
        input_image_reshaped = preprocess_image(image)

        # Make prediction
        prediction = model.predict(input_image_reshaped)  # Predict using the Keras model
        input_pred_label = np.argmax(prediction, axis=1).item()  # Get the predicted class index

        # Display prediction result with styling
        prediction_text = ""
        if input_pred_label == 1:
            prediction_text = "The person in the image is **not wearing a mask**."
        else:
            prediction_text = "The person in the image is **wearing a mask**."

        st.markdown(f"""
            <div class="prediction">
                <h4>Prediction Result</h4>
                <p>{prediction_text}</p>
            </div>
        """, unsafe_allow_html=True)

# About Section
elif nav_option == "About":
    st.markdown("<h1 class='header'>About This App</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class="about-section">
            <h3>Face Mask Detection App</h3>
            <p>This application uses a CNN model to detect whether a person in a photo is wearing a mask. The model has been trained on a dataset of images and is capable of classifying images as either 'Mask' or 'No Mask'.</p>
            <p>It is designed to help in environments where face masks are required, such as health-related settings and public spaces. Simply upload an image, and the model will predict whether the person in the image is wearing a mask or not.
            Dataset link: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset</p>
        </div>
    """, unsafe_allow_html=True)
    st.write("#### Credits:")
    st.image("about.jpg")
    st.write("Arnob Aich Anurag")
    st.write("Student at American International University Bangladesh")
    st.write("Dhaka, Bangladesh")
    st.write("For more information, please contact me at my email.")
    st.write("Email: aicharnob@gmail.com")
