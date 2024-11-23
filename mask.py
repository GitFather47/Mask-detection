import streamlit as st
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load the trained model (replace with .pt or .pth file)
model = torch.load('mask_model.pth')  # Load the PyTorch model
model.eval()  # Set the model to evaluation mode

# Function to preprocess the image
def preprocess_image(image):
    # Define the transformation
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    # Apply the transformations
    image_tensor = preprocess(image)
    # Add an extra dimension (for batch size) and ensure shape is [1, 3, 128, 128]
    image_reshaped = image_tensor.unsqueeze(0)  # Adds a batch dimension
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
    with torch.no_grad():  # Disable gradient calculation
        input_prediction = model(input_image_reshaped)
        input_pred_label = torch.argmax(input_prediction).item()  # Get the predicted class index

    # Display result
    if input_pred_label == 1:
        st.write("The person in the image is **not wearing a mask**.")
    else:
        st.write("The person in the image is **wearing a mask**.")
