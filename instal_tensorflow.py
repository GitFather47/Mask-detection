import subprocess
import sys

# Check if TensorFlow is installed
try:
    import tensorflow as tf
    print("TensorFlow is already installed.")
except ImportError:
    print("TensorFlow is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])

# Check if Streamlit is installed
try:
    import streamlit
except ImportError:
    print("Streamlit is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
