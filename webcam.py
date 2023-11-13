import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import io

# Load the TensorFlow SavedModel
model_path = "./saved_model_directory"  # Change to the path where you saved the converted model
model = tf.saved_model.load(model_path)

# Preprocess the image
def preprocess_image(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_array = np.expand_dims(normalized_frame, axis=0)
    return input_array

# Make predictions using the model
def predict_image(frame):
    input_array = preprocess_image(frame)
    result = model(input_array)
    return result

# Streamlit app
st.title("Teachable Machine Image Classifier")

# Counter for uploaded files
file_count = st.session_state.get("file_count", 0)

# File uploader for images
uploaded_image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Counter for captured webcam images
captured_image_count = st.session_state.get("captured_image_count", 0)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open webcam.")
else:
    st.write("Webcam is on. Press 'Capture' to capture an image.")

    # Button to capture an image
    if st.button('Capture'):
        # Read a single frame from the webcam
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Could not read frame.")
        else:
            # Increment the captured image count
            captured_image_count += 1
            st.session_state.captured_image_count = captured_image_count

            # Display the captured image
            st.image(frame, caption=f"Captured Image {captured_image_count}", use_column_width=True)

            # Make predictions
            predictions = predict_image(frame)

            # Display predictions
            st.write("Class probabilities:")
            for i, prob in enumerate(predictions[0]):
                st.write(f"Class {i}: {prob:.4f}")

# Display the total file count
st.write(f"Total Uploaded Files: {file_count}")
# Display the total captured image count
st.write(f"Total Captured Images: {captured_image_count}")
