import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

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
            # Display the captured image
            st.image(frame, caption="Captured Image", use_column_width=True)

            # Make predictions
            predictions = predict_image(frame)

            # Display predictions
            st.write("Class probabilities:")
            for i, prob in enumerate(predictions[0]):
                st.write(f"Class {i}: {prob:.4f}")

    # Release the webcam when the app is closed
    st.button('Stop Webcam')  # This is a dummy button to stop the webcam when pressed

    # Closing the webcam
    if st.button('Stop Webcam'):
        cap.release()
        st.write("Webcam stopped.")
