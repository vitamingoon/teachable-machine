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

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Increment the file count
    file_count += 1
    st.session_state.file_count = file_count

    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Display the uploaded image
    st.image(image, caption=f"Uploaded Image {file_count}.", use_column_width=True)

    # Make predictions
    predictions = predict_image(image)

    # Display predictions
    st.write("Class probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"Class {i}: {prob:.4f}")

# Display the file count
st.write(f"Total Uploaded Files: {file_count}")


# streamlit run your_script.py
