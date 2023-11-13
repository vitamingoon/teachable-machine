import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import zipfile
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

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "zip"])

if uploaded_file is not None:
    if uploaded_file.type == "application/zip":
        # Extract images from the zip file
        with zipfile.ZipFile(uploaded_file) as zip_file:
            image_files = [name for name in zip_file.namelist() if name.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for image_file in image_files:
                with zip_file.open(image_file) as file:
                    image_data = io.BytesIO(file.read())
                    image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), 1)

                    # Display the uploaded image
                    st.image(image, caption=f"Extracted Image: {image_file}", use_column_width=True)

                    # Make predictions
                    predictions = predict_image(image)

                    # Display predictions
                    st.write(f"Class probabilities for {image_file}:")
                    for i, prob in enumerate(predictions[0]):
                        st.write(f"Class {i}: {prob:.4f}")

    else:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Make predictions
        predictions = predict_image(image)

        # Display predictions
        st.write("Class probabilities:")
        for i, prob in enumerate(predictions[0]):
            st.write(f"Class {i}: {prob:.4f}")
