import cv2
import os
import tensorflow as tf
import numpy as np

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path):
    # Read the image from the file
    frame = cv2.imread(image_path)

    # Resize the image to the size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))

    # Preprocess the image (normalize pixel values to be between 0 and 1)
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to create a batch (model expects a batch of images)
    input_array = np.expand_dims(normalized_frame, axis=0)
    return input_array

def main(image_directory):
    # Load the Teachable Machine model
    model_path = "path/to/your/model"  # Replace with the path to your model file
    model = load_model(model_path)

    # List all files in the specified directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg', '.zip'))]

    if not image_files:
        print(f"No images found in {image_directory}")
        return

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)

        # Preprocess the captured frame
        input_array = preprocess_image(image_path)

        # Make predictions using the model
        predictions = model.predict(input_array)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions)

        # Display the result
        print(f"Image: {image_file}, Predicted Class: {predicted_class}")

if __name__ == "__main__":
    # Specify the directory containing the images
    image_directory = "path/to/your/images"  # Replace with the path to your image directory
    main(image_directory)
