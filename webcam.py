import cv2
import tensorflow as tf
import numpy as np

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(frame):
    # Resize the image to the size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))
    # Preprocess the image (normalize pixel values to be between 0 and 1)
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to create a batch (model expects a batch of images)
    input_array = np.expand_dims(normalized_frame, axis=0)
    return input_array

def main():
    # Load the Teachable Machine model
    model_path = "path/to/your/model"  # Replace with the path to your model file
    model = load_model(model_path)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a single frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the captured frame
        input_array = preprocess_image(frame)

        # Make predictions using the model
        predictions = model.predict(input_array)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions)

        # Display the result on the frame
        cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Teachable Machine", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
