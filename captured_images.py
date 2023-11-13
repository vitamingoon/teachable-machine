import cv2
import streamlit as st
import os
import shutil
from datetime import datetime

def capture_images_and_save(output_directory="captured_images"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    image_count = 0
    st.image([], caption='Webcam Feed', use_column_width=True, channels='BGR')
    st.write("Press the 'Capture' button to save an image. Press 'Finish' when done.")

    while True:
        # Read a single frame from the webcam
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Could not read frame.")
            break

        # Display the frame
        st.image(frame, caption=f'Captured Image {image_count}', use_column_width=True, channels='BGR')

        # Check if the user pressed the 'Capture' button
        if st.button('Capture'):
            # Save the captured image
            image_count += 1
            image_filename = f"{output_directory}/captured_image_{image_count}.jpg"
            cv2.imwrite(image_filename, frame)
            st.success(f"Image {image_count} captured and saved as {image_filename}")

        # Check if the user pressed the 'Finish' button
        if st.button('Finish'):
            break

    # Release the webcam
    cap.release()

    # Create a zip file
    zip_filename = f"{output_directory}_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    shutil.make_archive(zip_filename, 'zip', output_directory)
    st.success(f"Images zipped and saved as {zip_filename}")

    # Remove the individual image files if needed
    # shutil.rmtree(output_directory)

if __name__ == "__main__":
    # Streamlit app
    st.title("Webcam Image Capture App")
    capture_images_and_save()
