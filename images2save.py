import cv2
import os
import shutil
from datetime import datetime

def capture_images_and_save(output_directory="captured_images"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    image_count = 0
    while True:
        # Read a single frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Save the captured image
        image_count += 1
        image_filename = f"{output_directory}/captured_image_{image_count}.jpg"
        cv2.imwrite(image_filename, frame)
        print(f"Image {image_count} captured and saved as {image_filename}")

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # Create a zip file
    zip_filename = f"{output_directory}_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    shutil.make_archive(zip_filename, 'zip', output_directory)
    print(f"Images zipped and saved as {zip_filename}")

    # Remove the individual image files if needed
    # shutil.rmtree(output_directory)

if __name__ == "__main__":
    # Capture images until 'q' key is pressed and save them
    capture_images_and_save()
