# Teachable machine copy page

Edit [Hello.py](./Hello.py) to customize this app to your heart's desire. ❤️

Check it out on [Streamlit Community Cloud](https://st-hello-app.streamlit.app/)


(https://teachablemachine.withgoogle.com/train)

Firstly, you need to convert the TensorFlow.js model to a format compatible with TensorFlow or TensorFlow Lite. You can use the TensorFlow.js converter for this. Install the converter using:

pip install tensorflowjs

Then, convert the model:
tensorflowjs_converter --input_format=tfjs_layers_model --output_format=tf_saved_model ./your_model_directory ./saved_model_directory

pip install tensorflow streamlit opencv-python
