from flask import Flask, render_template,url_for
import tensorflow as tf 
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.layers import Dense
import h5py as h5
import numpy as np 
import os

app = Flask(__name__)

if os.path.exists('model/waste_model.h5'):
    print("Model file exists.")
    try:
        model = tf.keras.models.load_model('model/waste_model.h5')
        print("Model loaded successfully!")
    except OSError as e:
        print(f"Error loading model: {e}")
else:
    print("Model file does not exist.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request contains the file part
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    # Check if a file is selected
    if file.filename == '':
        return 'No selected file'
    if file:
        # Save the uploaded file to the 'uploads' directory
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        
        # Load the image with the target size of (128, 128)
        img = image.load_img(img_path, target_size=(128, 128))
        # Convert the image to an array
        img_array = image.img_to_array(img)
        # Expand dimensions to match the model's input shape and normalize the image
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make a prediction using the loaded model
        prediction = model.predict(img_array)
        # Define the class labels
        classes = ['garbage', 'recycling', 'organic']
        # Get the predicted class based on the model's output
        predicted_class = classes[np.argmax(prediction)]
        
        # Return the predicted class
        return f'Predicted Class: {predicted_class}'


if __name__ == "__main__":
    app.run(debug=True)