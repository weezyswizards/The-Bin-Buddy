from flask import Flask, render_template, url_for, request, send_from_directory
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np 
import os

app = Flask(__name__)

# Configure and load ML model
mapDic = {0 : "Garbage", 1 : "Compost", 2 : "Recycling"}
model = load_model('model/model.keras')
model.make_predict_function()

#Configuration for image uploads
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ML prediction function
def predict_disp(img_path):
    try:
        i = image.load_img(img_path, target_size=(150, 150))
        i = image.img_to_array(i) / 255.0
        i = i.reshape(1, 150, 150, 3)
        p = model.predict(i)
        class_idx = np.argmax(p, axis=1)[0]
        return mapDic[class_idx]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Showing result
@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = request.files['file']

        # Saving the uploaded image
        img_path = UPLOAD_FOLDER + img.filename	
        img.save(img_path)

        # Call ML function for prediction
        p = predict_disp(img_path)
    
        return render_template("index.html", prediction = p, img_path = img_path)
    return render_template("index.html")

# Serve the uploaded image=
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)