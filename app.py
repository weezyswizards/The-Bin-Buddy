from flask import Flask, render_template, url_for, request
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np 
import os

app = Flask(__name__)

mapDic = {0 : "Garbage", 1 : "Compost", 2 : "Recycling"}

model = load_model('model/model.keras')

model.make_predict_function()

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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = request.files['file']

        img_path = "uploads/" + img.filename	
        img.save(img_path)

        p = predict_disp(img_path)
    
    return render_template("index.html", prediction = p, img_path = img_path)


if __name__ == "__main__":
    app.run(debug=True)