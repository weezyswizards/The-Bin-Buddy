from flask import Flask, render_template,url_for
#import tensorflow as tf 
#from tensorflow.keras.preprocessing import image
#import numpy as np 
import os

app = Flask(__name__)
""" model = tf.keras.models.load_model('model/waste_model.h5')"""

@app.route('/')
def index():
    return render_template('index.html')

# ***********************************************************
#
#
# APP ROUTE FOR MODEL INTEGRATION TO FLASK
#
#
# ***********************************************************

if __name__ == "__main__":
    app.run(debug=True)