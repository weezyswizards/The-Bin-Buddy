import tensorflow as tf 
from tf import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Creating the model
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,3)),
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Change 3 to any number for more/less categories
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# *********************************************************************
#
#   TRAIN DATAGEN FROM PATH
#
#
#   VALIDATE DATEGEN FROM PATH
#
#
# **********************************************************************

# Training model
""" model.fit(...) """

# Save model
model.save('model.h5')
