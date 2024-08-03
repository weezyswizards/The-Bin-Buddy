import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py as h5

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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, 
                                        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
train_generator = train_datagen.flow_from_directory('./dataset/train', target_size=(128, 128), batch_size=32, class_mode='categorical')

# Validation Data Prep
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory('./dataset/validation', target_size=(128, 128), batch_size=32, class_mode='categorical')

# Training model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save model
model.save('model/waste_model.h5')
