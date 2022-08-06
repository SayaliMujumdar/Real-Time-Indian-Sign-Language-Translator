import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import cv2

def train_val_generators(TRAINING_DIR, VALIDATION_DIR):

  # Instantiate the ImageDataGenerator class
  train_datagen = ImageDataGenerator(rescale=1./255,zoom_range = 0.3,
                                     width_shift_range=0.2,height_shift_range=0.2
                                     )

  # Pass the images from directory to the gernerator  the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=16,
                                                      class_mode='categorical',
                                                      target_size=(28,28),
                                                      color_mode='grayscale'
                                                      )

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator(rescale=1./255)

  # Pass in the appropiate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=16,
                                                                class_mode='categorical',
                                                                target_size = (28, 28),
                                                                color_mode = 'grayscale'
                                                                )
  return train_generator, validation_generator
def create_model():
  model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(15, (3, 3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax'),

 ])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
               metrics=["accuracy"])
  return model


TRAINING_DIR="C:/Users/91992/Documents/Project Documents/Dataset/training"
VALIDATION_DIR="C:/Users/91992/Documents/Project Documents/Dataset/testing"
train_generator,validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

model = create_model()

# Train the model
history = model.fit(train_generator,
                    epochs=2,
                    verbose=1,
                    validation_data=validation_generator)

model.summary()

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("cnnpreprocessed1.model")
model.save("savedmodelonself.h5")
print("saved model to the disk")

'''
def create_model():
 model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(128, 128, 1),activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(112, activation='relu'),
     tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(96, activation='relu'),
     tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(80, activation='relu'),
     tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])
'''