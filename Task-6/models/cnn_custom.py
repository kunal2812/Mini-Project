from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
from pathlib import Path

NUM_CLASSES = 4


class Hist:
    """ Dummy class

    """
    def __init__(self):
        pass

def main():
    """ Load data.
    Normalize and encode.
    Train custom CNN model.
    Print accuracy on test data.

    """

    train_dir = Path('../dataset/train')

    training_datagen = ImageDataGenerator(rescale = 1./255.0,     #Each pixel value is divided by 255(maximum value of any pixel)
                                        rotation_range = 90,     #Image is rotated randomly in a range of (0 - 90) degrees
                                        width_shift_range = 0.4, #Width is changed by 40%
                                        height_shift_range = 0.4,#Height is changed by 40%
                                        shear_range = 0.2,       #Changes orientation of image
                                        zoom_range = 0.2,        #Crops the image by zooming into it by 20%
                                        horizontal_flip = True,  #Horizontal Flipping is True
                                        fill_mode = 'nearest',   #If augmentation results in dead pixels in any image then that is fixed according to pixel values of the 'nearest' neighbours
                                        validation_split = 0.2
                                        )
    train_generation = training_datagen.flow_from_directory(train_dir,
                                                       subset='training',
                                                       target_size = (180, 180),    #Images are resized to (224,224) as mentioned in the original paper
                                                       class_mode = 'categorical', #Since we have 6 classes
                                                       batch_size = 16)

    print(train_generation.class_indices)
    print('For Validation:')
    valid_generation = training_datagen.flow_from_directory(train_dir,
                                                            subset='validation',
                                                            target_size = (180, 180),
                                                            class_mode = 'categorical',
                                                            batch_size = 16)

    # Custom CNN model
    model_custom = Sequential((
        layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                      input_shape=(180, 180, 3)),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128),
        layers.Activation('relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')))

    model_custom.compile(optimizer=Adam(),
                         loss="categorical_crossentropy",
                         metrics=['accuracy'])

    model_history = model_custom.fit(train_generation, epochs = 5, validation_data = valid_generation, verbose = 1) 

    print(model_custom.summary())

    # save model
    model_custom.save('../results/models/custom.h5')


if __name__ == "__main__":
    main()
