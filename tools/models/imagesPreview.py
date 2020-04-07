import numpy as np
import os, sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.applications.mobilenet_v2 import preprocess_input

np.set_printoptions(threshold=sys.maxsize)


def flipAndPreprocess(x):
    x = np.fliplr(x)
    # x = preprocess_input(x)
    return x

preprocessingFunctions = [flipAndPreprocess]

for preprocessingFunction in preprocessingFunctions:
    imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input, width_shift_range=30, height_shift_range=30,
                                  rotation_range=10, brightness_range=[0.90, 1.10],
                                  shear_range=5, fill_mode='constant', cval=0, zoom_range=0.05)

    generator = imageDataGen.flow_from_directory(
        '../../data/train/fungus/',
        target_size=(512, 512),
        batch_size=1,
        shuffle=False,
        class_mode=None)

    for i in range(100):
        batch = generator.next()
        image = batch[0]
        image += 1
        image *= 128
        image = image.astype(int)
        print(image[255, :])
        plt.imshow(image)
        plt.show()


