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
    imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)

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


