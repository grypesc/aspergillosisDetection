import numpy as np
import os

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def flipAndPreprocess(x):
    x = np.fliplr(x)
    x = preprocess_input(x)
    return x


if os.path.isfile('vgg19_train.csv'):
    os.remove("vgg19_train.csv")
file = open('vgg19_train.csv', 'a')

model = Sequential()
model.add(Cropping2D(cropping=((50, 50), (50, 50)), input_shape=(512, 512, 3)))
model.add(VGG19(weights='imagenet', include_top=False, input_shape=(412, 412, 3), pooling='avg'))
for l in model.layers:
    l.trainable = False

preprocessingFunctions = [preprocess_input]

for preprocessingFunction in preprocessingFunctions:
    imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input, width_shift_range=30,
                                      height_shift_range=30,
                                      rotation_range=20, brightness_range=[0.90, 1.10],
                                      shear_range=8, fill_mode='constant', cval=0, zoom_range=0.075,
                                      horizontal_flip=True)
    generator = imageDataGen.flow_from_directory(
        '../../../data/train/notFungus',
        target_size=(512, 512),
        batch_size=32,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 0)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

    generator = imageDataGen.flow_from_directory(
        '../../../data/train/fungus',
        target_size=(512, 512),
        batch_size=32,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 1)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

####### Validation features #######

if os.path.isfile('vgg19_validation.csv'):
    os.remove("vgg19_validation.csv")
file = open('vgg19_validation.csv', 'a')

imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input, width_shift_range=30,
                                  height_shift_range=30,
                                  rotation_range=20, brightness_range=[0.90, 1.10],
                                  shear_range=8, fill_mode='constant', cval=0, zoom_range=0.075,
                                  horizontal_flip=True)
generator = imageDataGen.flow_from_directory(
    '../../../data/valid/notFungus',
    target_size=(512, 512),
    batch_size=32,
    class_mode=None)
features = model.predict_generator(generator, verbose=1)
labels = np.full((features.shape[0], 1), 0)
np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

generator = imageDataGen.flow_from_directory(
    '../../../data/valid/fungus',
    target_size=(512, 512),
    batch_size=32,
    class_mode=None)
features = model.predict_generator(generator, verbose=1)
labels = np.full((features.shape[0], 1), 1)
np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")
