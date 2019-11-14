import matplotlib.pyplot as plt
import numpy as np
import os

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import flip_left_right

def flipAndPreprocess(x):
    x = np.fliplr(x)
    x = preprocess_input(x)
    return x

if os.path.isfile('resnet_train.csv'):
    os.remove("resnet_train.csv")
file = open('resnet_train.csv','a')

model = Sequential()
model.add(Cropping2D(cropping=((100, 100), (100, 100)), input_shape=(512, 512, 3)))
model.add(ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3)))

preprocessingFunctions = [preprocess_input, flipAndPreprocess]

for preprocessingFunction in preprocessingFunctions:
    imageDataGen = ImageDataGenerator(preprocessing_function=preprocessingFunction)

    generator = imageDataGen.flow_from_directory(
        '../../../data/train/notFungus',
        target_size=(512, 512),
        batch_size=128,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 0)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

    generator = imageDataGen.flow_from_directory(
        '../../../data/train/fungus',
        target_size=(512, 512),
        batch_size=128,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 1)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

    generator = imageDataGen.flow_from_directory(
        '../../../data/train/notLungs',
        target_size=(512, 512),
        batch_size=128,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 2)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")
