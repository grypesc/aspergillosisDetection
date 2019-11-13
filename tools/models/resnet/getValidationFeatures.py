import matplotlib.pyplot as plt
import numpy as np
import os

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Cropping2D
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator

if os.path.isfile('resnet_validation.csv'):
    os.remove("resnet_validation.csv")
file = open('resnet_validation.csv','a')

model = Sequential()
model.add(Cropping2D(cropping=((100, 100), (100, 100)), input_shape=(512, 512, 3)))
model.add(ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3)))

imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)

generator = imageDataGen.flow_from_directory(
    '../../../data/valid/notFungus',
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical')
features = model.predict_generator(generator, verbose=1)
labels = np.full((features.shape[0], 1), 0)
np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

generator = imageDataGen.flow_from_directory(
    '../../../data/valid/fungus',
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical')
features = model.predict_generator(generator, verbose=1)
labels = np.full((features.shape[0], 1), 1)
np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

generator = imageDataGen.flow_from_directory(
    '../../../data/valid/notLungs',
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical')
features = model.predict_generator(generator, verbose=1)
labels = np.full((features.shape[0], 1), 2)
np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")
