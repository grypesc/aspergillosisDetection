import matplotlib.pyplot as plt
import numpy as np


from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Cropping2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import flip_left_right

def flipAndPreprocess(x):
    x = np.fliplr(x)
    x = preprocess_input(x)
    return x

file = open('resnet_train.csv','a')

model = Sequential()
model.add(Cropping2D(cropping=((100, 100), (100, 100)),
                     input_shape=(512, 512, 3)))

resnet = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

model.add(resnet)


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])

preprocessingFunctions = [preprocess_input, flipAndPreprocess]
for preprocessingFunction in preprocessingFunctions:
    imageDataGen = ImageDataGenerator(preprocessing_function=preprocessingFunction)

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

    generator = imageDataGen.flow_from_directory(
        '../../../data/train/notLungs',
        target_size=(512, 512),
        batch_size=32,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 2)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")
