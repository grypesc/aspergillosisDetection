import numpy as np
import os

from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def flipAndPreprocess(x):
    x = np.fliplr(x)
    x = preprocess_input(x)
    return x


if os.path.isfile('mobileNetV2_train.csv'):
    os.remove("mobileNetV2_train.csv")
file = open('mobileNetV2_train.csv', 'a')

model = Sequential()
model.add(Cropping2D(cropping=((50, 50), (50, 50)), input_shape=(512, 512, 3)))
model.add(NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg'))

preprocessingFunctions = [preprocess_input]

for preprocessingFunction in preprocessingFunctions:
    imageDataGen = ImageDataGenerator(preprocessing_function=preprocessingFunction, width_shift_range=30, height_shift_range=30, rotation_range=20, brightness_range=[0.80, 1.20], shear_range=5, horizontal_flip=True)
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

####### Validation features #######

if os.path.isfile('mobileNetV2_validation.csv'):
    os.remove("mobileNetV2_validation.csv")
file = open('mobileNetV2_validation.csv', 'a')

imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
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

# generator = imageDataGen.flow_from_directory(
#     '../../../data/valid/notLungs',
#     target_size=(512, 512),
#     batch_size=64,
#     shuffle = True,
#     class_mode='categorical')
# features = model.predict_generator(generator, verbose=1)
# labels = np.full((features.shape[0], 1), 2)
# np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")
