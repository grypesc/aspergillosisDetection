import numpy as np
import os

from keras.applications.xception import Xception, preprocess_input
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator

def flipAndPreprocess(x):
    x = np.fliplr(x)
    x = preprocess_input(x)
    return x

if os.path.isfile('xception_train.csv'):
    os.remove("xception_train.csv")
file = open('xception_train.csv', 'a')

model = Sequential()
model.add(Cropping2D(cropping=((50, 50), (50, 50)), input_shape=(512, 512, 3)))
model.add(Xception(weights='imagenet', include_top=False, input_shape=(412, 412, 3), pooling='avg'))

preprocessingFunctions = [preprocess_input]

for preprocessingFunction in preprocessingFunctions:
    imageDataGen = ImageDataGenerator(preprocessing_function=preprocessingFunction)

    generator = imageDataGen.flow_from_directory(
        '../../../data/train/notFungus',
        target_size=(512, 512),
        batch_size=16,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 0)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

    generator = imageDataGen.flow_from_directory(
        '../../../data/train/fungus',
        target_size=(512, 512),
        batch_size=16,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 1)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

    generator = imageDataGen.flow_from_directory(
        '../../../data/train/notLungs',
        target_size=(512, 512),
        batch_size=16,
        class_mode=None)
    features = model.predict_generator(generator, verbose=1)
    labels = np.full((features.shape[0], 1), 2)
    np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

####### Validation features #######

if os.path.isfile('xception_validation.csv'):
    os.remove("xception_validation.csv")
file = open('xception_validation.csv','a')

imageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = imageDataGen.flow_from_directory(
    '../../../data/valid/notFungus',
    target_size=(512, 512),
    batch_size=16,
    class_mode=None)
features = model.predict_generator(generator, verbose=1)
labels = np.full((features.shape[0], 1), 0)
np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

generator = imageDataGen.flow_from_directory(
    '../../../data/valid/fungus',
    target_size=(512, 512),
    batch_size=16,
    class_mode=None)
features = model.predict_generator(generator, verbose=1)
labels = np.full((features.shape[0], 1), 1)
np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")

# generator = imageDataGen.flow_from_directory(
#     '../../../data/valid/notLungs',
#     target_size=(512, 512),
#     batch_size=64,
#     class_mode='categorical')
# features = model.predict_generator(generator, verbose=1)
# labels = np.full((features.shape[0], 1), 2)
# np.savetxt(file, np.append(features, labels, axis=1), delimiter=",")
