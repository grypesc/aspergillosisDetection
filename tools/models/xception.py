import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import sys
np.set_printoptions(threshold=sys.maxsize)

from keras.applications import Xception
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


dirTrain = '../../data/train/fungus'
dirTrainNotFungus = '../../data/train/notFungus'
dirValidFungus = '../../data/train/fungus'
dirValidNotFungus = '../../data/train/notFungus'

fungusTrainCount, notFungusTrainCount = 0, 0
fungusValidCount, notFungusValidCount = 0, 0

for dirpath, subdirs, files in os.walk(dirTrainFungus):
    fungusTrainCount += len(files)
print("Number of fungus train images: " + str(fungusTrainCount))

for dirpath, subdirs, files in os.walk(dirTrainNotFungus):
    notFungusTrainCount += len(files)
print("Number of not fungus train images: " + str(notFungusTrainCount))

for dirpath, subdirs, files in os.walk(dirValidFungus):
    fungusValidCount += len(files)
print("Number not fungus valid images: " + str(fungusValidCount))

for dirpath, subdirs, files in os.walk(dirValidNotFungus):
    notFungusValidCount += len(files)
print("Number of not fungus valid images: " + str(notFungusValidCount))


trainImageDataGen = ImageDataGenerator(horizontal_flip=False)
trainGenerator = trainImageDataGen.flow_from_directory(
    '../../data/train',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')

validImageDataGen = ImageDataGenerator()
validGenerator = validImageDataGen.flow_from_directory(
    '../../data/validation',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')

xception = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

for layer in xception.layers:
    layer.trainable = False

model = Sequential()

model.add(xception)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
    trainGenerator,
    epochs=5,
    validation_data=validGenerator)

model.save("../../app/resources/models/Xception.h5")

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
