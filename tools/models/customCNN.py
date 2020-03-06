import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Cropping2D, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

trainImageDataGen = ImageDataGenerator(horizontal_flip=False)
trainGenerator = trainImageDataGen.flow_from_directory(
    '../../data/train',
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale')

validImageDataGen = ImageDataGenerator()
validGenerator = validImageDataGen.flow_from_directory(
    '../../data/valid',
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale')

model = Sequential()
model.add(Cropping2D(cropping=((100, 100), (100, 100)), input_shape=(512, 512, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="valid", strides=2, input_shape=(312, 312, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))

model.add(Conv2D(32, kernel_size=(3, 3), padding="valid", strides=2, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(3, 3), padding="valid", strides=2, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
    trainGenerator,
    epochs=50,
    validation_data=validGenerator,
    verbose=2)

# model.save("../../app/resources/models/lol.h5")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
