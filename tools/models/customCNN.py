import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Cropping2D, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

trainImageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input, width_shift_range=30,
                                       height_shift_range=30, rotation_range=20, brightness_range=[0.90, 1.10],
                                       shear_range=5, fill_mode='constant', cval=0, zoom_range=0.05,
                                       horizontal_flip=True)
trainGenerator = trainImageDataGen.flow_from_directory(
    '../../data/train',
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical')

validImageDataGen = ImageDataGenerator()
validGenerator = validImageDataGen.flow_from_directory(
    '../../data/valid',
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical')

modelMobile = MobileNetV2(weights='imagenet', include_top=False, input_shape=(412, 412, 3), pooling='avg')
# modelMobile = Model(modelMobile.input, modelMobile.layers[26].output)
modelMobile.summary()
exit()

model = Sequential()
model.add(Cropping2D(cropping=((50, 50), (50, 50)), input_shape=(512, 512, 3)))
model.add(modelMobile)
for l in model.layers:
    l.trainable = False

model.add(Flatten())
# model.add(Dense(100, activation='relu'))
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
