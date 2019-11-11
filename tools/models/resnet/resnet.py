import matplotlib.pyplot as plt
import numpy as np
import os


from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Cropping2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K




trainImageDataGen = ImageDataGenerator(horizontal_flip=False)
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

model = Sequential()
model.add(Cropping2D(cropping=((100, 100), (100, 100)),
                     input_shape=(512, 512, 3)))

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(312, 312, 3))
for layer in resnet.layers:
    layer.trainable = False
model.add(resnet)

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
    trainGenerator,
    epochs=5,
    validation_data=validGenerator,
    verbose=1)

model.save("../../app/resources/models/resnet50.h5")

print(history.history.keys())
plt.plot(history.history['acc'])
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
