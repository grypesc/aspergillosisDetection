import matplotlib.pyplot as plt
import numpy as np
import os


from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Cropping2D, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator




trainImageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
trainGenerator = trainImageDataGen.flow_from_directory(
    '../../../data/train',
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical')

validImageDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
validGenerator = validImageDataGen.flow_from_directory(
    '../../../data/valid',
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical')

model = Sequential()
model.add(Cropping2D(cropping=((100, 100), (100, 100)), input_shape=(512, 512, 3)))

model.add(MobileNetV2(weights='imagenet', include_top=False, input_shape=(312, 312, 3)))
for layer in model.layers:
    layer.trainable = False

model.add( GlobalAveragePooling2D(data_format='channels_last'))


model.add(Dense(2, activation='softmax'))


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Nadam(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
    trainGenerator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validGenerator,
    verbose=2)

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
