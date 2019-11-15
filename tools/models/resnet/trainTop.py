import matplotlib.pyplot as plt
import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

trainData = np.loadtxt('resnet_train.csv', delimiter=",")
s = np.arange(trainData.shape[0])
np.random.shuffle(s)
trainData = trainData[s]

labels = trainData[:,len(trainData[0])-1]
labels = to_categorical(labels, num_classes=2)

validationData = np.loadtxt('resnet_validation.csv', delimiter=",")

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=2*1e-5),
              metrics=['acc'])

history = model.fit(
    x=trainData[:, :-1],
    y=labels,
    epochs=100,
    batch_size=1024,
    validation_data=(validationData[:,:-1], to_categorical(validationData[:,-1], num_classes=2)),
    verbose=2)

model.save("resnetTop.h5")

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
