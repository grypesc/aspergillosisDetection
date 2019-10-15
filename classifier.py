import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import sys
np.set_printoptions(threshold=sys.maxsize)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam


folderFungus = 'data/fungus'
folderNoFungus = 'data/noFungus'
folderNoLungs = 'data/noLungs'

fungusImages, noFungusImages = 0, 0
for dirpath, subdirs, files in os.walk(folderFungus):
    fungusImages += len(files)
print("Number of Fungus images: " + str(fungusImages))

for dirpath, subdirs, files in os.walk(folderNoFungus):
    noFungusImages += len(files)
print("Number of noFungus images: " + str(noFungusImages))

trainX = np.zeros(shape=(fungusImages+noFungusImages,512, 512, 1), dtype = "float16")

index = 0
for dirpath, subdirs, files in os.walk(folderFungus):
    for file in files[:]:
        trainX[index] = pydicom.read_file(dirpath + "/" + file).pixel_array.reshape(512,512,1)
        index+=1

for dirpath, subdirs, files in os.walk(folderNoFungus):
    for file in files[:]:
        trainX[index] = pydicom.read_file(dirpath + "/" + file).pixel_array.reshape(512,512,1)
        index+=1

print("Finished loading data to RAM")

trainX /= 2048

trainY = np.zeros(trainX.shape[0])
trainY[0:fungusImages] = 1
trainY = to_categorical(trainY, num_classes= 2)

s = np.arange(trainX.shape[0])
np.random.shuffle(s)
trainX = trainX[s]
trainY = trainY[s]


def build(input_shape, lr = 1e-4, num_classes= 2, init= 'normal'):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),padding = "valid", strides = 2, input_shape=input_shape,
                     activation= 'relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Conv2D(16, kernel_size=(3, 3), padding = "valid", strides = 2,
                     activation ='relu', kernel_initializer = 'glorot_uniform'))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=init))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(optimizer = Adam(lr=lr), loss = "binary_crossentropy", metrics=["accuracy"])
    return model

input_shape = (512,512,1)
lr = 1e-4
init = 'normal'
epochs = 5
batch_size = 64

model = build(lr=lr, init= init, input_shape= input_shape)

history = model.fit(trainX, trainY, validation_split=0.2,
                    epochs= epochs, batch_size= batch_size, verbose=2
                   )


print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['validation'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
