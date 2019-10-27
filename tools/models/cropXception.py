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


folderFungus = '../../data/fungus'
folderNoFungus = '../../data/noFungus'
folderNoLungs = '../../data/noLungs'

fungusImages, noFungusImages = 0, 0
for dirpath, subdirs, files in os.walk(folderFungus):
    fungusImages += len(files)
print("Number of Fungus images: " + str(fungusImages))

for dirpath, subdirs, files in os.walk(folderNoFungus):
    noFungusImages += len(files)
print("Number of noFungus images: " + str(noFungusImages))

trainX = np.zeros(shape=(fungusImages+noFungusImages,299, 299, 3), dtype = "float16")

index = 0
for folder in [folderFungus, folderNoFungus]:
    for dirpath, subdirs, files in os.walk(folder):
        for file in files:
            img = pydicom.read_file(dirpath + "/" + file).pixel_array
            img = img[105:404,105:404].reshape(299,299)
            stackedImg = np.stack((img,)*3, axis=-1)
            trainX[index] = stackedImg
            index+=1


print("Finished loading data to RAM")

trainX += 2048
trainX /= 4096

trainY = np.zeros(trainX.shape[0])
trainY[0:fungusImages] = 1
trainY = to_categorical(trainY, num_classes= 2)

s = np.arange(trainX.shape[0])
np.random.shuffle(s)
trainX = trainX[s]
trainY = trainY[s]

xception = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

for layer in xception.layers:
    layer.trainable = False

model = Sequential()

# Add the vgg convolutional base model
model.add(xception)

# Add new layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# # Show a summary of the model. Check the number of trainable parameters
# model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])

history = model.fit(trainX, trainY, validation_split=0.2,
                    epochs= 5, batch_size= 32, verbose=2)

model.save("../../app/resources/models/Xception.h5")

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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
