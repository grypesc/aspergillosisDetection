import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import sys
np.set_printoptions(threshold=sys.maxsize)

from keras.models import load_model
from keras.utils.np_utils import to_categorical

folderFungus = 'data/fungus'
folderNoFungus = 'data/noFungus'
folderNoLungs = 'data/noLungs'
folderTest = 'data/test'

fungusImages, noFungusImages = 0, 0
for dirpath, subdirs, files in os.walk(folderTest):
    fungusImages += len(files)
print("Number of test fungus images: " + str(fungusImages))

testX = np.zeros(shape=(fungusImages,512, 512, 1), dtype = "float16")

index = 0
for dirpath, subdirs, files in os.walk(folderTest):
    for file in files[:]:
        testX[index] = pydicom.read_file(dirpath + "/" + file).pixel_array.reshape(512,512,1)
        index+=1

testX /= 2048

testY = np.ones(testX.shape[0])
testY = to_categorical(testY, num_classes= 2)

model = load_model('models/doubleConv.h5')
yPredictions = model.predict(x=testX, batch_size=32, verbose=1)
print(yPredictions)
accuracy = np.sum(yPredictions[:,1])/fungusImages
print(accuracy)
