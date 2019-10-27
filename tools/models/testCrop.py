import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import sys
np.set_printoptions(threshold=sys.maxsize)

from keras.models import load_model
from keras.utils.np_utils import to_categorical

def generateProbabilityPlot(X, Y):
    plt.figure(figsize=(8,2))
    plt.xlim([0, len(X)])
    plt.ylim([0, 1])
    plt.plot(X, Y)
    plt.ylabel('Fungus probability')
    plt.show()

folderFungus = '../../data/fungus'
folderNoFungus = '../../data/noFungus'
folderNoLungs = '../../data/noLungs'
folderTest = '../../data/fungus'

fungusImages, noFungusImages = 0, 0
for dirpath, subdirs, files in os.walk(folderTest):
    fungusImages += len(files)
print("Number of test fungus images: " + str(fungusImages))

testX = np.zeros(shape=(fungusImages,299, 299, 3), dtype = "float16")

index = 0
for dirpath, subdirs, files in os.walk(folderTest):
    for file in files[:]:
        img = pydicom.read_file(dirpath + "/" + file).pixel_array
        img = img[105:404,105:404].reshape(299,299)
        stackedImg = np.stack((img,)*3, axis=-1)
        testX[index] = stackedImg
        index+=1

testX += 2048
testX /= 4096



model = load_model('../../app/resources/models/Xception.h5')
Y = model.predict(x=testX, batch_size=64, verbose=1)

print(Y)
accuracy = np.sum(Y[:,1])/fungusImages
print(accuracy)
generateProbabilityPlot([i for i in range(testX.shape[0])], Y[:,1])
