import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom

from matplotlib.figure import Figure
from keras.models import load_model
from PyQt5.QtCore import QObject, pyqtSignal

from .image_meta_data import ImageMetaData

class Model(QObject):
    imagesDirectorySignal = pyqtSignal(str)
    imagesReadySignal = pyqtSignal(list)
    probPlotSignal = pyqtSignal(str)
    resetSignal = pyqtSignal()

    @property
    def imagesDirectory(self):
        return self._imagesDirectory

    @imagesDirectory.setter
    def imagesDirectory(self, value):
        self._imagesDirectory = value

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, value):
        self._images = value

    def __init__(self):
        super().__init__()
        self._images = []
        self._imagesDirectory = ''
        self._classifierName = 'example.h5'
        self._probPlotName = '.probPlot.png'

    def evaluateImages(self):
        testX = np.zeros(shape=(len(self._images),512, 512, 1), dtype = "float16")
        for index in range (0, len(self.images)):
            testX[index] = pydicom.read_file(os.path.join(self._imagesDirectory, self.images[index].name)).pixel_array[0].reshape(512,512,1)

        testX /= 2048
        model = load_model(os.path.join('resources', 'models', self._classifierName))
        Y = model.predict(x=testX, batch_size=32, verbose=1)

        for index, prediction in enumerate(Y, start=0):

            if prediction[0] >= 0.5:
                self.images[index].diagnosis = "No Fungus"
                self.images[index].probability = prediction[0]
            else:
                self.images[index].diagnosis = "Fungus"
                self.images[index].probability = prediction[1]

        accuracy = np.sum(Y[:,1])/len(self.images)
        self.imagesReadySignal.emit(self.images)

        self._generateProbabilityPlot([i for i in range(testX.shape[0])], Y[:,1])

    def reset(self):
        self._images = []
        self._imagesDirectory = ''
        self.resetSignal.emit()

    def _generateProbabilityPlot(self, X, Y):
        plt.figure(figsize=(8,2))
        plt.xlim([0, len(X)])
        plt.ylim([0, 1])
        plt.plot(X, Y)
        plt.ylabel('Fungus probability')
        plt.savefig(self._probPlotName)
        self.probPlotSignal.emit(self._probPlotName)
