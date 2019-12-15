import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from matplotlib.figure import Figure
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
        self._classifierName = 'mobileNetV2Top0.5009_0.7834.h5'
        self._probPlotName = '.probPlot.png'
        self._model = load_model(os.path.join('resources', 'models', self._classifierName))

    def predictImages(self):
        testX = np.zeros(shape=(len(self._images),512, 512, 3), dtype = "float32")
        for index in range (0, len(self.images)):
            img = image.load_img(os.path.join(self._imagesDirectory, self.images[index].name), target_size=(512, 512, 3))
            pixelArray = image.img_to_array(img)
            testX[index] = pixelArray
        testX = preprocess_input(testX)
        predictions = self._model.predict(testX, verbose=1)
        for index, prediction in enumerate(predictions, start=0):
            if prediction[0] >= prediction[1] and prediction[0] >= prediction[2]:
                self.images[index].diagnosis = "No fungi"
                self.images[index].probability = prediction[0]
            elif prediction[1] >= prediction[0] and prediction[1] >= prediction[2]:
                self.images[index].diagnosis = "Fungi"
                self.images[index].probability = prediction[1]
            else:
                self.images[index].diagnosis = "No lungs"
                self.images[index].probability = prediction[2]

        self.imagesReadySignal.emit(self.images)
        self._generateProbabilityPlot([i for i in range(predictions.shape[0])], predictions[:,1])

    def reset(self):
        self._images = []
        self._imagesDirectory = ''
        self.resetSignal.emit()

    def _generateProbabilityPlot(self, X, Y):
        plt.figure(figsize=(8,2))
        plt.xlim([0, len(X)])
        plt.ylim([0, 1])
        plt.plot(X, Y)
        plt.plot(X, [0.5]*len(X))
        plt.ylabel('Fungus probability')
        plt.savefig(self._probPlotName)
        self.probPlotSignal.emit(self._probPlotName)
