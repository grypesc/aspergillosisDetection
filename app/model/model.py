import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import sys
np.set_printoptions(threshold=sys.maxsize)

from keras.models import load_model
from PyQt5.QtCore import QObject, pyqtSignal
from .image_meta_data import ImageMetaData

class Model(QObject):
    amount_changed = pyqtSignal(int)
    even_odd_changed = pyqtSignal(str)
    enable_reset_changed = pyqtSignal(bool)
    imagesDirectorySignal = pyqtSignal(str)
    imagesPathsSignal = pyqtSignal(list)
    imagesPredictionSignal = pyqtSignal(dict)

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, value):
        self._amount = value
        self.amount_changed.emit(value)

    @property
    def even_odd(self):
        return self._even_odd

    @even_odd.setter
    def even_odd(self, value):
        self._even_odd = value
        self.even_odd_changed.emit(value)

    @property
    def enable_reset(self):
        return self._enable_reset

    @enable_reset.setter
    def enable_reset(self, value):
        self._enable_reset = value
        self.enable_reset_changed.emit(value)

    @property
    def imagesDirectory(self):
        return self._imagesDirectory

    @imagesDirectory.setter
    def imagesDirectory(self, value):
        self._imagesDirectory = value
        self.imagesDirectorySignal.emit(value)

    @property
    def imagesPaths(self):
        return self._imagesPaths

    @imagesPaths.setter
    def imagesPaths(self, value):
        self._imagesPaths = value
        self.imagesPathsSignal.emit(value)

    @property
    def imagesPrediction(self):
        return self._imagesPrediction

    @imagesPrediction.setter
    def imagesPrediction(self, value):
        self._imagesPrediction = value
        self.imagesPredictionSignal.emit(value)

    def __init__(self):
        super().__init__()
        self._amount = 0
        self._even_odd = ''
        self._enable_reset = False
        self.imagesDirectory = ''
        self._imagesPaths = ["a", "b"]
        self._imagesPrediction = {}
        self._classifierName = 'example.h5'
        self._imagesMetaData = []

    def evaluateImages(self):
        testX = np.zeros(shape=(len(self._imagesPaths),512, 512, 1), dtype = "float16")
        index = 0
        for path in self._imagesPaths:
            testX[index] = pydicom.read_file(os.path.join(self._imagesDirectory, path)).pixel_array[0].reshape(512,512,1)
            index+=1

        testX /= 2048
        model = load_model(os.path.join('resources', 'models', self._classifierName))
        yPredictions = model.predict(x=testX, batch_size=32, verbose=1)

        for index, prediction in enumerate(yPredictions, start=0):
            if prediction[0] > 0.5:
                self._imagesPrediction[index] = {"diagnosis": "Fungus", "probability": prediction[0]}
            else:
                self._imagesPrediction[index] = {"diagnosis": "No Fungus", "probability": prediction[1]}

        accuracy = np.sum(yPredictions[:,1])/len(self._imagesPaths)

        self.imagesPredictionSignal.emit(self._imagesPrediction)
