import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from PyQt5.QtCore import QObject, pyqtSignal


class Model(QObject):
    images_ready_signal = pyqtSignal(list)
    prob_plot_signal = pyqtSignal(str)
    reset_signal = pyqtSignal()

    @property
    def images_directory(self):
        return self._images_directory

    @images_directory.setter
    def images_directory(self, value):
        self._images_directory = value

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, value):
        self._images = value

    def __init__(self):
        super().__init__()
        self._images = []
        self._images_directory = ''
        self._classifier_name = 'amobileNetV2Top0.3484_0.8699.h5'
        self._prob_plot_name = '.probPlot.png'
        self._model = None

    def predict_images(self):
        if self._model is None:
            self._model = load_model(os.path.join('resources', 'models', self._classifier_name))

        test_X = np.zeros(shape=(len(self._images), 512, 512, 3), dtype="float32")
        for index in range(0, len(self.images)):
            img = image.load_img(os.path.join(self._images_directory, self.images[index].name),
                                 target_size=(512, 512, 3))
            test_X[index] = image.img_to_array(img)
        test_X = preprocess_input(test_X)
        predictions = self._model.predict(test_X, verbose=1)
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

        self.images_ready_signal.emit(self.images)
        self._generate_probability_plot([i for i in range(predictions.shape[0])], predictions[:, 1])

    def reset(self):
        self._images = []
        self._images_directory = ''
        self.reset_signal.emit()

    def _generate_probability_plot(self, X, y):
        plt.figure(figsize=(8, 2))
        plt.xlim([0, len(X)])
        plt.ylim([0, 1])
        plt.plot(X, y)
        plt.plot(X, [0.5] * len(X))
        plt.ylabel('Fungus probability')
        plt.savefig(self._prob_plot_name)
        self.prob_plot_signal.emit(self._prob_plot_name)
