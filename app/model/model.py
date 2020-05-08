import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from PyQt5.QtCore import QObject, pyqtSignal
from tensorflow.keras.models import Sequential


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
        self._prob_plot_name = '.probPlot.png'
        self._extracting_model = None
        self._is_lungs_model_name = 'mobileNetV2Top_is_lungs_0.9901_0.0420.h5'
        self._is_lungs_model = None
        self._is_fungus_model_name = 'mobileNetV2Top_is_fungus_0.4035_0.8201_69.h5'
        self._is_fungus_model = None

    def predict_images(self):
        if self._is_lungs_model is None:
            self._is_lungs_model = load_model(os.path.join('resources', 'models', self._is_lungs_model_name))
        if self._is_fungus_model is None:
            self._is_fungus_model = load_model(os.path.join('resources', 'models', self._is_fungus_model_name))

        test_X = np.zeros(shape=(len(self._images), 512, 512, 3), dtype="float32")
        for index in range(0, len(self.images)):
            img = image.load_img(os.path.join(self._images_directory, self.images[index].name),
                                 target_size=(512, 512, 3))
            test_X[index] = image.img_to_array(img)
        test_X = preprocess_input(test_X)

        if self._extracting_model is None:
            self._extracting_model = Sequential()
            self._extracting_model.add(Cropping2D(cropping=((50, 50), (50, 50)), input_shape=(512, 512, 3)))
            self._extracting_model.add(
                MobileNetV2(weights='imagenet', include_top=False, input_shape=(412, 412, 3), pooling='avg'))
        features = self._extracting_model.predict(test_X, verbose=1)

        is_lungs_predictions = self._is_lungs_model.predict(features)
        is_fungus_predictions = self._is_fungus_model.predict(features)
        plot_prob_y = []
        for index, prediction in enumerate(is_lungs_predictions, start=0):
            if prediction[0] <= 0.5:
                self.images[index].diagnosis = "No lungs"
                self.images[index].probability = 1 - prediction[0]
                plot_prob_y.append(0)
            else:
                if is_fungus_predictions[index][0] > 0.5:
                    self.images[index].diagnosis = "Fungus"
                    self.images[index].probability = is_fungus_predictions[index][0]
                else:
                    self.images[index].diagnosis = "No fungus"
                    self.images[index].probability = 1 - is_fungus_predictions[index][0]
                plot_prob_y.append(is_fungus_predictions[index][0])

        self.images_ready_signal.emit(self.images)
        self._generate_probability_plot([i for i in range(len(self._images))], plot_prob_y)

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
