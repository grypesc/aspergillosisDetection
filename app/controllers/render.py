import numpy as np
import os

from controllers.main import MainController
from model.image_meta_data import ImageMetaData

from keras.preprocessing import image
from mayavi import mlab
from mayavi.tools.helper_functions import volume_slice
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QMessageBox
from tvtk.util.ctf import PiecewiseFunction


class RenderController(QObject):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def volume_render(self, min_depth, max_depth):
        if min_depth >= max_depth:
            MainController.display_message_box(QMessageBox.Warning, "Error",
                                               "Minimal depth must be less then maximal depth.")
            return
        if self._model._images_directory == '':
            MainController.display_message_box(QMessageBox.Warning, "Error", "Load images first.")
            return
        images = np.zeros(shape=(len(self._model._images), 512, 512), dtype="float32")
        for index in reversed(range(0, len(self._model.images))):
            img = image.load_img(os.path.join(self._model._images_directory, self._model.images[index].name),
                                 target_size=(512, 512), color_mode='grayscale')
            images[index] = img

        s = images[:, min_depth:max_depth, :]
        mlab.figure('Volume render')
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(s))

        otf = PiecewiseFunction()
        otf.add_point(0, 0.0)
        otf.add_point(140, 0.0001)
        otf.add_point(200, 0.01)
        otf.add_point(255, 1)
        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)
        mlab.show()

    def slice_render(self):
        if self._model._images_directory == '':
            MainController.display_message_box(QMessageBox.Warning, "Error", "Load images first.")
            return
        images = np.zeros(shape=(len(self._model._images), 512, 512), dtype="float32")
        for index in range(0, len(self._model.images)):
            img = image.load_img(os.path.join(self._model._images_directory, self._model.images[index].name),
                                 target_size=(512, 512), color_mode='grayscale')
            images[index] = img

        mlab.figure('Slice render')
        volume_slice(images, plane_orientation='x_axes', colormap='black-white')

        mlab.show()
