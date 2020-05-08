import numpy as np
import os

from controllers.main import MainController

from tensorflow.keras.preprocessing import image
from mayavi import mlab
from mayavi.tools.helper_functions import volume_slice
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QMessageBox
from tvtk.util.ctf import PiecewiseFunction


class RenderController(QObject):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def volume_render(self, min_depth, max_depth, is_bronchioles_mode, is_no_contrast_mode, is_advanced_mode):
        if min_depth >= max_depth:
            MainController.display_message_box(QMessageBox.Warning, "Error",
                                               "Minimal depth must be less then maximal depth.")
            return

        if self._model._images_directory == '':
            MainController.display_message_box(QMessageBox.Warning, "Error", "Load images first.")
            return

        if is_advanced_mode:
            mlab.options.backend = 'envisage'
        else:
            mlab.options.backend = 'auto'
        mlab.figure('Volume render')

        images = np.zeros(shape=(len(self._model._images), 512, 512), dtype="float32")
        for index in range(0, len(self._model.images)):
            img = image.load_img(os.path.join(self._model._images_directory, self._model.images[index].name),
                                 target_size=(512, 512), color_mode='grayscale')
            images[index] = img

        s = images[:, min_depth:max_depth, :]
        s = s[:, :, ::-1]
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(s))

        otf = PiecewiseFunction()
        if is_bronchioles_mode and not is_no_contrast_mode:
            otf.add_point(0, 0.0)
            otf.add_point(99, 0.0)
            otf.add_point(100, 0.1)
            otf.add_point(190, 0.1)
            otf.add_point(191, 0.0)
            otf.add_point(255, 0)
        elif not is_no_contrast_mode:
            otf.add_point(0, 0.0)
            otf.add_point(139, 0.0)
            otf.add_point(140, 0.1)
            otf.add_point(200, 0.1)
            otf.add_point(255, 1)
        elif is_bronchioles_mode:
            otf.add_point(0, 0.0)
            otf.add_point(1, 0.2)
            otf.add_point(30, 0.2)
            otf.add_point(30, 0.0)
            otf.add_point(255, 0.0)
        else:
            otf.add_point(0, 0.0)
            otf.add_point(1, 0.1)
            otf.add_point(50, 1.0)

        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)

        mlab.show()

    def slice_render(self, axis):
        if self._model._images_directory == '':
            MainController.display_message_box(QMessageBox.Warning, "Error", "Load images first.")
            return
        images = np.zeros(shape=(len(self._model._images), 512, 512), dtype="float32")
        for index in range(0, len(self._model.images)):
            img = image.load_img(os.path.join(self._model._images_directory, self._model.images[index].name),
                                 target_size=(512, 512), color_mode='grayscale')
            images[index] = img

        mlab.figure('Slice render')
        volume_slice(images, plane_orientation= axis + '_axes', colormap='black-white')

        mlab.options.backend = 'auto'
        mlab.show()
