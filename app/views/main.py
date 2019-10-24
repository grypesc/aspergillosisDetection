import os
import pydicom
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap

from views.main_ui import Ui_MainWindow


class MainView(QMainWindow):
    def __init__(self, model, main_controller):
        super().__init__()

        self._model = model
        self._main_controller = main_controller
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # connecting widgets to controller
        self._ui.actionLoad_directory.triggered.connect(self._main_controller.loadDirectory)
        self._ui.actionReset.triggered.connect(self._main_controller.resetModel)
        self._ui.actionQuit.triggered.connect(self._main_controller.exitApplication)

        # listeners of model event signals
        self._model.imagesReadySignal.connect(self.onImagesReady)
        self._model.probPlotSignal.connect(self.onProbPlotReady)
        self._model.resetSignal.connect(self.onModelReset)

        # listeners of user created events
        self._ui.tableWidget.itemClicked.connect(self.onItemClicked)

    def onImagesReady(self, value):
        self._ui.tableWidget.setRowCount(len(self._model.images))
        for index, image in enumerate(self._model.images, start=0):
            self._ui.tableWidget.setItem(index, 0, QTableWidgetItem(image.name))
            self._ui.tableWidget.setItem(index, 1, QTableWidgetItem(image.diagnosis))
            self._ui.tableWidget.setItem(index, 2, QTableWidgetItem(str(image.probability)))

    def onItemClicked(self, value):
        ds = pydicom.read_file(os.path.join(self._model.imagesDirectory,self._model.images[value.row()].name))
        img = ds.pixel_array[0] # get image array
        img += 2048
        print(img.dtype)
        image = QImage(img , 512, 512, QImage.Format_Grayscale8)
        self._ui.ctScanLabel.setPixmap(QPixmap.fromImage(image))

    def onProbPlotReady(self, value):
        pixmap = QPixmap(value)
        self._ui.probPlotLabel.setPixmap(pixmap)

    def onModelReset(self):
        self.__init__(self._model, self._main_controller)
