from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot
from views.main_ui import Ui_MainWindow


class MainView(QMainWindow):
    def __init__(self, model, main_controller):
        super().__init__()

        self._model = model
        self._main_controller = main_controller
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # connect widgets to controller
        self._ui.actionLoad_directory.triggered.connect(self._main_controller.loadDirectory)
        self._ui.actionQuit.triggered.connect(self._main_controller.exitApplication)
        # listen for model event signals
        self._model.imagesReadySignal.connect(self.onImagesReady)


    def onImagesReady(self, value):
        self._ui.tableWidget.setRowCount(len(self._model.images))
        for index, image in enumerate(self._model.images, start=0):
            self._ui.tableWidget.setItem(index, 0, QTableWidgetItem(image.name))
        images = self._model.images
        for index in range (0, len(self._model.images)):
            self._ui.tableWidget.setItem(index, 1, QTableWidgetItem(images[index].diagnosis))
            self._ui.tableWidget.setItem(index, 2, QTableWidgetItem(str(images[index].probability)))
