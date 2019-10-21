import os
from PyQt5.QtCore import QObject, pyqtSlot, QDir, QCoreApplication
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog)

class MainController(QObject):
    def __init__(self, model):
        super().__init__()
        self._model = model

    @pyqtSlot(int)
    def change_amount(self, value):
        self._model.amount = value

        # calculate even or odd
        self._model.even_odd = 'odd' if value % 2 else 'even'

        # calculate button enabled state
        self._model.enable_reset = True if value else False

    def loadDirectory(self):
        dir = str(QFileDialog.getExistingDirectory(None, "Select a directory containing images"))
        self._model.imagesDirectory = dir
        newFiles = []
        for dirpath, subdirs, files in os.walk(dir):
            dirPathRelative = dirpath.replace(dir, "")
            dirPathRelative = dirPathRelative.strip(os.sep)
            for file in files:
                if file.lower().endswith(".dcm"):
                    newFiles.append(os.path.join(dirPathRelative, file))
        self._model.imagesPaths = newFiles

    def evaluateImages(self):
        self._model.evaluateImages()

    def exitApplication(self):
        QCoreApplication.instance().quit()
