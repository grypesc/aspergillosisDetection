import os
from PyQt5.QtCore import QObject, pyqtSlot, QDir, QCoreApplication
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QMessageBox)
from model.image_meta_data import ImageMetaData

class MainController(QObject):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def loadDirectory(self):
        dir = str(QFileDialog.getExistingDirectory(None, "Select a directory containing images"))
        self._model.imagesDirectory = dir
        newFiles = []
        for dirpath, subdirs, files in os.walk(dir):
            dirPathRelative = dirpath.replace(dir, "")
            dirPathRelative = dirPathRelative.strip(os.sep)
            for file in files:
                if file.lower().endswith(".dcm"):
                    meta = ImageMetaData(os.path.join(dirPathRelative, file), "", "" )
                    newFiles.append(meta)
        self._model.images = newFiles
        if (len(self._model.images) <= 0):
            self.displayMessageBox(QMessageBox.Warning, "Warning", "No dicom images found in that directory.")
            return
        self._model.evaluateImages()

    def displayMessageBox(self, icon, title, text):
       msgBox = QMessageBox()
       msgBox.setIcon(icon)
       msgBox.setWindowTitle(title)
       msgBox.setText(text)
       msgBox.exec()


    def exitApplication(self):
        QCoreApplication.instance().quit()
