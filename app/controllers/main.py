import os

from PyQt5.QtCore import QObject, pyqtSlot, QDir, QCoreApplication
from PyQt5.QtWidgets import (QFileDialog, QMessageBox)

from model.image_meta_data import ImageMetaData

class MainController(QObject):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def loadDirectory(self):
        dir = str(QFileDialog.getExistingDirectory(None, "Select a directory containing images"))
        newFiles = []
        for dirpath, subdirs, files in os.walk(dir):
            dirPathRelative = dirpath.replace(dir, "")
            dirPathRelative = dirPathRelative.strip(os.sep)
            for file in sorted(files):
                if file.lower().endswith((".jpg", ".jpeg")):
                    meta = ImageMetaData(os.path.join(dirPathRelative, file), "", "")
                    newFiles.append(meta)
        if (len(newFiles) <= 0):
            self.displayMessageBox(QMessageBox.Warning, "Warning", "No jpg images found in that directory.")
            return
        self.resetModel()
        self._model.imagesDirectory = dir
        self._model.images = newFiles
        self._model.predictImages()

    def loadFiles(self):
        files = QFileDialog.getOpenFileUrls(None, "Select .jpg files")
        files = [file.path() for file in files[0]]
        if not files:
            return
        newFiles = []
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                meta = ImageMetaData(file, "", "" )
                newFiles.append(meta)
        self.resetModel()
        self._model.images = newFiles
        self._model.predictImages()

    def resetModel(self):
        self._model.reset()

    def exitApplication(self):
        QCoreApplication.instance().quit()

    def displayMessageBox(self, icon, title, text):
       msgBox = QMessageBox()
       msgBox.setIcon(icon)
       msgBox.setWindowTitle(title)
       msgBox.setText(text)
       msgBox.exec()
