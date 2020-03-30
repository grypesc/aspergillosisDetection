import os

from model.image_meta_data import ImageMetaData

from PyQt5.QtCore import QObject, QCoreApplication
from PyQt5.QtWidgets import (QFileDialog, QMessageBox)


class MainController(QObject):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def load_directory(self):
        dir = str(QFileDialog.getExistingDirectory(None, "Select a directory containing images"))
        images = []
        for dirpath, subdirs, files in os.walk(dir):
            dirPathRelative = dirpath.replace(dir, "")
            dirPathRelative = dirPathRelative.strip(os.sep)
            for file in sorted(files):
                if file.lower().endswith((".jpg", ".jpeg")):
                    meta = ImageMetaData(os.path.join(dirPathRelative, file), "", "")
                    images.append(meta)
        if len(images) <= 0:
            MainController.display_message_box(QMessageBox.Warning, "Warning", "No jpg images found in that directory.")
            return
        self.reset_model()
        self._model.images_directory = dir
        self._model.images = images
        self._model.images_ready_signal.emit(self._model.images)


    def load_files(self):
        files = QFileDialog.getOpenFileUrls(None, "Select .jpg files")
        files = [file.path() for file in files[0]]
        if not files:
            return
        images = []
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                meta = ImageMetaData(file, "", "")
                images.append(meta)
        self.reset_model()
        self._model.images = images
        self._model.predict_images()

    def predict(self):
        if self._model.images_directory == '':
            MainController.display_message_box(QMessageBox.Warning, "Error", "Load images first")
            return
        self._model.predict_images()

    def reset_model(self):
        self._model.reset()

    def exit_app(self):
        QCoreApplication.instance().quit()

    def display_message_box(icon, title, text):
        msgBox = QMessageBox()
        msgBox.setIcon(icon)
        msgBox.setWindowTitle(title)
        msgBox.setText(text)
        msgBox.exec()
