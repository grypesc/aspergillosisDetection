from PyQt5.QtWidgets import QMainWindow

from views.about_ui import Ui_AboutWindow

class AboutView(QMainWindow):
    def __init__(self):
        super().__init__()
        self._ui = Ui_AboutWindow()
        self._ui.setupUi(self)
        self.listenAndConnect()

    def listenAndConnect(self):
        self._ui.buttonBox.accepted.connect(lambda: self.close())
        self._ui.buttonBox.rejected.connect(lambda: self.close())
