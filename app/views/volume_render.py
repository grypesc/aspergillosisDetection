from PyQt5.QtWidgets import QMainWindow

from views.volume_render_ui import Ui_volume_renderWindow


class VolumeRenderView(QMainWindow):
    def __init__(self, render_controller):
        super().__init__()
        self._ui = Ui_volume_renderWindow()
        self._ui.setupUi(self)
        self.render_controller = render_controller
        self.listen_and_connect()

    def listen_and_connect(self):
        self._ui.buttonBox.accepted.connect(lambda: self.render_controller.volume_render(self._ui.spinBox_min.value(), self._ui.spinBox_max.value(), self._ui.checkBoxBronchioles.isChecked(), self._ui.checkBoxAdvanced.isChecked()))
        self._ui.buttonBox.rejected.connect(lambda: self.close())
