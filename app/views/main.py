import os

from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem
from PyQt5.QtGui import QPixmap

from views.main_ui import Ui_MainWindow
from views.about import AboutView
from views.volume_render import VolumeRenderView


class MainView(QMainWindow):
    def __init__(self, model, main_controller, volume_render_controller):
        super().__init__()
        self._model = model
        self._main_controller = main_controller
        self._render_controller = volume_render_controller
        self._volume_render_view = VolumeRenderView(self._render_controller)
        self._about_view = AboutView()
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.listen_and_connect()

    def listen_and_connect(self):
        # connecting widgets to controller
        self._ui.actionLoad_directory.triggered.connect(self._main_controller.load_directory)
        self._ui.actionLoad_files.triggered.connect(self._main_controller.load_files)
        self._ui.actionReset.triggered.connect(self._main_controller.reset_model)
        self._ui.actionExit.triggered.connect(self._main_controller.exit_app)
        self._ui.actionAbout.triggered.connect(self._about_view.show)
        self._ui.action3D_Model.triggered.connect(self._volume_render_view.show)
        self._ui.actionSlice.triggered.connect(self._render_controller.slice_render)
        # listeners of model event signals
        self._model.images_ready_signal.connect(self.on_images_ready)
        self._model.prob_plot_signal.connect(self.on_prob_plot_ready)
        self._model.reset_signal.connect(self.on_model_reset)
        # listeners of table events
        self._ui.tableWidget.itemClicked.connect(self.on_item_clicked)

    def on_images_ready(self, value):
        self._ui.tableWidget.setRowCount(len(self._model.images))
        for index, image in enumerate(self._model.images, start=0):
            self._ui.tableWidget.setItem(index, 0, QTableWidgetItem(image.name))
            self._ui.tableWidget.setItem(index, 1, QTableWidgetItem(image.diagnosis))
            self._ui.tableWidget.setItem(index, 2, QTableWidgetItem(str(image.probability)))

    def on_item_clicked(self, value):
        pixmap = QPixmap(os.path.join(self._model.images_directory, self._model.images[value.row()].name), )
        self._ui.ctScanLabel.setPixmap(pixmap)

    def on_prob_plot_ready(self, value):
        pixmap = QPixmap(value)
        self._ui.probPlotLabel.setPixmap(pixmap)

    def on_model_reset(self):
        self._ui.setupUi(self)
        self.listen_and_connect()
