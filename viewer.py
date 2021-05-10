#! /usr/bin/env python3

import sys
from tracker.gui_main import MainWindow
from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)
input_dir = sys.argv[1] if len(sys.argv)>1 else None
gui = MainWindow(input_dir)
gui.show()
sys.exit(app.exec_())
