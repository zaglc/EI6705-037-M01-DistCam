import typing
from PyQt6.QtCore import QObject
from central_monitor.controller import Controller
from central_monitor.viewer import Viewer
from PyQt6 import QtCore

import time
import numpy as np

class Camera():
    def __init__(
            self,
            login_config: dict,
            id: int,
        ) -> None:

        self.id = id
        self.viewer = Viewer(
            login_config=login_config,
        )
        self.controller = Controller(
            login_config=login_config,
        )
        
        pass



