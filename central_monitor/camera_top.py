from typing import List

from central_monitor.controller import Controller
from central_monitor.viewer import Viewer


class Camera():
    """
    camera interface top ONLY invoked by subprocess for frame display or transmiting ctrl signal

    viewer belongs to the frame part while controller belongs to ctrl part
    """

    def __init__(
            self,
            login_config: List[dict],
            id: int,
        ) -> None:

        self.id = id
        self.viewer = Viewer(
            login_config=login_config,
            id=id,
        )
        self.controller = Controller(
            login_config=login_config[0],
        )
        