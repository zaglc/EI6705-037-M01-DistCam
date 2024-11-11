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
            num_cam: int,
            ddp_size: int,
        ) -> None:

        self.id = id
        self.num_cam = num_cam
        self.viewer = Viewer(
            login_config=login_config,
            id=id,
        )
        self.controller = Controller(
            login_config=login_config[0],
        )

        # when num_cam cannot exact divide by ddp_size
        # last chunk pad 
        self.ddp_size = ddp_size
        self.ddp_pad = self.num_cam % ddp_size

    
    def local_id(self):
        return self.id % ((self.num_cam + self.ddp_pad) // self.ddp_size)


    def local_size(self):
        chunk = (self.num_cam + self.ddp_pad) // self.ddp_size
        if self.id // chunk == self.ddp_size-1:
            return chunk
        else:
            return chunk-self.ddp_pad