from queue import Queue as TQueue
from typing import List

from central_monitor.controller import Controller
from central_monitor.viewer import Viewer


class Camera:
    """
    camera interface top ONLY invoked by subprocess for frame display or transmiting ctrl signal
    viewer belongs to the frame part while controller belongs to ctrl part
    """

    def __init__(
        self,
        vid_type: str,
        login_config: dict,
        id: int,
        num_cam: int,
        ddp_size: int,
    ) -> None:

        self.id = id
        self.num_cam = num_cam
        self.viewer = Viewer(
            src_type=vid_type,
            login_config=login_config,
            id=id,
        )
        # TODO: 同一个摄像头，使用第一个登陆的，后续在其他窗口开了它的其他通道，把控制信号重定向回来
        self.controller = Controller(
            src_type=vid_type,
            login_config=login_config,
        )

        # when num_cam cannot exact divide by ddp_size
        # last chunk pad
        self.ddp_size = ddp_size
        self.ddp_pad = self.num_cam % ddp_size

    def local_id(self):
        return self.id % ((self.num_cam + self.ddp_pad) // self.ddp_size)

    def local_size(self):
        chunk = (self.num_cam + self.ddp_pad) // self.ddp_size
        if self.id // chunk == self.ddp_size - 1:
            return chunk
        else:
            return chunk - self.ddp_pad

    def switch_vid_src(self, src_type: str, login_config: dict):
        """
        invoke when video source change
        """

        self.viewer.switch_vid_src(src_type, login_config)
        self.controller.switch_vid_src(src_type, login_config)

    def start_thread(self, frame_queue: TQueue, local_command_queue: TQueue):
        """
        start thread for frame fetch and ctrl signal handling
        """

        self.viewer.start_thread(frame_queue)
        self.controller.start_thread(local_command_queue)

    @property
    def frame_config(self):
        return (self.controller.normalized_box, self.controller.brightness_factor)

    @property
    def resolution(self):
        return self.viewer.resolution
