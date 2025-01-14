import os
from ctypes import create_string_buffer as csb
from queue import Empty
from queue import Queue as TQueue
from threading import Thread, current_thread
from typing import Callable, Tuple

from src.central_monitor.HCNetSDK import (
    DOWN_LEFT,
    DOWN_RIGHT,
    FOCUS_FAR,
    FOCUS_NEAR,
    IRIS_CLOSE,
    IRIS_OPEN,
    NET_DVR_DEVICEINFO_V30,
    NET_DVR_PREVIEWINFO,
    PAN_LEFT,
    PAN_RIGHT,
    TILT_DOWN,
    TILT_UP,
    UP_LEFT,
    UP_RIGHT,
    ZOOM_IN,
    ZOOM_OUT,
    byref,
    system_get_platform_info,
)
from src.Qt_ui.utils import HIKVISION, LOCAL_VID, IP_CAM

__video_sources__ = [IP_CAM, HIKVISION, LOCAL_VID]


class Controller:
    def __init__(self, src_type: str, login_config: dict) -> None:
        self.normalized_box = [0.5, 0.5, 1.0, 1.0]  # xywh
        self.box_bound = (0.0, 1.0, 0.5)
        self.brightness_factor = 1.0
        self.brightness_bound = (0.5, 1.5)

        self._update_freq = 10
        self._change_rate = (0.025, 0.05)  # xy change half to wh change

        self.login_flag = True
        self.is_running = True
        self.src_type = src_type
        self.HCsdk = None
        # self.thread = Thread(target=self._update_box_thread, daemon=True)
        self.login_config = login_config

    def start_thread(self, local_command_queue: TQueue) -> None:

        self._local_command_queue = local_command_queue
        self.thread = Thread(target=self._update_box_thread, daemon=True)
        self.thread.start()
        self.switch_vid_src(self.src_type, self.login_config)

    def switch_vid_src(self, src_type: str, login_config: dict) -> None:
        """
        switch src_type and login to embedding device if needed
        """

        sys_platform, dll_loader = system_get_platform_info()
        self.src_type = src_type
        self.login_config = login_config
        if src_type == HIKVISION:
            self._init_dll(sys_platform, dll_loader)
            self._login(login_config)
        else:
            if not self.thread.is_alive():
                self.thread.join()
                self.thread = Thread(target=self._update_box_thread, daemon=True)
                self.thread.start()

    def _init_dll(self, sys_platform: str, dll_loader: Callable) -> None:
        """
        initialize necessary library

        current lib list:[
            "linux-lib/libhcnetsdk.so",
        ]
        """

        if self.login_flag:
            if sys_platform == "linux":
                self.HCsdk = dll_loader(os.path.join("libs", "linux", "libhcnetsdk.so"))
                self.HCsdk.NET_DVR_Init()
            elif sys_platform == "windows":
                self.HCsdk = dll_loader(os.path.join("libs", "windows", "HCNetSDK.dll"))
                self.HCsdk.NET_DVR_Init()
            else:
                print("Unsupported platform")

    def _login(self, login_config: dict) -> None:
        """
        login specifially device according to imformation in configs
        """

        if self.login_flag:
            # usr name and passwd etc.
            dev_ip = csb(login_config["IP"].encode())
            dev_port = int(login_config["PORT"])
            dev_user_name = csb(login_config["NAME"].encode())
            dev_password = csb(login_config["PASSWD"].encode())

            device_info = NET_DVR_DEVICEINFO_V30()
            lUserId = self.HCsdk.NET_DVR_Login_V30(dev_ip, dev_port, dev_user_name, dev_password, byref(device_info))

            if lUserId < 0:
                err = self.HCsdk.NET_DVR_GetLastError()
                print(f"Login device fail, error code is: {err}")
                self.HCsdk.NET_DVR_Cleanup()
                return

            # open preview
            preview_info = NET_DVR_PREVIEWINFO()
            preview_info.hPlayWnd = None
            preview_info.lChannel = 1
            preview_info.dwLinkMode = 0
            preview_info.bBlocked = 1

            # TODO: here the HCNetSDK offers REALDATACALLBACK
            # it acts like "hook", which will automatically invoked after the camera has obtained some data packets
            # the transmission deley through these API with C++ backend might be faster then implementing by cv2 with python backend
            self.lReadPlayHandle = self.HCsdk.NET_DVR_RealPlay_V40(lUserId, byref(preview_info), None, None)

    def handle_ctrl(self, op: Tuple[int]) -> None:
        """
        now the func can only execute basic PTZ ctrl commands
        argument "op" must be a tuple of int
        where the first elm represents command and the other means start or stop, 0:start 1:stop
        """

        if self.login_flag:
            if self.src_type == HIKVISION:
                ret = self.HCsdk.NET_DVR_PTZControl(self.lReadPlayHandle, op[0], op[1])
                if ret == 0:
                    print(
                        ("Start " if op[1] == 0 else "Stop ")
                        + f"ptz control fail, error code is: {self.HCsdk.NET_DVR_GetLastError()}"
                    )
            else:
                self._local_command_queue.put(op)

    def _update_box_thread(self) -> None:
        """
        update box info in frame thread
        """

        def adjust(item_ids, directions):
            adjusted = False
            for item_id, direction in zip(item_ids, directions):
                # xy change, directly refuse if out of bound
                if item_id == 0 or item_id == 1:
                    if (
                        direction == 1
                        and round(self.normalized_box[item_id] + self.normalized_box[item_id + 2] / 2, 3)
                        < self.box_bound[1]
                    ) or (
                        direction == -1
                        and round(self.normalized_box[item_id] - self.normalized_box[item_id + 2] / 2, 3)
                        > self.box_bound[0]
                    ):
                        self.normalized_box[item_id] += self._change_rate[0] * direction
                        self.normalized_box[item_id] = round(self.normalized_box[item_id], 3)
                        adjusted = True
                elif item_id == 2 or item_id == 3:
                    if (
                        direction == 1
                        and round(self.normalized_box[item_id - 2] + self.normalized_box[item_id] / 2, 3)
                        < self.box_bound[1]
                        and round(self.normalized_box[item_id - 2] - self.normalized_box[item_id] / 2, 3)
                        > self.box_bound[0]
                    ) or (direction == -1 and self.normalized_box[item_id] > self.box_bound[2]):
                        self.normalized_box[item_id] += self._change_rate[1] * direction
                        self.normalized_box[item_id] = round(self.normalized_box[item_id], 3)
                        adjusted = True
                    elif direction == 1 and self.normalized_box[item_id] < self.box_bound[1]:
                        mod_direct = (
                            1
                            if round(self.normalized_box[item_id - 2] + self.normalized_box[item_id] / 2, 3)
                            < self.box_bound[1]
                            else -1
                        )
                        self.normalized_box[item_id] += self._change_rate[1] * direction
                        self.normalized_box[item_id] = round(self.normalized_box[item_id], 3)
                        self.normalized_box[item_id - 2] += self._change_rate[0] * mod_direct
                        self.normalized_box[item_id - 2] = round(self.normalized_box[item_id - 2], 3)
                        adjusted = True
                else:
                    if (direction == 1 and self.brightness_factor < self.brightness_bound[1]) or (
                        direction == -1 and self.brightness_factor > self.brightness_bound[0]
                    ):
                        self.brightness_factor += self._change_rate[0] * direction
                        self.brightness_factor = round(self.brightness_factor, 3)
                    adjusted = True

            # if adjusted: print(self.normalized_box)

            return adjusted

        command_dicts = {
            TILT_UP: [[1], [-1]],
            TILT_DOWN: [[1], [1]],
            PAN_LEFT: [[0], [-1]],
            PAN_RIGHT: [[0], [1]],
            UP_LEFT: [[0, 1], [-1, -1]],
            UP_RIGHT: [[0, 1], [1, -1]],
            DOWN_LEFT: [[0, 1], [-1, 1]],
            DOWN_RIGHT: [[0, 1], [1, 1]],
            ZOOM_IN: [[2, 3], [-1, -1]],
            ZOOM_OUT: [[2, 3], [1, 1]],
            IRIS_OPEN: [[4], [1]],
            IRIS_CLOSE: [[4], [-1]],
        }
        thread_name = current_thread().name
        print(f"{thread_name} launch")

        while True:
            op = self._local_command_queue.get()
            if op[1] == 1:
                continue
            if not self.is_running:
                break
            while True:
                # when mouse up, the command end
                try:
                    _ = self._local_command_queue.get(timeout=1 / self._update_freq)
                    break
                except Empty:
                    if op[0] != FOCUS_FAR and op[0] != FOCUS_NEAR:
                        adjusted = adjust(*command_dicts[op[0]])
                        if not adjusted:
                            break
                        # print(self.normalized_box)

        print(f"{thread_name} normally quit")
