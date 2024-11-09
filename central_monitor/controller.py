from typing import Tuple
from ctypes import create_string_buffer as csb, cdll, CDLL

from central_monitor.HCNetSDK import (
    NET_DVR_PREVIEWINFO,
    NET_DVR_DEVICEINFO_V30,
    byref,
)


class Controller():
    def __init__(self, login_config: dict) -> None:
        self.login_flag = True
        # self._init_dll()
        # self._login(login_config)
        self.HCsdk : CDLL


    def _init_dll(self) -> None:
        """
        initialize necessary library
        
        current lib list:[
            "linux-lib/libhcnetsdk.so",
        ]
        """

        if self.login_flag:
            self.HCsdk = cdll.LoadLibrary("linux-lib/libhcnetsdk.so")
            self.HCsdk.NET_DVR_Init()


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
        where the first elm represents command and the other means start or stop
        """

        if self.login_flag:
            ret = self.HCsdk.NET_DVR_PTZControl(self.lReadPlayHandle, op[0], op[1])
            if ret == 0:
                print(("Start " if op[1] == 0 else "Stop ")+f"ptz control fail, error code is: {self.HCsdk.NET_DVR_GetLastError()}")