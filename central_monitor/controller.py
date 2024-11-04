from central_monitor.HCNetSDK import *
from typing import Tuple
from ctypes import create_string_buffer as csb, cdll, CDLL

class Controller():
    def __init__(self, login_config) -> None:
        self.login_flag = True
        # self._init_dll()
        # self._login(login_config)
        self.HCsdk : CDLL


    def _init_dll(self):
        if self.login_flag:
            self.HCsdk = cdll.LoadLibrary("linux-lib/libhcnetsdk.so")
        self.HCsdk.NET_DVR_Init()

    def _login(self, login_config):
        if self.login_flag:
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

            self.lReadPlayHandle = self.HCsdk.NET_DVR_RealPlay_V40(lUserId, byref(preview_info), None, None)


    def handle_ctrl(self, op: Tuple[int]):
        if self.login_flag:
            ret = self.HCsdk.NET_DVR_PTZControl(self.lReadPlayHandle, op[0], op[1])
            if ret == 0:
                print(("Start " if op[1] == 0 else "Stop ")+f"ptz control fail, error code is: {self.HCsdk.NET_DVR_GetLastError()}")
            else:
                print(("Start " if op[1] ==0 else "Stop ")+"ptz control success")