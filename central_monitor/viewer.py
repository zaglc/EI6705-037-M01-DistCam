import ctypes
import datetime
import json
import os
import time
from multiprocessing import Queue
from queue import Queue as TQueue
from threading import Lock, Thread, current_thread
from time import gmtime, strftime
from typing import Dict, List

import cv2
import numpy as np


class Viewer:
    def __init__(self, src_type, login_config: list, id: int) -> None:
        # url: rsdp address or local path
        self._url_lst: Dict[str, tuple] = {}
        self.src_type = src_type
        nickname, url = self._get_url(login_config)
        self.src_name = nickname
        self.current_url = url
        self._url_lst.update({self.src_name: (src_type, url)})

        # here if there is more than 1 channels included in config for 1 cam
        # this cam will be recognize as RGB+THERMOMETER
        self.resolution: tuple = None
        self.fps: int = None
        self.fig_size: Dict[str, str] = {}
        self.cbox: Dict[str, str] = {}
        self.obj_id: Dict[str, str] = {}
        self.use_cbox = False
        self.id = id
        self.max_buffer_size = 3
        self._format = {"pic": ".png", "vid": ".mp4"}

    def _set_cam_info(self, cam: cv2.VideoCapture):
        """
        set camera info
        """

        self.fps = int(cam.get(cv2.CAP_PROP_FPS))
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (width, height)
        print(f"camera {self.id} start video source with resolution: {self.resolution}, fps: {self.fps}")

    def _get_url(self, login_config: list):
        """
        get url for specific video source type
        """

        assert "NICKNAME" in login_config, "missing video name"
        vid_name = login_config["NICKNAME"]
        if self.src_type == "local-vid":
            assert "PATH" in login_config, "missing video path"
            url = os.path.join("data", "src", login_config["PATH"])
        elif self.src_type == "hikvision" or self.src_type == "ip-cam":
            assert (
                "NAME" in login_config
                and "PASSWD" in login_config
                and "IP" in login_config
                and "PORT" in login_config
                and "CHANNEL" in login_config
            ), "missing login info"
            name, passwd, ip, port, channel = (
                login_config["NAME"],
                login_config["PASSWD"],
                login_config["IP"],
                login_config["PORT"],
                login_config["CHANNEL"],
            )
            if self.src_type == "hikvision":
                url = f"rtsp://{name}:{passwd}@{ip}/Streaming/Channels/{channel}"
            else:
                url = f"rtsp://{name}:{passwd}@{ip}:{port}/live"

        return vid_name, url

    def start_thread(
        self,
        frame_queue: TQueue,
    ) -> None:
        """
        first initialize thread for reading frame
        then define synchronization primitive for process and thread
        all dynamic variables are defined here,
        which are invisiable to main process(QT)
        """

        self.current_active = 0

        # thread lock for protecting variable in this class
        self._lock = Lock()

        # whether allowing multi-channel streaming for one camera simultaneously or not
        self.simu_stream = False

        # whether need capture image in this frame or not
        self.need_capture = False

        # whether need record video in this frame or not
        self.need_record = False

        # current path for saving images or videos
        self.capture_path = ""

        # whether or not the viewer is recording in current frame
        self.is_recording = False

        # whether or not the subprocess should halt in current frame
        self.is_running = True

        # how many pkg loss has happened before last update in data_panel
        self.package_loss = 0

        # -1 denotes no model
        self.model_list = 0

        self.need_send = False

        # pass frame from frame_main to main
        self.frame_queue = frame_queue
        self.local_cams: Dict[str, cv2.VideoCapture] = {}
        self.local_cams.update({self.src_name: cv2.VideoCapture(self.current_url)})
        self._set_cam_info(self.local_cams[self.src_name])

        # starting thread
        self.threads: Dict[str, Thread] = {}
        self.threads.update({self.src_name: Thread(target=self.real_time_fetch_Main, args=(self.src_name,))})
        self.threads[self.src_name].start()

    def real_time_fetch_Main(self, name: str) -> None:
        """
        main func for camera read thread
        """

        thread_name = current_thread().name
        print(f"{thread_name} launched: Frame decode thread {self.id}-{name}")

        # TODO：清零
        vid_frame_cnt = 0
        last_fetch_time = time.time()
        while True:
            # get current status
            with self._lock:
                need_capture = self.need_capture
                need_record = self.need_record
                need_send = self.need_send
            run_flag = self.is_running
            if not run_flag:
                break
            with self._lock:
                cam = self.local_cams.get(name)

            # for local video data, should not read too fast
            current_time = time.time()
            time.sleep(max(0, 1 / self.fps - current_time + last_fetch_time))
            last_fetch_time = current_time

            # read frame
            ret = cam.grab()
            if not ret:
                # print("second")
                cam = cv2.VideoCapture(self.current_url)
                self.local_cams[name] = cam
                cam.grab()

            ret, frame = cam.retrieve()
            if not ret:
                print(f"Camera {name} is still not working after reconnect")
                raise RuntimeError()

            # handle condition when capture images
            # TODO：如果要存带框的图片，需要新加一个TQueue
            if need_capture or need_record:
                img_save = frame.copy()

                if need_capture or not self.is_recording:
                    self.set_saving_prefer()
                r = (self.fig_size[name][0] / img_save.shape[1], self.fig_size[name][1] / img_save.shape[0])
                final_size = self.fig_size[name]
                if img_save.shape[:2] != tuple(reversed(self.fig_size[name])):
                    img_save = cv2.resize(img_save, self.fig_size[name])
                if self.use_cbox:
                    c = [int(self.cbox[name][i] * r[i % 2]) for i in range(4)]
                    img_save = img_save[c[1] : c[1] + c[3], c[0] : c[0] + c[2], :]
                    final_size = (c[2], c[3])

            if need_capture:
                pic_pth = self._get_path("pic", name)
                cv2.imwrite(pic_pth, img_save)
                print(
                    f"""picture saving info:
                      \tcamera code  :\t{name}
                      \tsave path    :\t{pic_pth.replace(self.capture_path,"")[1:]}
                      \tresolution   :\t{final_size}\n""",
                    end="",
                )
                self.flip_inter_val("need_capture")

            # handle condition when recording
            elif need_record:
                # first frame for recording: initialize
                if not self.is_recording:
                    self.is_recording = True
                    vid_pth = self._get_path("vid", name)
                    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
                    self.videoWriter = cv2.VideoWriter(
                        vid_pth,
                        fourcc,
                        self.fps,
                        final_size,
                    )
                    vid_frame_cnt = 0
                self.videoWriter.write(img_save)
                vid_frame_cnt += 1

            # last frame for recording: shut down
            elif self.is_recording:
                self.is_recording = False
                self.videoWriter.release()
                print(
                    f"""video saving info:
                      \tcamera code  :\t{name}
                      \tsave path    :\t{vid_pth.replace(self.capture_path,"")[1:]}
                      \ttotal frame  :\t{vid_frame_cnt}
                      \tresolution   :\t{final_size}
                      \tactual time  :\t{strftime('%H:%M:%S',gmtime(vid_frame_cnt/self.fps))}\n""",
                    end="",
                )

            # report pkg loss if the buffer is full and write image
            if name == self.src_name:
                if self.frame_queue.qsize() >= self.max_buffer_size:
                    _ = self.frame_queue.get()
                    self.package_loss = 1
                else:
                    self.package_loss = 0

                self.frame_queue.put((frame if need_send else np.zeros((1, 1)), self.package_loss))

        print(f"{thread_name} normally quit: Frame decode thread {self.id}-{name}")

    def set_saving_prefer(self, src_name=None):
        """
        set default saving preference when start the platform
        """

        if src_name is None:
            src_name = self.src_name
        if os.path.exists("data/temp/box_config.json"):
            with open("data/temp/box_config.json", "r") as f:
                s_p = json.load(f)
                cur_b = s_p[src_name]
                self.fig_size[src_name] = (cur_b["res_w"], cur_b["res_h"])
                self.cbox[src_name] = (cur_b["cp_x"], cur_b["cp_y"], cur_b["cp_w"], cur_b["cp_h"])
                self.use_cbox = s_p["apply_cbox"]
                self.obj_id[src_name] = cur_b["class_id"]
                self.capture_path = s_p["select_path"]

    def switch_vid_src(self, src_type: str, login_config: dict) -> None:
        """
        switch channel invoked by outer subprocess for frame fetch
        """

        with self._lock:
            simu_stream = self.simu_stream

        # remove all cameras and threads if not simultaneous stream
        if not simu_stream:
            self.is_running = False
            for name, cam in self.local_cams.items():
                self.threads[name].join()
                cam.release()
            self.local_cams.clear()
            self.threads.clear()
            self._url_lst.clear()
            self.is_running = True

        name, url = self._get_url(login_config)
        if name not in self._url_lst:
            self._url_lst.update({name: (src_type, url)})
            self.local_cams.update({name: cv2.VideoCapture(url)})
            self.threads.update({name: Thread(target=self.real_time_fetch_Main, args=(name,))})
            self.threads[name].start()

        self.src_name = name
        self.current_url = url
        self.src_type = src_type

    def flip_inter_val(self, attr: str) -> None:
        """
        set value externally, only for bool vars
        """

        with self._lock:
            old_val = getattr(self, attr)
            setattr(self, attr, not old_val)

    def _get_path(self, type: str, src_name: str):
        """
        get save path for pic and vid
        """

        folder = os.path.join(self.capture_path, os.path.join(str(self.id) + "_" + src_name, type))
        if not os.path.exists(folder):
            os.makedirs(folder)

        sfx = self._format[type]
        pfx = f"{self.id}_{self.obj_id[src_name]}_"
        files = [f.startswith(pfx) for f in os.listdir(f"{folder}")]
        num = "0" * (4 - len(str(sum(files) + 1))) + str(sum(files) + 1)
        cur = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        pth = os.path.join(folder, pfx + num + "_" + cur + sfx)

        return pth
