
import cv2, os, json
import numpy as np
import threading
import queue, datetime, ctypes
from time import gmtime, strftime
from threading import Lock
from multiprocessing.sharedctypes import SynchronizedArray
from typing import List


class Viewer():
    def __init__(self, login_config: list, id: int) -> None:
        # url: rsdp address 
        self._url = []
        for chan in login_config:
            assert (
                "NAME" in chan and
                "PASSWD" in chan and 
                "IP" in chan and
                "PORT" in chan and 
                "CHANNEL" in chan
            ), "missing login info"

            name, passwd, ip, port, channel = (
                chan["NAME"],
                chan["PASSWD"],
                chan["IP"],
                chan["PORT"],
                chan["CHANNEL"], 
            )
            url = f"rtsp://{name}:{passwd}@{ip}:{port}/live"
            # url = f"rtsp://{name}:{passwd}@{ip}/Streaming/Channels/{channel}"
            self._url.append(url)

        # here if there is more than 1 channels included in config for 1 cam
        # this cam will be recognize as RGB+THERMOMETER
        self.num_chan = len(login_config)
        self.fig_size = [None]*self.num_chan
        self.cbox = [None]*self.num_chan
        self.obj_id = [None]*self.num_chan
        self.use_cbox = None
        self.id = id
        self.fps = 25
        self._format = {"pic": ".png", "vid": ".mp4"}


    def start_thread(
            self, 
            frame_buffer: SynchronizedArray, 
            frame_buffer_out: SynchronizedArray
        ) -> None:
        """
        first initialize thread for reading frame
        then define synchronization primitive for process and thread
        all dynamic variables are defined here, 
        which are invisiable to main process(QT)
        """

        self.current_active = 0
        self.local_cams: List[None|cv2.VideoCapture] = [None for _ in range(self.num_chan)]

        # starting thread
        self.threads = [threading.Thread(
            target=self.real_time_fetch_Main, args=(i, ))
              for i in range(self.num_chan)]
        
        # queue-like data struct sharing between camera read thread(sub) and fetch thread(main)
        self.buffers = [queue.Queue(3) for _ in range(self.num_chan)]
        self.threads[self.current_active].start()


        # shared mem for updating frame between main process(QT) and subprocess for frame fetch
        self.frame_buffer = frame_buffer
        self.frame_buffer_out = frame_buffer_out
        
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

        # whether or not the camera switching is needed in current frame
        self.need_switch = False
        
        # whether or not the viewer is recording in current frame
        self.is_recording = False

        # whether or not the subprocess should halt in current frame
        self.run_flag = True

        # how many pkg loss has happened before last update in data_panel
        self.package_loss = 0

        # -1 denotes no model
        self.model_list = 0

        self.need_read = True


    def real_time_fetch_Main(self, chan_id: int) -> None:
        """
        main func for camera read thread
        """

        # start camera indicated by chan_id
        if self.local_cams[chan_id] is None:
            cam = cv2.VideoCapture(self._url[chan_id])
            print(f"Camera {self.id}-{chan_id} starting with resolution of ({cam.get(3)}, {cam.get(4)})")
            self.local_cams[chan_id] = cam
        self.buffers[chan_id].queue.clear()

        vid_frame_cnt = 0
        while True:
            # get current status
            with self._lock:
                need_switch = self.need_switch
                need_capture = self.need_capture
                need_record = self.need_record
                need_read = self.need_read
                cur_active = self.current_active
                run_flag = self.run_flag and (not (need_switch and not self.simu_stream) or chan_id != self.current_active)

            if not run_flag: break

            # read frame
            cam.grab()
            if not need_read:
                continue
            ret, frame = cam.retrieve()
            img = frame if ret else None

            tp = "RGB" if chan_id == 0 else "THER"
            # handle condition when capture images
            if need_capture or need_record:
                # only RGB channel is current active, we read from buffer
                if chan_id == 0 and cur_active == 0:
                    with self.frame_buffer_out.get_lock():
                        tmp_array = np.frombuffer(
                            self.frame_buffer_out.get_obj(), 
                            dtype=ctypes.c_uint8,
                        )
                        img_save = tmp_array[:frame.nbytes].copy()
                else:
                    img_save = frame.copy()
                img_save = img_save.reshape(frame.shape)

                if need_capture or not self.is_recording:
                    self.set_saving_prefer(chan_id)
                r = (self.fig_size[chan_id][0]/img_save.shape[1], self.fig_size[chan_id][1]/img_save.shape[0])
                final_size = self.fig_size[chan_id]
                if img_save.shape[:2] != tuple(reversed(self.fig_size[chan_id])):
                    img_save = cv2.resize(img_save, self.fig_size[chan_id])
                if self.use_cbox:
                    c = [int(self.cbox[chan_id][i]*r[i%2]) for i in range(4)]
                    img_save = img_save[c[1]:c[1]+c[3], c[0]:c[0]+c[2], :]
                    final_size = (c[2], c[3])

            if need_capture:
                pic_pth = self._get_path("pic", chan_id)
                cv2.imwrite(pic_pth, img_save)
                print(f'''picture saving info:
                      \tcamera code  :\t{self.id}-{tp}
                      \tsave path    :\t{pic_pth.replace(self.capture_path,"")[1:]}
                      \tresolution   :\t{final_size}\n''', end="")
            
            # handle condition when recording
            elif need_record:
                # first frame for recording: initialize
                if not self.is_recording:
                    self.is_recording = True
                    vid_pth = self._get_path("vid", chan_id)
                    fourcc = cv2.VideoWriter.fourcc('m','p','4','v')
                    self.videoWriter = cv2.VideoWriter(
                        vid_pth,
                        fourcc,
                        self.fps,
                        final_size,
                    )

                self.videoWriter.write(img_save)
                vid_frame_cnt += 1

            # last frame for recording: shut down
            elif self.is_recording:
                self.is_recording = False
                self.videoWriter.release()
                print(f'''video saving info:
                      \tcamera code  :\t{self.id}-{tp}
                      \tsave path    :\t{vid_pth.replace(self.capture_path,"")[1:]}
                      \ttotal frame  :\t{vid_frame_cnt}
                      \tresolution   :\t{final_size}
                      \tactual time  :\t{strftime('%H:%M:%S',gmtime(vid_frame_cnt/self.fps))}\n''', end="")
            
            # report pkg loss if the buffer is full
            if self.buffers[chan_id].full():
                self.buffers[chan_id].queue.clear()
                # TODO: 会有争用，但问题不大
                if cur_active == chan_id:
                    self.package_loss += 1
            
            # save image to buffer
            if img is not None:
                self.buffers[chan_id].put(img)
            else:
                raise RuntimeError
                self.buffers[chan_id].put(None)

        self.reset_cam(chan_id)
        
        self.buffers[chan_id].queue.clear()
        print(f"Frame decode thread {self.id}-{chan_id} end")

    
    def fetch_frame(
            self, 
            need_capture: bool,
            need_record: bool,
        ) -> tuple | None:
        """
        func invoked by subprocess for frame fetch 
        """

        # save current state
        with self._lock:
            self.need_capture = need_capture
            self.need_record = need_record
            chan_id = self.current_active

        frame: np.ndarray = self.buffers[chan_id].get()

        # update frame_buffer
        with self.frame_buffer.get_lock():
            tmp_array = np.frombuffer(
                self.frame_buffer.get_obj(), 
                dtype=ctypes.c_uint8,
            )
            tmp_array[:frame.nbytes] = frame.reshape(-1)[:]

        return frame.shape


    def set_saving_prefer(self, chan_id):
        """
        set default saving preference when start the platform        
        """

        if os.path.exists("data/temp/box_config.json"):
            with open("data/temp/box_config.json", "r") as f:
                s_p = json.load(f)
                sfx = f"_{chan_id}" if chan_id else ""
                cur_b = s_p[f"camera{self.id}{sfx}"]
                self.fig_size[chan_id] = (cur_b["res_w"], cur_b["res_h"])
                self.cbox[chan_id] = (cur_b["cp_x"], cur_b["cp_y"], cur_b["cp_w"], cur_b["cp_h"])
                self.use_cbox = s_p["apply_cbox"]
                self.obj_id[chan_id] = cur_b["class_id"]
                self.capture_path = s_p["select_path"]


    def switch_cam(self) -> None:
        """
        switch channel invoked by outer subprocess for frame fetch
        """

        with self._lock:
            self.need_switch = True
            simu_stream = self.simu_stream
            current_active = self.current_active

        if not simu_stream:
            self.threads[current_active].join()
            self.threads[current_active] = threading.Thread(
            target=self.real_time_fetch_Main, args=(current_active, ))
        current_active = (current_active+1)%2
        if not self.threads[current_active].is_alive():
            self.threads[current_active].start()
        
        with self._lock:
            self.need_switch = False
            self.current_active = current_active


    def reset_cam(self, chan_id: int) -> None:
        """
        reset camera indicated by chan_id
        """

        self.local_cams[chan_id].release()
        self.local_cams[chan_id] = None


    def flip_inter_val(self, attr: str) -> None:
        """
        set value externally, only for bool vars
        """

        with self._lock:
            old_val = getattr(self, attr)
            setattr(self, attr, not old_val)


    def _get_path(self, type: str, chan_id: int):
        """
        get save path for pic and vid        
        """

        tp = "RGB" if chan_id == 0 else "THER"
        folder = self.capture_path+f"/{self.id}_{tp}/{type}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        sfx = self._format[type]
        pfx = f"{tp[0]}{self.id}_{self.obj_id[chan_id]}_"
        files = [f.startswith(pfx) for f in os.listdir(f"{folder}")]
        num = "0"*(4-len(str(sum(files)+1)))+str(sum(files)+1)
        cur = datetime.datetime.now().strftime('%m-%d_%H:%M:%S')
        pth = f"{folder}/{pfx}{num}_{cur}{sfx}"

        return pth