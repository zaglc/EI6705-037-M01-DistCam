
import cv2, os
import numpy as np
import threading
import queue, time, datetime, ctypes
from threading import Lock, current_thread
from multiprocessing.sharedctypes import SynchronizedArray


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
            # url = f"rtsp://{name}:{passwd}@{ip}/Streaming/Channels/{channel}"
            url = f"rtsp://{name}:{passwd}@{ip}:{port}/live"
            self._url.append(url)

        # here if there is more than 1 channels included in config for 1 cam
        # this cam will be recognize as RGB+THERMOMETER
        self.num_chan = len(login_config)
        self.vid_size = (960, 640)
        self.id = id
        self.fps = 25


    def start_thread(self, frame_buffer: SynchronizedArray) -> None:
        """
        first initialize thread for reading frame
        then define synchronization primitive for process and thread
        all dynamic variables are defined here, 
        which are invisiable to main process(QT)
        """

        # starting thread
        self.thread = threading.Thread(
            target=self.real_time_fetch_Main, args=())
        
        # queue-like data struct sharing between camera read thread(sub) and fetch thread(main)
        self.buffer = queue.Queue(3)
        self.thread.start()

        # shared mem for updating frame between main process(QT) and subprocess for frame fetch
        self.frame_buffer = frame_buffer
        
        # thread lock for protecting variable in this class
        self._lock = Lock()
        
        # whether or not the camera switching is needed in current frame
        self.need_switch = False
        
        # whether or not the viewer is recording in current frame
        self.is_recording = False

        # whether or not the subprocess should halt in current frame
        self.run_flag = True

        # how many pkg loss has happened before last update in data_panel
        self.package_loss = 0


    def real_time_fetch_Main(self) -> None:
        """
        main func for camera read thread
        """

        if not hasattr(self, "cam"):
            self.cam = cv2.VideoCapture(self._url[0])
            print(f"camera {self.id} starting...")
            self.current_active = 0

        while True:
            # get current status
            with self._lock:
                need_switch = self.need_switch
                self.need_switch = False
                run_flag = self.run_flag
            
            # switch cam
            if need_switch:
                self._switch_cam()

            # read frame
            ret, frame = self.cam.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
            
            # report pkg loss if the buffer is full
            if self.buffer.full():
                self.buffer.queue.clear()
                self.package_loss += 1
            
            # save image to buffer
            if img is not None:
                self.buffer.put(img)
            else:
                raise RuntimeError
                self.buffer.put(None)

            if not run_flag: break

        self.reset_cam()

    
    def fetch_frame(
            self, 
            need_capture: bool,
            need_record: bool,
            capture_path: str,
        ) -> tuple | None:
        """
        func invoked by subprocess for frame fetch 
        """

        frame: np.ndarray = self.buffer.get()

        # handle condition when capture images
        if need_capture:
            folder = capture_path+f"/pics_{self.id}"
            if not os.path.exists(folder):
                os.mkdir(folder)
            
            curtime = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            cv2.imwrite(
                f"{folder}/{curtime}.png", 
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
        
        # handle condition when recording
        elif need_record:
            # first frame for recording: initialize
            if not self.is_recording:
                self.is_recording = True
                folder = capture_path+f"/vids_{self.id}"
                if not os.path.exists(folder):
                    os.mkdir(folder)
                
                curtime = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                fourcc = cv2.VideoWriter.fourcc('M','J','P','G')
                self.videoWriter = cv2.VideoWriter(
                    f"{folder}/{curtime}.avi", 
                    fourcc, 
                    self.fps, 
                    self.vid_size,
                )
    
            self.videoWriter.write(
                cv2.cvtColor(cv2.resize(frame, self.vid_size), cv2.COLOR_RGB2BGR))

        # last frame for recording: shut down
        elif self.is_recording:
            self.is_recording = False
            self.videoWriter.release()
        
        # update frame_buffer
        with self.frame_buffer.get_lock():
            tmp_array = np.frombuffer(
                self.frame_buffer.get_obj(), 
                dtype=ctypes.c_uint8,
            )
            tmp_array[:frame.nbytes] = frame.reshape(-1)[:]
    
        return frame.shape


    def switch_cam(self) -> None:
        """
        switch channel invoked by outer subprocess for frame fetch
        """

        with self._lock:
            self.need_switch = True
        self.buffer.queue.clear()


    def _switch_cam(self) -> None:
        """
        switch channel invoked internally, responding to "switch_cam"
        """

        assert self.thread.ident == current_thread().ident, "thread dismatch"
        self.reset_cam()
        self.current_active = (self.current_active+1)%2
        self.cam = cv2.VideoCapture(self._url[self.current_active])


    def reset_cam(self) -> None:
        self.cam.release()
        