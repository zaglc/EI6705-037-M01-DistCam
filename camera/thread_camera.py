import cv2
import threading

class CameraCapture(threading.Thread):
    """
    A thread class to capture frames from a video source (camera).
    """

    def __init__(self, source: int):
        """
        Initializes the CameraCapture thread.
        
        :param source: Index of the camera or video file path.
        """
        super().__init__()
        self.source = source
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open video source: {self.source}")
        
        self.frame = None
        self.running = True

    def run(self):
        """
        Continuously captures frames from the video source.
        """
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame
            else:
                self.running = False

    def stop(self):
        """
        Stops the video capture.
        """
        self.running = False
        self.capture.release()

    def get_frame(self):
        """
        Returns the latest captured frame.
        :return: The most recent frame or None if not available.
        """
        return self.frame
    
if __name__ == "__main__":
    camera_indices = [0, 1]
    for index in camera_indices:
        camera = CameraCapture(index)
        camera.start()
    try:
        while True:
            # Get frames from both cameras
            for index in camera_indices:
                frame = camera.get_frame()
                if frame is not None:
                    cv2.imshow(f"Camera {index}", frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        for index in camera_indices:
            camera.stop()
        cv2.destroyAllWindows()
