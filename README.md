# EI6705-037-M01-DISTCAM

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Python Version](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## 📖 Introduction

**EI6705-037-M01-DISTCAM** is a distributed camera system developed as a course project for `SJTU EI6705-037-M01`. This system integrates multiple cameras and a central server to provide real-time video processing and AI-powered object detection and classification. Key features include:

- Switching between video sources.
- Simulated PTZ (Pan-Tilt-Zoom) control.
- AI model integration (YOLOv3 and YOLOv11) for object detection and tracking.

---

## 🎥 Functionality Showcase

### 1. Change Video Source
Switch seamlessly between video sources.

![Switch Video Source](doc/figs/function/switch.gif)

### 2. Smooth Canvas Scaling
Experience smooth zooming and scaling.

![Smooth Canvas Scaling](doc/figs/function/scale.gif)

### 3. Simulated PTZ Control
Control pan, tilt, and zoom on selected frames.

![PTZ Control](doc/figs/function/ptz-ctrl.gif)

### 4. YOLOv3 Inference Activation
Run YOLOv3 for real-time object detection.

![YOLOv3 Inference](doc/figs/function/yolov3.gif)

### 5. YOLOv11 Inference and Tracking
Enable advanced object tracking and set restriction areas.

![YOLOv11 Inference](doc/figs/function/yolov11.gif)

---

## 🔧 Setup Instructions

### 1. Clone the Repository
Clone the repository and its submodules (yolov3 module)
```bash
git clone https://github.com/zaglc/EI6705-037-M01-DistCam.git
cd EI6705-037-M01-DistCam
git submodule update --init --recursive
```

Download miniconda script and yolov3 model weights:
```bash
# in Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo bash Miniconda3-latest-Linux-x86_64.sh
wget -P src/running_models/weights https://github.com/zaglc/EI6705-037-M01-DistCam/releases/download/v5.0/yolov3-base.pt
```


### 2. Build Environment
Create a virtual environment and install dependencies:

```bash
conda create -n distcam python=3.10.9
conda activate distcam
pip install -r requirements.txt
pip install torch==1.12.1-cu116 torchvision==0.13.1-cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 3. Configure Video Sources
Edit `configs/video_source_pool.json` to specify your video sources:

```json
"choices": [
    ["local-vid", "JH-WY3-001"],
    ["local-vid", "JH-WY3-001"]
],
"sources": {
    "local-vid": [
        {"NICKNAME": "JH-WY3-001", "PATH": "Jackson-Hole-WY3@06-27_07-05-02.mp4"}
    ],
    "ip-cam": [
        {"NICKNAME": "XiaoMi-14-Pro", "NAME": "admin", "PASSWD": "1234", "IP": "192.168.31.209", "PORT": "8554", "CHANNEL": "1"}
    ]
}
```

### 4. Run the Program
Navigate to the project root and execute the main script:

```bash
python run.py --num_cam 2
```

---

## 📝 Version Iterations

### v1: 
- use `multiprocessing` for communication between Qt-page and camera
- build basic layout, including view panel (left) and control panel (right)
- basic control button: start streaming and exit
- support connecting to IP-CAMERA on mobile through `rtsp`
- add ctrl unit supporting basic PTZ control: 8 directions rotation(can)
  ![local](doc/figs/layout/v1.PNG)  

### v2: 
- stablize input video stream using `threading`, make it really 'real time', now it running with delay
- add `data panel` to show realtime frame rate and drop rate
- add option of taking videos or picture of one or more cameras simultaneously, by using button `select` to select cameras first then using `capture` or `record`
- add PTZ control for `ZOOM_IN/OUT`, `FOCUS_NEAR/FAR`, `IRIS_OPEN/CLOSE` and updating their icons
- add path select unit for file saving through `SEL_PATH`, and the path will be shown in textbrowser above it
- redirect output prompt from terminal to `Output` text browser in Qt_ui page
- add button `view` in order to check specfic camera in full screen mode(in view panel zone), subsequent click will recover to status of multi-camera view
  ![local](doc/figs/layout/v2.PNG) 


### v3: 
- re-construct repo architecture, using `queue` in `multiprocessing` and `threading`, replacing `shareMemory` and supporting more flexible scaleing and maintainability
- add `set_save_prefer`, where users can assign the interest area in the image, and the area will be saved in the video or picture
- add `model_inference` in menu bar, where users can select a model and run inference on the selected camera
- other improvements: scale the layout, show/hide some tool bars.
  ![local](doc/figs/layout/v3.PNG) 

### v4:
- add a new tool bar that showing realtime-detection result of `yolov3` and `yolov11`
- remove redundant `sel_path` button
- support set hyper parameters for `yolov3` and `yolov11` in `model_selection` window
- save all hyper parameters in some neat config files.
- support smooth window scaling when mainwindow size changed
  ![local](doc/figs/layout/v4.PNG) 

### v5:
- use `qdarkstyle` to make the UI more beautiful
- support showing log info in `Output` text browser, while saving as `.log` file
- add `inference cost` stats in realtime data panel
- fix bugs for model selection function
- support set restriction area by dragging anchers in `model_selection` window
  ![local](doc/figs/layout/v5.PNG) 
---

## 🛠️ Technologies Used

- **Languages:** Python
- **Frameworks:** PyTorch, Qt
- **Models:** YOLOv3, YOLOv11

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👥 Contributors

Thanks to all the contributors who made this project possible:

<a href="https://github.com/zaglc/EI6705-037-M01-DistCam/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zaglc/EI6705-037-M01-DistCam" />
</a>

---
