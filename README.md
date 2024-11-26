# EI6705-037-M01-DISTCAM

### SETUP
+ create a new virtual environment through `conda`
  ```bash
  conda create -n distcam python=3.10.9
  conda activate distcam
  ```
+ install necessary packages
  ```
  pip install -r requirements.txt
  ```
+ install `pytorch`
  ```
  pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
  ```


### DONE:
#### v1: 
+ use `multiprocessing` for communication between Qt-page and camera

#### v2: 
+ add ctrl unit supporting basic PTZ control: 8 directions rotation(can)


#### v3: 
+ stablize input video stream using `threading`, make it really 'real time', now it running with delay
+ add `data panel` to show realtime frame rate and drop rate
+ add option of taking videos or picture of one or more cameras simultaneously, by using button `select` to select cameras first then using `capture` or `record`
+ add PTZ control for `ZOOM_IN/OUT`, `FOCUS_NEAR/FAR`, `IRIS_OPEN/CLOSE` and updating their icons
+ add path select unit for file saving through `SEL_PATH`, and the path will be shown in textbrowser above it
+ redirect output prompt from terminal to `Output` text browser in Qt_ui page
+ add button `view` in order to check specfic camera in full screen mode(in view panel zone), subsequent click will recover to status of multi-camera view

### UNDER CONSTRUCTION
+ other necessary data in panel
+ other control such as adjust resolution
+ set max frame window size

### PREVIEW 
+ v2: ![local](doc/figs/layout/v2.png) 
+ v3: ![local](doc/figs/layout/v3.png) 
