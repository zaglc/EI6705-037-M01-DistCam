# 整体实现思路

项目执行过程中，我们开发了一个可视化摄像头监控与操作平台。该平台实现了对多个海康威视摄像机的图像实时预览与云台控制，如云台旋转；同时，可以监控视频流基本信息，如实时帧率和掉帧率；此外，还集成了目标检测模型，如yolov3，对视频流进行实时推理；最后，还可以根据用户需要，同时进行多设备的抓图或录像，在本项目中主要用于数据采集与实时监控。其中我们使用了六个相机，分别是四台可见光和两台红外+可见光，后者可以通过平台自由切换通道。

该平台的可视化界面基于Python-QT搭建，通过rtsp-ip协议访问相机数据，并利用opencv-python读取视频流，控制方面则采用海康原生python sdk；同时，检测模型使用主流torch框架搭建，运行时平台的逻辑控制与并发同步采用python内置的multiprocessing库实现。具体而言，每个相机的显示界面各占用一个进程，相机控制单独用一个进程，模型推理采用模型冗余的数据并行，并行的大小可自行配置，取决于对资源和延时的要求（默认配置为2，即两个进程，各自负责三个相机的推理）。

该平台使用方便，通过简单的配置文件，写入相机ip等信息，就可直接在控制台运行。它能够满足海康相机40fps的实时性限制，且可扩展性强，易于增加新功能，集成其他模型。

### 1. 整体框架设计-进程与线程
整体上采用多进程控制的思路，便于扩展并满足视频流读取的实时性；由于python对多线程的支持较差，GIL的存在使得仅有IO密集型线程能够做到很好的并发，大多数非IO并行功能都使用进程完成；部分内部读取功能采用线程，减少资源分配同时便于管理共享变量

+ 主进程：负责初始化其他进程以及Qt界面的绝大多数的事件响应
  + Qt子线程0-5 `QThread`：负责绝大多数Qt界面的异步事件相应和实时图像更新
+ 子进程0-5 `frame_main`：负责六台相机的实时显示，与Qt子线程一一对应
  + 子线程：负责管理相机的读、存等IO操作，仅对各自的父进程（0-5）可见
+ 子进程6：负责控制信号的传输，由于同时只控制一个，故只用一个进程
+ 子进程7：负责模型的推理

### 2. 进程与线程的同步+通信
+ 每个子进程0-5与Qt子线程0-5：
  + 条件变量`frame_flag`：用来同步这组通信对，当Qt线程需要更新图片时，才会唤醒`frame_main`
  + 共享内存`frame_val4exec_seq`：用指示性信号（整数）来传递操作指令
    + `frame_main`设置：
    + `QThread`设置：