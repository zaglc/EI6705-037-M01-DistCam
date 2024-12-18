# 开发需求说明

目前需要三个功能，分别是：**模型运行时切换**，**动态自适应帧率**，**全屏显示窗口问题修复**
`run.py`中`Frame_main`负责对单个相机相关的命令进行响应，目前实现为每个帧时间内（如33ms）仅响应用户一次命令，具体代码如下，其检测变量`cmd`的值，并触发相应操作
```python
need_switch = cmd == FV_SWITCH_CHANNEL_Q
need_capture = cmd == FV_CAPTURE_IMAGE_Q
need_record = cmd == FV_RECORD_VIDEO_Q
need_refresh = cmd == FV_FLIP_SIMU_STREAM_Q
need_model = cmd == FV_FLIP_MODEL_ENABLE_Q
need_pause = cmd == FV_QTHREAD_PAUSE_Q
need_ctrl = cmd == FV_PTZ_CTRL_Q
```


## 1. 模型运行时切换

### 1.1 功能说明

在运行时，用户可以切换不同的模型，以实现不同的功能，例如目前已经有的目标检测，目标计数和目标追踪。用户可以通过菜单栏`Settings->model inference`选择不同的模型，以及模型运行需要的配置，例如推理的batch size，输入图片的大小img size等

### 1.2 预期功能
+ 点击菜单栏`Settings->model inference`，弹出模型选择窗口
+ 在模型选择窗口中，选择不同的模型，以及模型运行需要的配置，点击`save`可以保存当前设置并切换模型
+ 模型输出结果渲染在视频画面中，如边界框
+ 详细数据可以在实时数据区，即界面下方的表格中展示
+ 也可以存储成txt文件或json文件

### 1.3 需要修改的文件
+ 设计一个弹窗，放在`Qt_ui/childwins`中
+ `Qt_ui/mainwin.py`中添加对应的槽函数和信号，以及触发机制
+ `running_model/extern_model.py`中添加新模型对应的宏，例如现在的`YOLOV3-DETECT`，在`Engine`中添加对模型的特殊处理方式，可以与`YOLOV3-DETECT`适当合并
+ `running_model/extern_model.py`中输入预处理函数`preprocess_img`和后处理函数`process_result`中添加对新模型的处理方式，模型初始化函数`initialize_model`中添加对新模型的初始化方式
+ `run.py`中`Frame_main`添加对模型切换信号的响应，并通知`model_main`进行模型的切换
+ 其他文件待补充

## 2. 动态自适应帧率

### 2.1 功能说明

由于边缘服务器性能受限，无法同时解码多条高清视频流并进行模型推理，需要降低视频帧率或推理分辨率以降低计算量。用户可以根据运行情况手动调整推理帧率，或者系统通过当前掉帧率等指标自动调整推理帧率。调整后，视频窗口更新画面的频率降低，或者模型推理的频率降低

> 注意：如果需要降低视频帧率，请不要降低`Frame_main`识别命令的频率，可以在不需要更新视频窗口的时间戳内，传输一个**1×1**的ndarray给主窗口，这样的array会被自动过滤

### 2.2 预期功能
+ 点击菜单栏`Settings->frame rate`，弹出帧率设置窗口
+ 在帧率设置窗口中，用户可以手动设置推理帧率，或者选择自动调整帧率
+ 如果选择自动调整帧率，则系统会根据当前掉帧率等指标自动调整推理帧率

### 2.3 需要修改的文件
+ 设计一个弹窗，放在`Qt_ui/childwins`中
+ `Qt_ui/mainwin.py`中添加对应的槽函数和信号，以及触发机制
+ `run.py`中`Frame_main`添加对帧率调整信号的响应，在`camera`实例的`viewer`属性中设置解码频率并修改解码函数
+ 如果需要自动调整，在`viewer`中添加一个类，专门负责帧率调整，如`scheduler`
+ 其他文件待补充

## 3. 全屏显示窗口问题修复

### 3.1 功能说明

目前，对于全屏功能实现方式是，点击每个窗口下方的`view`按钮，槽函数**隐藏**其他所有窗口，当前窗口自动填充至原先窗口所占的区域，再次点击则恢复。当窗口数量不足时，如只有一个，未点击`view`时所占的区域就只有一个小窗的范围，点击view无法全屏。

### 3.2 预期功能
+ 点击每个窗口下方的`view`按钮，画面能占满除了右侧和下方工具栏的整个区域，再次点击恢复

### 3.3 需要修改的文件
+ `Qt_ui/mainwin.py`中添加对应的槽函数和信号，以及触发机制
+ `Qt_ui/display.py`和`frame_window.py`文件中对窗口的配置
+ 需要了解QT的对象布局层级，必要时可以对当前的层级进行修改，当前层级为：`Qt_ui/mainwin.py`的`custom_window`--`Qt_ui\view_panel\display.py`的`Ui_MainWindow`--`Qt_ui\view_panel\frame_window.py`的`frame_win`（6个）