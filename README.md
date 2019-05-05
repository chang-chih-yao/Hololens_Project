# Hololens_Project

- 開發時間：2018/09/20 ~ now
- 開發平台：Win 10, GTX 1060, i7-7700k
- 開發裝置：Microsoft Hololens 1
- 開發工具：Python 3.6, Unity 2018.3.0f
- 技術：Tensorflow 1.9.0 (pose-estimation, human-detection), PyTorch 0.4.1 (action-recognition)

***

## FlowChart

玩家穿戴AR眼鏡拍攝對手的肢體動作，透過wifi將影像傳送至運算伺服器，根據預訓練之深度類神經網路模型估測出對手的2D人體姿態與動作類別，根據目前做的動作類別，在定義好的骨架節點觸發相對應的特效，將觸發節點之位置與對應特效的資訊傳回AR眼鏡端，由眼鏡裝置繪製特效，產生虛實對應的AR效果。例如 "火影忍者" 當中許多 "忍術" 可以透過此方法把特效添加在真實人物上。

由於Server1接收到的frame是透過Hololens回傳的，所以當配戴Hololens的人，沒有看著對方的時候，Server1接收到的frame會是沒有人的影像。
此時就必須依靠外部的4台camera幫忙偵測對方(敵人)的動作是什麼，並把資訊回傳給Server1。

![image](https://github.com/chang-chih-yao/Hololens_Project/blob/master/Flow_Chart2.JPG)

![image](https://github.com/chang-chih-yao/Hololens_Project/blob/master/Flow_Chart.png)

***

## Server1 & Hololens

![image](https://github.com/chang-chih-yao/Hololens_Project/blob/master/1.png)

例如螺旋丸：  
![image](https://github.com/chang-chih-yao/Hololens_Project/blob/master/4.gif)

在Hololens眼鏡裡面就會看到這樣：  
![image](https://github.com/chang-chih-yao/Hololens_Project/blob/master/3.JPG)

***

## Our Dataset



***

## Installation for Hololens Toolkit and Unity

[Installation for Hololens Toolkit and Unity](https://github.com/Microsoft/MixedRealityToolkit-Unity/blob/2017.4.3.0/GettingStarted.md)

Version：
- Win 10 SDK 10.0.17134.0
- Visual Studio 2017
- .NET 4.6

***

## Install Opencv with Hololens for Unity

[HoloLensWithOpenCVForUnityExample](https://github.com/EnoxSoftware/HoloLensWithOpenCVForUnityExample)

- Windows 10 Pro 1809
- Windows 10 SDK 10.0.17134.0
- Visual Studio 2017
- Unity 2018.3.0f
- [HoloToolkit-Unity](https://github.com/Microsoft/MixedRealityToolkit-Unity/releases) 2017.4.3.0
- [OpenCV for Unity](https://assetstore.unity.com/packages/tools/integration/opencv-for-unity-21088?aid=1011l4ehR&utm_source=aff) 2.3.3
- [HoloLensCameraStream](https://github.com/VulcanTechnologies/HoloLensCameraStream)

在Client裡面的Assets資料夾內，若上述全部都裝完應該會長這樣：  
![image](https://github.com/chang-chih-yao/Hololens_Project/blob/master/2.JPG)

***

## Python package

- python 3.6.7
- opencv 3.4.4
- tensorflow-gpu 1.8.0
- pytorch 0.4.1
- torchvision 0.2.1

## Pose Estimation Network

[OpenPose using Tensorflow](https://github.com/ildoonet/tf-pose-estimation)

***

## Action Recognition Network
