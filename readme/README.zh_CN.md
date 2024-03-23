# 声纳外参标定

>en_US [English](../README.md)

## 简介

本项目是论文`Lambertian-TRF: Lambertian Tensorial Radiance Fields for 3D Reconstruction using Underwater Imaging Sonar`（[IEEE Transactions on Instrumentation & Measurement](https://www2.cloud.editorialmanager.com/tim/default2.aspx)期刊审稿中）中实验的声呐外参的标定部分。

标定板上装有四个反射球，运动捕捉系统可以获得这四个球体的姿态。同样，声纳图像可以清楚地区分标定板上的四个反射球。基于声纳图像和运动捕捉系统之间的变换关系，我们求解声纳上的Body坐标系$\mathcal{F}_{B}$与声纳坐标系$\mathcal{F}_{S}$之间的变换矩阵。

## 流程

1. 将标定板放置在安装有动捕系统的水池中，保证标定板能够同时被动捕系统和声纳观察到
2. 基于ROS系统调用ros节点同时搜集包含动捕位姿GroundTruth和声呐图像数据
3. 使用rospy库预处理rosbag数据，将声呐rawImage和位姿信息提取出来并保存在motionCaptureData.pkl文件中，这部分在[preProcessFinal.py](../src/preProcessFinal.py)
4. 基于预处理的图像数据，手动找到对应的标定板上的反光球并将图像上反光球的位置信息保存到文件markerCoordData.pkl中，这部分在[getSonarImageCoord.py](../src/getSonarImageCoord.py)
5. 基于声呐图像中反光球与动捕系统反光球之间的位置变换关系，利用最小二乘优化，数值求解变换矩阵，这部分在[multiImageCalibration.py](../src/multiImageCalibration.py)
6. 根据求出来的变换矩阵，进行重投影验证

![重投影结果示例](../final_biaoding/reProjectionImage/0-19.562288570933188.png)

>搜集的数据保存在[谷歌云盘](https://drive.google.com/file/d/1LjyENhdwCi62JH226Mk2wNh8gRYH6XEl/view?usp=sharing)上