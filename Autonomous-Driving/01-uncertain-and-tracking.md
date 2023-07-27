# 不确定性决策

## Gaussian YOLOv3 (待细读)

论文：Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving

Python PyTorch 实现 (基于 coco 数据集的训练代码已经调试好并上传到 github 上)：https://github.com/ZonePG/PyTorch_Gaussian_YOLOv3

**解决的问题**：本文提出了一种提高目标检测精度的方法，YOLOv3+高斯参数+new loss fuction。此外，本文还提出了对 bbox 不确定性进行建模，可以显著减少假阳性FP并增加真阳性TP。

**思路方法**
- 使用单个高斯模型来分别预测tx、ty、tw和th的不确定性。
- 由于输出是作为高斯模型的参数，bbox的损失函数将修改为负对数似然(negative log likelihood, NLL)损失。
- 将objectness、class和Uncertainty结合作为最后的分数。由于将box的不确定性考虑到最终的分数中，因此可以大量降低FP结果。

## Informative Data Selection with Uncertainty for Multi-modal Object Detection （待阅读）

论文：Informative Data Selection with Uncertainty for Multi-modal Object Detection

为了减少噪声数据的影响，设计的模型可以自适应地从多模态数据中选择有效的信息数据。
- It adopts a multi-pipeline loosely coupled architecture to combine the features and results from point clouds and images
- To quantify the correlation in multimodal information, we model the uncertainty, as the inverse of data information, in different modalities and embed it in the bounding box generation.

# 感知跟踪任务调研

## 感知任务基础

### 目标检测

检测不同于直接对图像进行分类，一张图可能有多个检测目标对象，检测需要对图像中的目标进行定位，即需要在图像中标出目标的位置，一般使用矩形框来标记目标的位置。

经典的目标检测模型包括：
- Two-stage模型：Faster RCNN、Cascade RCNN、MaskRCNN。**输入图片，先生成建议区域（Region Proposal），然后送入分类器分类，两个任务由不同的网络完成。**
- One-stage模型：Yolo 系列、SSD、RetinaNet、FCOS、CornerNet。**输入图片，输出 bounding box 和分类标签，由一个网络完成。直接预测目标的位置和类别。**

### 图像分割

图像分割是一种 pixel-level 的任务，需要先识别目标检测框中的目标，然后基于目标像素和目标边缘梯度信息进行分割，并理解对象的类别。

分割过程有两个粒度：
- 语义分割（semantic segmentation）：例如把猫和狗相关的像素分离出来，猫涂上绿色，狗涂上红色。
- 实例分割（instance segmentation）：标识区分每个类别中的所有实例。如果图像中有两条狗，语义分割将所有的狗分别为一个实例（同一种颜色），实例分割区分每条狗（不同颜色）

方法包括：
- 传统方法：基于阈值的方法，基于边缘的方法，基于图的方法，基于聚类的方法，基于区域的方法。
- **深度学习方法**
  - FCN：将 CNN 分类网络的后面几个全连接层都换成卷积，获得张2维的feature map，后接softmax层获得每个像素点的分类信息，从而解决了分割问题
  - U-Net：左边的网络是收缩路径，使用卷积和 maxpooling；右边的网络是扩张路径，使用上采样产生的特征图与左侧收缩路径对应层产生的特征图进行 concatenate 操作。
  - SegNet，RefineNet，PSPNet
  - DeepLab 系列是：结合了深度卷积神经网络和概率图模型（DenseCRFs）的方法
  - Mask-R-CNN

### 目标跟踪

概念参考资料
- https://encord.com/blog/object-tracking-guide/
- https://viso.ai/deep-learning/object-tracking/

目标跟踪涉及跟踪目标对象的运动并预测图像或视频中对象的位置。

目标跟踪与目标检测（如常用的 YOLO 算法）不同：目标检测仅限于单个帧或图像，而目标检测是一种用于通过跟踪对象的轨迹来预测目标对象位置的技术。

## 目标跟踪任务

根据跟踪对象的数量可以分为单目标跟踪(Single object tracking)和多目标跟踪(Multi-object tracking, MOT)；根据是否跨摄像头可分为单镜头跟踪(MOT)和跨镜头跟踪(Multi-Target Multi-Camera Tracking，MTMCT)两种模式。

### MOT

参考资料：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/mot/README.md

优秀论文 list：https://github.com/luanshiyinyang/awesome-multiple-object-tracking

多目标跟踪是定位多个感兴趣的目标，并在连续帧之间维持个体的 ID 信息并记录轨迹。

**主流的做法是 Tracking By Detecting 方式，算法包括两部分：Detection + Embedding**
- Detection部分即针对视频，检测出每一帧中的潜在目标
- Embedding 部分则将检出的目标分配和更新到已有的对应轨迹上(即 ReID 重识别任务)，进行物体间的长时序关联

根据实现的不同（类似目标检测的 one-stage 和 two-stage），又可以划分为 SDE 系列和 JDE 系列算法。
- SDE(Separate Detection and Embedding)：**分离Detection 和 Embedding 两个环节**，这样的设计可以使系统无差别的适配各类检测器，可以针对两个部分分别调优，但由于流程上是**串联**的导致速度慢耗时较长。
  - ByteTrack
  - OC-SORT
  - BoT-SORT
  - DeepSORT
  - CenterTrack
- JDE(Joint Detection and Embedding)：**在一个共享神经网络中同时学习 Detection 和 Embedding**，使用多任务学习的思路设置损失函数。这样的设计兼顾精度和速度，可以实现高精度的实时多目标跟踪。
  - JDE
  - FairMOT
  - MCFairMOT


### MTMCT

参考资料：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/mot/mtmct/README_cn.md

MTMCT 跨镜头多目标跟踪是某一场景下的不同摄像头拍摄的视频进行多目标跟踪，MTMCT 预测的是同一场景下的不同摄像头拍摄的视频。

### AI CITY CHALLENGE

CityFlow benchmark paper 2019：https://arxiv.org/abs/1903.09254

比赛官网：https://www.aicitychallenge.org/

2023 Challenge Track 1: Multi-Camera People Tracking

参与团队将结合真实数据和合成数据，通过多个摄像头跟踪人员。室内环境的性质，加上真实和合成数据、包括重叠视野的摄像机定位等。在跟踪多个摄像机中出现的人物时准确率最高的团队将被宣布为该赛道的获胜者。如果多个团队在该赛道上表现同样出色，则需要最少人工监督的算法将被选为获胜者。

近两年比赛的 Top Teams：
- https://github.com/NVIDIAAICITYCHALLENGE/2023AICITY_Code_From_Top_Teams
- https://github.com/NVIDIAAICITYCHALLENGE/2022AICITY_Code_From_Top_Teams

