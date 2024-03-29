# 2023/2022 AI City Challenge MTMCT 赛道调研

AICity 由 NVIDIA 公司主办，已成功举办过七届，AICity 旨在利用 AI 技术来提升城市空间的运营效率。AICity 挑战赛主要关注计算机视觉和人工智能交叉领域两个具有巨大潜力的问题 - 智能交通系统 (ITS) 和实体零售业务，Top Teams 论文通常发表在 CVPR Workshop 上面。

## 2023

### 介绍

2023 年第七届挑战有五个赛道，吸引了来自 46 个国家的 508 个团队的创纪录数量的参赛请求，赛道包括：
- **Multi-camera people tracking，多摄像头人员跟踪**
  - 参与团队将结合真实数据和合成数据，合成数据是用 NVIDIA Omniverse 平台创建的，通过多个摄像头跟踪人员。室内环境的性质，加上真实和合成数据、包括重叠视野的摄像机定位等，使得这是与过去几年的车辆多摄像机跟踪截然不同的挑战。在跟踪多个摄像机中出现的人物时准确率最高的团队将被宣布为该赛道的获胜者。如果多个团队在该赛道上表现同样出色，则需要最少人工监督的算法将被选为获胜者。
- Tracked-Vehicle Retrieval by Natural Language Descriptions，通过自然语言描述进行跟踪车辆检索
- Naturalistic Driving Action Recognition，自然驾驶动作识别
- Multi-Class Product Counting & Recognition for Automated Retail Checkout，自动零售结账的多类产品计数和识别
- Detecting Violation of Helmet Rule for Motorcyclists，检测摩托车手佩戴头盔的违规行为

数据集的信息包括：

- 数据集包含 7 个环境 (真实仓库环境和六个合成环境)，共 22 个子集
- 训练、验证、测试划分为 10：5：7
- 数据集总共包含 129 个摄像头、156 个人和 8674590 个边界框
- 所有视频总长度为 1491 分钟，分辨率为 1080p，每秒 30 帧
- 同一子集下的视频都是同步的，同时提供了每个环境的顶部俯视平面图用于校准
- 评估指标：IDF1 score
  - IDP：识别精确度(Identification Precision)是指每个检测框中 ID 识别的精确度，IDP = IDTP / (IDTP + IDFP)，其中 IDTP 表示正确识别的框，IDFP 表示错误识别的框
  - IDR：识别召回率(Identification Recall)是指每个检测框中 ID 识别的召回率，IDR = IDTP / (IDTP + IDFN)，其中 IDFN 表示未识别的框
  - IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

### 结果

大多数团队都遵循 MTMCT 任务的典型工作流程：
- 第一步目标检测采用 YOLO-based 模型
- ReID 重识别模型用于提取鲁棒外观特征
  - 冠军团队 (https://github.com/ipl-uw/AIC23_Track1_UWIPL_ETRI) 使用 OSNet
  - HCMIU 和 Nota 使用联合架构
- 单摄像头跟踪对于构建可靠的轨迹十分重要。大多数团队使用 SORT-based 方法
  - 冠军团队使用 BoT-SORT
- 基于外观和时空信息的聚类
  - 冠军团队和 Nota 使用匈牙利算法聚类

## 2022

### 介绍

2022 年第六届挑战有四个赛道，收到了 27 个国家 254 个团队的参赛请求。赛道包括：
- **City-Scale Multi-Camera Vehicle Tracking，城市规模多摄像头车辆跟踪**
  - 参赛团队将通过单个十字路口和城市中多个十字路口的多个摄像头跟踪车辆。这有助于交通工程师了解整个公路的行程时间。跟踪出现在多个摄像机中的车辆的准确度最高的团队将被宣布为该赛道的获胜者。如果多个团队在该赛道上表现同样出色，则需要最少人工监督的算法将被选为获胜者。
- 赛道二、三、四和 2023 赛道二、三、四一样

数据集信息包括：
- 数据集是 CityFlowV2，包括 880 个不同的车辆和 313931 个边界框，只有经过至少两个摄像头的车辆才被标注释，CityFlow 是第一个城市规模的车辆跟踪数据集
- CityFlowV2 包含 3.58 小时（215.03 分钟）的视频，视频是从跨越 16 个十字路口的 46 个摄像头收集的，两个最远同步相继的距离为 4km
- 数据集涵盖了不同的位置类型，包括十字路口、公路和高速公路
- 数据集分为 6 个场景，训练、验证、测试划分为 3：2：1
- 每个视频的分辨率至少为 960p，大多数视频的帧率为每秒 10 帧
- 评估指标：IDF1 score

### 结果

大多数团队都遵循 MTMCT 任务的典型工作流程：
- 第一步是车辆检测，性能最好的团队使用了 YOLOv5 和 Cascade R-CNN
- 其次，利用 ReID 模型提取鲁棒的外观特征
- 基于检测结果(边界框)和相应的特征嵌入形成单摄像头轨迹
  - Baidu 使用 DeepSort
  - BOE 团队 (https://github.com/coder-wangzhen/AIC22-MCVT) 使用MedianFlow、多级关联和基于区域的合并合并合并来合并增强轨迹预测，以优化轨迹
  - Fraunhofer IOSB 团队通过基于外观的轨迹分割、聚类和轨迹补全进一步增强了单摄像机轨迹
  - SUTPC团队提出了一个遮挡感知模块来连接破碎的轨迹
- MTMCT 跟踪最终的部分是相机间关联，大多数团队构建了具有外观和时空信息的相似度矩阵并应用层次聚类
  - Baidu 使用 k-reciprocal 最近邻进行聚类，约束旅行时间、道路结构和交通规则来减少搜索空间
  - Alibaba 团队引入了一种基于区域内和时间衰减的匹配机制

## Resource
- 比赛官网：https://www.aicitychallenge.org/
- 2023 比赛报告 paper：https://arxiv.org/abs/2304.07500
- 2022 比赛报告 paper：https://arxiv.org/abs/2204.10380
- CityFlow 数据集 paper (2022 赛道一车辆跟踪用的是 CityFlowV2)：https://arxiv.org/abs/1903.09254
- 2023 top teams: https://github.com/NVIDIAAICITYCHALLENGE/2023AICITY_Code_From_Top_Teams
- 2022 top teams: https://github.com/NVIDIAAICITYCHALLENGE/2022AICITY_Code_From_Top_Teams

