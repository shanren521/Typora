### YOLOV5功能增加

+ 模型系列化、分布式训练、FP16模型存储与推理、断电续训、批量预测图像、Tensorboard训练可视化
+ ONNX/TorchScript/CoreML导出、第二阶段分类器(2nd Stage Classifier)、模型集成预测(Model Ensembling)
+ 模型剪枝(Model Pruning)、Conv与BN融合

### YOLOV5训练与预测技巧

+ 预处理
  + 图像归一化、矩形推理缩放
+ 数据增强
  + CutMix、Cutout、Minup
+ 训练优化
  + Warmup、超参数进化、自动计算锚框(AutoAnchor)、模型EMA(Exponential MovingAverage)
  + 学习率调整(余弦退火衰减)、多尺度训练、梯度累积

+ 后处理
  + MergeNMS、TTA(Test Time Augmentation)

## 基础

### 目标检测(物体检测)

+ 类别标签(Category label)、置信度得分(Confidence score)、位置(最小外接矩形，Bounding box)
+ 单类物体检测：分类、分类+定位
+ 多类物体检测：物体检测、图像分割
+ 定位和检测：
  + 定位：找到检测图像中带有一个给定标签的单个目标
  + 检测：找到图像中带有给定标签的所有目标

### 目标检测数据集

+ PASCAL VOC2012
  + 20个分类
+ MS COCO 2014
  + 20万图像，11.5万训练集图像、5千验证集图像、2万测试集图像
  + 80个分类 超过50万个目标标注

### 目标检测性能指标

+ 检测精度
  + Precision(精确度)、Recall(召回率)、F1 Score、IOU(交并比)、P-R curve、AP(平均精确率)、mAP(均值平均精度)
+ 检测速度
  + 前传耗时、每秒帧数、浮点运算量
+ 混淆矩阵
  + ![image-20230529215754000](D:\learn\Typora\pictures\image-20230529215754000.png)

+ IOU
  + ![image-20230529220205400](D:\learn\Typora\pictures\image-20230529220205400.png)

+ AP

  + ![image-20230529220341603](D:\learn\Typora\pictures\image-20230529220341603.png)

  + ![image-20230529220743198](D:\learn\Typora\pictures\image-20230529220743198.png)

+ P-R curve
  + ![image-20230529221506290](D:\learn\Typora\pictures\image-20230529221506290.png)

### YOLOV5 PASCAL VOC训练

+ github YOLOV5 下载源码
+ 测试 python310 detect.py --source ./data/images/ --weights yolov5x6.pt --conf 0.4
  + 小于0.4会丢弃
+ 训练 python310 train.py 

### 目标检测的基本思想

+ ![image-20230529231530487](D:\learn\Typora\pictures\image-20230529231530487.png)

+ FPN多尺度融合
  + ![image-20230529231920150](D:\learn\Typora\pictures\image-20230529231920150.png)

+ 基本思想

  + 13*13的特征图(相当于416\*416图片大小)，stride=32，将输入图像分成13\*13个grid cells
  + 可以跨层预测，bbox在多个预测层都算正样本，数量为3~9个
  + 预测得到的输出特征图有两个维度是提取到的特征的维度，比如13x13，还有一个维度(深度)是Bx(5+C)
  + B表示每个grid cell预测的边界框的数量，C表示边界框的类别数(没有背景类)，5表示4个坐标信息和一个目标性得分。
  + 每个预测框的类别置信度得分计算
    + class confidence score = box confidence score x conditional class probability
    + 它测量分类和定位(目标对象所在的位置)的置信度
    + ![image-20230529233139052](D:\learn\Typora\pictures\image-20230529233139052.png)

  + 后处理：NMS非极大抑制
    + IOU来找到最佳的
  + 损失函数
    + 分类损失
    + 定位损失(预测边界框与GT之间的误差)
    + 置信度损失(框的目标性)
    + 总的损失函数=以上三个损失的和

### YOLOV5网络架构与组件

+ ![image-20230530212256210](D:\learn\Typora\pictures\image-20230530212256210.png)

+ ![image-20230529233615473](D:\learn\Typora\pictures\image-20230529233615473.png)

+ 网络可视化工具：netron
  + python models/export.py --weights yolov5.pt --img 640 --batch 1
+ SPP空间金字塔池化
  + ![image-20230530212040368](D:\learn\Typora\pictures\image-20230530212040368.png)
+ PANet路径聚合网络
  + ![image-20230530212153214](D:\learn\Typora\pictures\image-20230530212153214.png) 

### YOLOV5损失函数

+ 分类损失

  + YOLO使用softmax函数将得分转换为总和为1的概率。如果使用多标签分类，得分总和可以大于1

  + YOLOV5用多个独立的逻辑分类器替换softmax函数，以计算输入属于特定标签的可能性。
  + 在计算分类损失进行训练时，YOLOV5对每个标签使用二元交叉熵损失。可以避免使用softmax函数降低计算复杂度

+ 边界框回归

  + 是许多2D/3D计算机视觉任务中最基本的组件之一
  + 使用IOU计算的度量损失取代  替代回归损失来改进

+ GIOU

  + GIOU = IOU - |C\A∪B|/|C|

+ DIOU

  + 更容易收敛

+ CIOU

  + 

+ 定位损失(预测框与GT框之间的误差)

+ 置信度损失(框的目标性)

+ 总的损失=分类+定位+置信度

+ YOLOV5使用二元交叉熵损失函数计算类别概率和目标置信度得分的损失

+ YOLOV5使用CIOU Loss作为bounding box回归的损失

### YOLOV5目标框回归与跨网格匹配策略

+ ![image-20230530214643071](D:\learn\Typora\pictures\image-20230530214643071.png)

+ 中心点的坐标和宽高都做了归一化处理

+ **目标框的回归**

  + ![image-20230530215237106](D:\learn\Typora\pictures\image-20230530215237106.png)
  + ![image-20230530215304461](D:\learn\Typora\pictures\image-20230530215304461.png)

  + ![image-20230530215517110](D:\learn\Typora\pictures\image-20230530215517110.png)

+ YOLOV5跨网格匹配策略

  + ![image-20230530215723123](D:\learn\Typora\pictures\image-20230530215723123.png)

### YOLOV5训练技巧

+ 训练预热WarmUP

  + ![image-20230530215918421](D:\learn\Typora\pictures\image-20230530215918421.png)

  + 余弦退火调整学习率
  + ![image-20230530220108330](D:\learn\Typora\pictures\image-20230530220108330.png)
  + 自动计算锚框
  + ![image-20230530220317056](D:\learn\Typora\pictures\image-20230530220317056.png)

  + 超参数进化(遗传进化)
  + ![image-20230530220449633](D:\learn\Typora\pictures\image-20230530220449633.png)

  + 自动混合精度训练
  + ![image-20230530220641977](D:\learn\Typora\pictures\image-20230530220641977.png)
    + 如何在PyTorch中使用自动混合精度
    + 使用autocast+GradScaler
    + autocast   使用torch.cuda.amp中的autocast类，只包含网络的前向过程(包括loss的计算), 而不要包含反向传播，BP的op会使用和前向op相同的类型
    + GradScaler 使用torch.cuda.amp.GradScaler 需要在训练开始之前实例化一个GradScaler对象。通过放大loss的值来防止梯度的underflow

+ **支持断点续训**

  + ![image-20230530221349623](D:\learn\Typora\pictures\image-20230530221349623.png)

+ 多GPU**训练**

  + DataParallel支持单机多卡，不支持多机多卡
  + Distributed DataParallel 支持单机多卡，多机多卡
  + ![image-20230530221444586](D:\learn\Typora\pictures\image-20230530221444586.png)

+ **并行加载数据**

  +  ![image-20230530221622615](D:\learn\Typora\pictures\image-20230530221622615.png)

  + ![image-20230530221724536](D:\learn\Typora\pictures\image-20230530221724536.png)

### YOLOV5目录结构

+ ![image-20230530222123618](D:\learn\Typora\pictures\image-20230530222123618.png)

+ ![image-20230530222144337](D:\learn\Typora\pictures\image-20230530222144337.png)

+ ![image-20230530222326430](D:\learn\Typora\pictures\image-20230530222326430.png)

+ ![image-20230530222359078](D:\learn\Typora\pictures\image-20230530222359078.png)

+ ![image-20230530222453526](D:\learn\Typora\pictures\image-20230530222453526.png)

+ ![image-20230530222513763](D:\learn\Typora\pictures\image-20230530222513763.png)

+ ![image-20230530222546929](D:\learn\Typora\pictures\image-20230530222546929.png)

### 激活函数：非线性处理单元

+ ![image-20230530222645405](D:\learn\Typora\pictures\image-20230530222645405.png)

+ ![image-20230530222712977](D:\learn\Typora\pictures\image-20230530222712977.png)

+ ![image-20230530222742711](D:\learn\Typora\pictures\image-20230530222742711.png)

+ ![image-20230530222928790](D:\learn\Typora\pictures\image-20230530222928790.png)

+ ![image-20230530223131799](D:\learn\Typora\pictures\image-20230530223131799.png)

### 模型构建相关代码解析

+ 网络组件代码
  + models/common.py

![image-20230613221821970](D:\learn\Typora\pictures\image-20230613221821970.png)

![image-20230613222313787](D:\learn\Typora\pictures\image-20230613222313787.png)

![image-20230613233152057](D:\learn\Typora\pictures\image-20230613233152057.png)

减少过拟合

![image-20230613233256323](D:\learn\Typora\pictures\image-20230613233256323.png)