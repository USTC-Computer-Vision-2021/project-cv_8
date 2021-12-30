# 基于图像拼接技术实现(A look into past)

## 基本信息：

**组员与分工：**

李一肖（PB18061363）：调研、编写代码、撰写文档

王鹏程（PB18051046）：调研、编写代码、完善文档

**关键步骤与实现技术：**

1. 特征点提取与匹配（SIFT）
2. 图像配准（RANSAC）
3. 图像融合（Poisson Image Editing）

## 背景：

电影《生命因你而动听》中的一首《昨日重现》（*yesterday once more*）惊艳了时光，很多人都怀念过去的日子，过去的风景，过去的人。

往往同一事物的新旧对比总是会让人们感叹沧海桑田的变化，我们想用一张“惊鸿一瞥”（A look into past）的照片来记录这样的变化。

## 原理与实现：

### 原理：

**SIFT：**

最重要的特征点提取与匹配的算法主要有三种：SIFT（尺度不变特征变化）， SURF（加速鲁棒特征），ORF（定向 FAST 等）

这篇论文对于这三种算法的性能进行了对比：

[Image Matching Using SIFT, SURF, BRIEF and ORB: Performance Comparison for Distorted Images](https://arxiv.org/pdf/1710.02726.pdf)

论文中提到：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxvrn863rnj30qo0au416.jpg" alt="Screen Shot 2021-12-30 at 12.50.19 PM" style="zoom:50%;" />

​		由于SIFT的专利已经到期目前可以免费使用，而SURF专利仍然有效，所以我们选择SIFT算法作为我们的关键点检测算法。

**泊松融合：**

Poisson Blending，参见论文：Patrick P´erez 《Poisson Image Editing》，Microsoft Research UK

原理参见：https://blog.csdn.net/hjimce/article/details/45716603

具有强大的图像融合的能力

### 输入/输出

输入：2 张随机形状的图像 + 重叠对象的矩形框

- 一张图片是我们所说的“新”一张，表明它是近期拍摄的。
- 另一个图像是“旧的”，意思是它是在过去拍摄的。

输出： 组合了输入中 2 张图像的新图像

- 我们总是把老图像放在新图像前面，其中对象来自老图像，背景来自新图像。
- 对象需要人工注释，和手动调整基本参数。

下面的两张照片是2020年代的科大老北门和上个世纪70年代恢复高考第一批新生在老北门的照片，在老照片中，四个灯笼高高挂起，我们想复现出这四个灯笼高高挂起的场景。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxvsingyi3j323q0smnef.jpg" alt="Screen Shot 2021-12-30 at 1.22.11 PM" style="zoom:50%;" />

### 第一步：注释重叠的对象

我们假设 2 个图像有一些共同的对象特征，否则就没有这样的“回顾过去”。

因此，我们要求使用者应该首先注释对象，以防有太多对象无法匹配并得到不想要的结果。

![Screen Shot 2021-12-30 at 2.00.35 PM](/Users/wangpengcheng/Library/Application Support/typora-user-images/Screen Shot 2021-12-30 at 2.00.35 PM.png)

架构：元组表示对角线上的左上角和右下角点`--box_[src] 'y_upper-left, x_upper-left, y_bottom-right, x_bottom-right'`。例如

```python
--box_old '158,61,426,539' 
--box_new '85,156,455,700' 
```

### 第二步：提取局部特征

我们使用 SIFT 来提取尺度不变的局部特征。查看有关 SIFT 以及如何在 Python 中实现它的更多信息，请参阅参考

- 实施：[OpenCV中的特征匹配](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
- 理论：[教程 2：图像匹配](https://ai.stanford.edu/~syyeung/cvweb/tutorial2.html)

提取过程：

图像 --> SIFT --> 关键点 (loc_x, loc_y) 和形状特征向量 (num_feature, 128)

```python
sift = cv2.SIFT_create()  # initial SIFT class
kp_old, des_old = sift.detectAndCompute(obj_old, None)
kp_new, des_new = sift.detectAndCompute(obj_new, None)
```

需要注意的是，不同的图像一般具有不同数量的特征。

### 第三步：匹配特征

计算每个特征向量的欧几里德距离并使用 KNN，我们设置 k=2，以找到最“相似”的关键点

这是通过调用自动实现特征匹配的：

```python
 bf  =  cv2.BFMatcher ()
 matches =  bf.knnMatch(feature_vectors_1, feature_vectors_2, k = 2)
```

即使我们有 KNN 来避免极值或关键点，我们仍然添加了一个基于规则的过滤器来控制关键点的质量，我们称之为“good points”![Screen Shot 2021-12-30 at 2.13.55 PM](https://tva1.sinaimg.cn/large/008i3skNly1gxvu04a5qdj30ts0cw40e.jpg)

从图片可以看到，中国科学技术大学的文字以及一教的窗户是关键的匹配的特征点，当然也有一些特征点的匹配是错误的，我们可以用设定的阈值控制它。

### 第四步：对齐新旧图像

我们从 2 张图像中提取了相似的特征。现在我们需要在像素中对齐它们。例如，特征 A 在新图像中位于 (3, 5) 而在旧图像中位于 (5, 6)，因此我们需要学习一种映射旧图像以适应新图像的变换。

基本上，我们使用随机采样一致算法（RANSAC）来估计转换。维基百科中的[RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)。

我们实现的时候，调用`H, status = cv2.findHomography(kp_old, kp_new, cv2.RANSAC, 5.0)`，它将返回一个 3X3 变换矩阵 H。我们用这个仿射矩阵来变换对象图像，使其与背景图像对齐。

这就旧照片的原图：

<img src="/Users/wangpengcheng/Desktop/Screen Shot 2021-12-30 at 2.17.25 PM.png" alt="Screen Shot 2021-12-30 at 2.17.25 PM" style="zoom:50%;" />

这是经过仿射变换对齐之后的图像与原图的对比：可以发现文字的部分已经对齐，但是一教的部分由于当时相机拍摄的视角以及建筑的位置问题，对齐比较困难。

![Screen Shot 2021-12-30 at 2.18.34 PM](https://tva1.sinaimg.cn/large/008i3skNly1gxvu4s115vj321o0p645s.jpg)

### 第五步：平滑边缘并缝合它们

然而，简单的将新旧图像拼接很容易出现接缝太过明显的问题，我们尝试了两种方法：

1. 使用均值滤波模糊接缝处

   ```python
   blurred_img = cv2.blur(img_result_color, (9, 9), 0)
   ```

​	<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxvubrbw1bj30pm0gyjtt.jpg" alt="Screen Shot 2021-12-30 at 2.24.32 PM" style="zoom:50%;" />

​	可以发现效果并不是很好，接缝仍然明显。

2. 使用泊松融合的方法

   ```python
   result = cv2.seamlessClone(img_old.color_trans[20:-70, 60:-70, ::-1].astype('uint8'), img_new.color.astype('uint8'),mask.astype('uint8'), ((c[1]+60+c[3]-70)//2, (c[0]+20+c[2]-70)//2), cv2.MONOCHROME_TRANSFER)
   ```

![result_north](https://tva1.sinaimg.cn/large/008i3skNly1gxvvz1xsk2j30nw0fx434.jpg)

  校名部分的效果比较好，已经成功地挂上了四个灯笼。
## 工程结构

```
.
├── images
├── README.md
├── requirements.txt
├── result
├── run.sh
├── utils.py
└── main.py
```

## 运行说明

python库及版本：

```shell
matplotlib==3.5.0
numpy==1.21.2
opencv_python==4.5.4.60
```

运行：
```shell
pip install -r requirements.txt
python main.py
```


