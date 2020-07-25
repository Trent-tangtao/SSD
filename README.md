# 数据增强(Data Augmentation)

<img src="https://pic2.zhimg.com/v2-29de0f961a54d48fadaf53caed60ec81_1440w.jpg?source=172ae18b" alt="【技术综述】深度学习中的数据增强方法都有哪些？" style="zoom: 33%;" />

[TOC]



### 1.在pipeline的何处进行增强数据

**线下增强（offline augmentation）**：事先执行所有转换，实质上会增加数据集的大小，适用于较小的数据集；

**线上增强（online augmentation）**：在送入机器学习之前，在小批量（mini-batch）上执行这些转换；

### 2.常见的数据增强技术

##### **(1) 几何变换类**

几何变换类即对图像进行几何变换，包括**翻转，旋转，裁剪，变形，缩放**等各类操作；

##### **(2) 颜色变换类**

上面的几何变换类操作，没有改变图像本身的内容，它可能是选择了图像的一部分或者对像素进行了重分布。如果要改变图像本身的内容，就属于颜色变换类的数据增强了，常见的包括**噪声、模糊、颜色变换、擦除、填充**等；

##### **(3) 其他**

GAN生成数据集，mosaic拼接四张图片等；

**2.1 翻转（Flip）**

<img src="https://static.leiphone.com/uploads/new/article/740_740/201805/5aec0802ae0cf.jpeg" alt="数据增强：数据有限时如何使用深度学习 ？ （续）" style="zoom: 33%;" />

可以对图片进行水平和垂直翻转，从左侧开始，原始图片，水平翻转的图片，垂直翻转的图片。

**2.2 旋转（Rotation）**

<img src="https://static.leiphone.com/uploads/new/article/740_740/201805/5aec082705ecb.jpeg" alt="数据增强：数据有限时如何使用深度学习 ？ （续）" style="zoom: 33%;" />

从左向右，图像相对于前一个图像顺时针旋转90度

**2.3 缩放比例（Scale）**

<img src="https://static.leiphone.com/uploads/new/article/740_740/201805/5aec084b36d11.jpeg" alt="数据增强：数据有限时如何使用深度学习 ？ （续）" style="zoom:33%;" />

图像可以向外或向内缩放。向外缩放时，最终图像尺寸将大于原始图像尺寸。大多数图像框架从新图像中剪切出一个部分，其大小等于原始图像。从左到右，原始图像，向外缩放10%，向外缩放20%

**2.4 裁剪（Crop）**

<img src="https://static.leiphone.com/uploads/new/article/740_740/201805/5aec086ca75f0.jpeg" alt="数据增强：数据有限时如何使用深度学习 ？ （续）" style="zoom:33%;" />

与缩放不同，我们只是从原始图像中随机抽样一个部分。然后，我们将此部分的大小调整为原始图像大小。这种方法通常称为随机裁剪（random crop）。从左至右，原始图像，左上角裁剪的图像，右下角裁剪的图像。

**2.5 移位（Translation）**

<img src="https://static.leiphone.com/uploads/new/article/740_740/201805/5aec0898689d7.jpeg" alt="数据增强：数据有限时如何使用深度学习 ？ （续）" style="zoom:33%;" />

移位只涉及沿X或Y方向（或两者）移动图像。从左至右，原始图像，向右移位，向上移位。

**2.6 调整大小（Resize）**

<img src="https://img-blog.csdnimg.cn/20200506160551330.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTkyODc3Mw==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 25%;" />

对图像进行缩放，以保证模型具有尺度不变性

**2.7 高斯噪声（Gaussian Noise）**

<img src="https://static.leiphone.com/uploads/new/article/740_740/201805/5aec08bb45ad8.png?imageMogr2/format/jpg/quality/90" alt="数据增强：数据有限时如何使用深度学习 ？ （续）" style="zoom:33%;" />

神经网络试图学习可能无用的高频特征（大量出现的模式）时，通常会发生过度拟合。具有零均值的高斯噪声基本上在所有频率中具有数据点，从而有效地扭曲高频特征，添加适量的噪音可以增强学习能力。从左至右，原始图形，加入高斯噪声图片，加入盐和胡椒噪声图片

**2.8 RGB 颜色扰动（Color distortion）& 调整亮度，对比度，饱和度，色调（brightness, contrast, saturation, and hue）**

​                                                  <img src="https://picb.zhimg.com/80/v2-00e0a2d2483f15f2bfae6b92080ebed2_1440w.jpg" alt="img" style="zoom:25%;" />             <img src="https://picb.zhimg.com/v2-b62995447a4269e1abee6762bf1883ff_r.jpg" alt="preview" style="zoom: 25%;" />

**2.9 随机擦除（Random erase、CutOut）**

<img src="https://img-blog.csdnimg.cn/20200506171813581.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTkyODc3Mw==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 33%;" />

对图像中随机选取一个矩形区域用特定的值(随机值或者数据均值)进行覆盖

**2.10 Mixup**

<img src="https://img-blog.csdnimg.cn/20200506175306850.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTkyODc3Mw==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 25%;" />

对两张图像和对应标签进行线性叠加

**2.11 CutMix**

<img src="https://img-blog.csdnimg.cn/20200506195809111.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTkyODc3Mw==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 50%;" />

在Mixup和CutOut的基础上，将图像中的某一区域去除，填充成另一图像

**2.12 GAN**

<img src="https://static.leiphone.com/uploads/new/article/740_740/201805/5aec08e4e365d.jpeg" alt="数据增强：数据有限时如何使用深度学习 ？ （续）" style="zoom: 50%;" />

风格转换、生成fake图片

**2.13 Mosaic**

<img src="https://img-blog.csdnimg.cn/20200508160045845.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center" alt="img" style="zoom:50%;" />

mosaic 利用了四张图片拼接，丰富检测物体的背景，而且在BN计算的时候一下子会计算四张图片的数据

**参考资料：**

【1】Data Augmentation | How to use Deep Learning when you have Limited Data

【2】知乎-深度学习中的数据增强方法都有哪些？

【3】CSDN-深度学习数据增强方法总结

### 3.动手实践

基于SSD的数据增强，代码如下：

```python
import torchvision.transforms.functional as FT
```

**1.flip**

```python
def flip(image, boxes):
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes
```

**2.resize**

```python
def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes
```

**3.padding**

```python
def expand(image, boxes, filler):
    """Helps to learn to detect smaller objects.
    """
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0) 

    return new_image, new_boxes
```

**4.Distort brightness, contrast, saturation, and hue**

```python
def photometric_distort(image):
    new_image = image
    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image
```

**5.random crop**

```python
def random_crop(image, boxes, labels, difficulties):
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping
        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)
            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue
            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)
            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),boxes)  
            overlap = overlap.squeeze(0)  
            if overlap.max().item() < min_overlap:
                continue
            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)
            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)
            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (bb_centers[:, 1] < bottom)  
            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue
            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]
            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2]) 
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties
```

##### **6.Mosaic**

```python
    def get_data_with_Mosaic(self, index, image, boxes, labels, difficulties):
        """loads images in a mosaic
        """
        img_size = 300
        image_s4 = Image.new('RGB', (img_size*2, img_size*2), (255, 255, 255))
        boxes_s4 = torch.FloatTensor([])
        labels_s4 = labels
        difficulties_s4 = difficulties
				# 3 additional image indices
        indices = [index] + [random.randint(0, self.num_samples - 1) for _ in range(3)]  
        for i, index in enumerate(indices):
            image = Image.open(self.images[index], mode='r')
            image = image.convert('RGB')
            objects = self.objects[index]
            boxes = torch.FloatTensor(objects['boxes'])
            labels = torch.LongTensor(objects['labels'])
            difficulties = torch.ByteTensor(objects['difficulties'])
            new_image, new_boxes = resize(image, boxes, dims=(img_size, img_size), return_percent_coords=False)

            # place img in img4
            if i == 0:  # top left
                image_s4.paste(new_image, (0, img_size))
                box = new_boxes+torch.FloatTensor([0, img_size, 0, img_size]).unsqueeze(0)
            elif i == 1:  # top right
                labels_s4 = torch.cat([labels_s4,labels])
                difficulties_s4 = torch.cat([difficulties_s4,difficulties])
                image_s4.paste(new_image, (img_size, img_size))
                box = new_boxes + torch.FloatTensor([img_size, img_size, img_size, img_size]).unsqueeze(0)
            elif i == 2:  # bottom left
                labels_s4 = torch.cat([labels_s4, labels])
                difficulties_s4 = torch.cat([difficulties_s4, difficulties])
                image_s4.paste(new_image, (0, 0))
                box = new_boxes
            elif i == 3:  # bottom right
                labels_s4 = torch.cat([labels_s4, labels])
                difficulties_s4 = torch.cat([difficulties_s4, difficulties])
                image_s4.paste(new_image, (img_size, 0))
                box = new_boxes + torch.FloatTensor([img_size, 0, img_size, 0]).unsqueeze(0)

            boxes_s4 = torch.cat([boxes_s4, box], dim=0)
        return image_s4, boxes_s4, labels_s4, difficulties_s4
```

##### **7.简单的验证**

基于SSD300的框架的检验：

（PS：因为实验室服务器远程连接的人数太多了，连不上，所以个人电脑CPU运行的代码，运行时间一个epoch都是四个多小时，测试时间也是四五个小时，所以进行了简单的检验，主要是对Mosaic方法的检验，初始实验一天，对照实验一天，中间还悲惨的跑了一天的测试发现结果不理想，原来是mosaic的数据框标注反了，后面可视化mosaic才看到，可视化的结果也在下面）

epoch：2，batch_size：8，数据集：VOC2007

初始实验：有正常的flip，padding，random crop等，2个epoch之后，结果如下，mAP：0.064，非常低，因为跑的轮数太少了；

<img src="/Users/tang/Library/Application Support/typora-user-images/image-20200724111626818.png" alt="image-20200724111626818" style="zoom:50%;" />         <img src="/Users/tang/Library/Application Support/typora-user-images/image-20200725094234930.png" alt="image-20200725094234930" style="zoom:33%;" />

在前面的基础加上Mosaic方法后，同样2个epoch后，mAP：0.099，有所进步，结果如下，因为SSD300输入是固定的300x300，感觉Mosaic和前面的padding一样，将物体的尺寸缩小了，同时也有利于小物体的检测；

​      <img src="/Users/tang/Library/Application Support/typora-user-images/image-20200725101218361.png" alt="image-20200725101218361" style="zoom: 50%;" />    <img src="/Users/tang/Library/Application Support/typora-user-images/image-20200725094157387.png" alt="image-20200725094157387" style="zoom:33%;" />

Mosaic的可视化：

​     <img src="/Users/tang/Library/Application Support/typora-user-images/image-20200724192214695.png" alt="image-20200724192214695" style="zoom:25%;" />              <img src="/Users/tang/Library/Application Support/typora-user-images/image-20200724192248551.png" alt="image-20200724192248551" style="zoom:25%;" />

**参考资料：**

【1】Github：a-PyTorch-Tutorial-to-Object-Detection

【2】YOLOv4: Optimal Speed and Accuracy of Object Detection

### 4.总结感想

1.平时数据增强都是直接利用pytorch 中 **torchvision.transforms** 几个数据增强函数，小的比赛中基本上都没怎么用数据增强，没有太多的关注这方面。其实也是很有启发性的，比如Mixup的思路和我们TIP在投论文的水下数据的增强网络大致想法是一样的，我们还利用了GAN更好的生成数据，当时我是受了显著性目标检测的启发想到的，不过YOLOv4今年4月份才出来；

2.本科时候主要还是看的比较多，虽然课内的所有大作业和各种比赛的代码核心都是自己，但是因为时间（大部分时间得刷课内成绩和奖学金）和算力卡（实验室资源留给本科生的不多）的限制，很少动手去复现和实践CV方面的任务，大部分都是跑实验之类。所以也是很想读博，希望给自己一个很长的时间去提升自己。

3.现在成型的框架太多了，都是一个团队很久的积淀，也有MMdetection等工具，很疑惑好像自己个人造轮子写出来或者复现的东西有点差劲

