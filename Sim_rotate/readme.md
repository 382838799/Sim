## 日志：

### 2025 2.21 发现 bug
在 `def generate_image_double_log(starImg, opts, starmat, starAngle, starLength)`生成恒星的函数中，此处需要修正一个bug：

在`starImg[startY:endY+1, startX:endX+1] = starImg[startY:endY+1, startX:endX+1] + crtNode`这条语句中，matlab的尺寸范围是1-8900而 python 是 0-8899，即不存在8900这个值，此时框会不匹配，考虑该如何修正？

当边框超过8899时，该怎么处理这种情况，mb生成函数中，也存在此问题。

解决思路：判断这种情况，对crtNode进行截断  （后考虑并不需要设计这种逻辑，直接对crtNode框超出大图像边界的星进行舍去）

#### 2025 2.22 已解决该问题

在`generate_image_double_log`以及`generate_target`的代码中针对以下代码片段，将原matlab代码中的
`minbounder < 1 || maxbounder > 8900` 修改为`minbounder < 0 or maxbounder > 8900-1`

```python
maxbounder = max(startX, startY, endX, endY)
minbounder = min(startX, startY, endX, endY)

if minbounder < 0 or maxbounder > 8900-1:
	continue
```





### 2025 2.22
#### 实现新逻辑

- 实现 matlab 中的 `imtranslate(img, translation, 'bicubic');` 双三次插值函数, 且与 matlab 中的逻辑以及结果完全一致, 该代码位于 `translation.py` 文件中
  `def imtranslate(A, translation, method='bicubic', output_view='same', fill_values=0)` ,该函数需要同时用到其上面的两个函数

- 但需要注意的是 双三次插值中的 `translation` 与已经实现的双线性插值不一样，双线性插值中平移是纵轴为 x，横轴为 y [yoffset, xoffset] (或许就应该为[xoffset, yoffset], 到最后转置一下整体即可）

  但双三次插值代码中因为与 matlab 完全一致，因此其平移为[xoffset, yoffset]，xoffset 就是左右移动，yoffset 就是上下移动（其实这个才是反的）

  `starImg[startY:endY+1, startX:endX+1] = starImg[startY:endY+1, startX:endX+1] + crtNode`  与这条语句关系密切与模糊方向也有关

  python 中 x 与 y 的方向与 matlab中的不一致，因此需要再仔细检查一下，按照一种方式来，最后将图像转置即可，切勿两种方向混用在python中，就严格按照 python 的方式来实现（已修改）

- 现有的代码可以先按照双线性插值来继续，等最后再提供一个双三次插值函数的替换（暂未替换）


#### 修改：目前法线与x、y方向相关的暂时只有这三处地方

1. 将 双线性插值 中的 hx 与 mb 中的 `[yoffset, xoffset]` 改为了 `[xoffset, yoffset]`

2. `starImg[startY:endY+1, startX:endX+1] = starImg[startY:endY+1, startX:endX+1] + crtNode `    

  matlab与python中的这个索引都是先行后列

  修改为了`starImg[startX:endX+1, startY:endY+1] = starImg[startX:endX+1,startY:endY+1] + crtNode`

3. 运动模糊函数中，将 p3 = Direc 修改为 p3 = 90 - Direc



#### 发现新问题，在python中查看图片时，使用以下代码会使得原矩阵中的数值发生变化，因此需要另找一种新的查看矩阵（starImg）的方法(暂未解决)

```python
import numpy as np
import cv2


# 将矩阵转换为8位整数型（OpenCV需要）
starImg = starImg.astype(np.uint8)

# 使用OpenCV显示图像
cv2.imshow("Large Matrix", starImg)

# 保存图像为PNG格式
cv2.imwrite("large_matrix_image.png",starImg)

# 等待按键事件并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

print("图像已保存为 large_matrix_image.png")

```



原因：`# 将矩阵转换为8位整数型（OpenCV需要） starImg = starImg.astype(np.uint8)`  对元素进行了取整操作，非常不正确

另外一种勉强查看图像的方法，将值进行归一化限制在0-255（.....总感觉还是有问题，因为取整了）

```python
import numpy as np
import cv2


# 找到矩阵的最小值和最大值
min_val = np.min(starImg)
max_val = np.max(starImg)

# 归一化到0-255范围
normalized_matrix = (starImg - min_val) / (max_val - min_val) * 255

# 将归一化后的矩阵转换为8位整数型（OpenCV需要）
normalized_matrix = normalized_matrix.astype(np.uint8)

# 使用OpenCV显示图像
cv2.imshow("Normalized Matrix", normalized_matrix)

# 保存图像为PNG格式
cv2.imwrite("temp/normalized_matrix_image.png", normalized_matrix)

# 等待按键事件并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

print("图像已保存为 normalized_matrix_image.png")

```

### 2025/2/25

#### 实现新功能

实现了提取距离目标过近的恒星的星等的函数filter_star_magnitude.py 但是有些参数还没有具体设置。