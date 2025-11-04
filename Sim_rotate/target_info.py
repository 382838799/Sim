# ================================================================
# 函数: 加载 mb 的数据
# ================================================================
import random

def get_targets_loc_info(i, data, targetsNum, targetInfo):
    # mag=[8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5,7.2,7.9,7.6]
    # mag=[8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2]

    # mag = [random.uniform(8,8.5) for _ in range(targetsNum)]
    mag = [(7.2 + (0.6 * _ ) / (targetsNum)) for _ in range(targetsNum)]

    # 对.mat文件进行解包
    X = data[0, 0]['X'][0, 0]
    Y = data[0, 0]['Y'][0, 0]

    # 遍历每个目标
    for iter_targets in range(targetsNum):  # Python 中的索引从 0 开始
        x_data = X[i][iter_targets].item()  # 获取 X 数据
        y_data = Y[i][iter_targets].item()  # 获取 Y 数据

        # 填充目标信息
        targetInfo[iter_targets]['xcenter'] = x_data  # 第一个元素
        targetInfo[iter_targets]['ycenter'] = y_data  # 第一个元素
        targetInfo[iter_targets]['beginFrame'] = 1  # 固定为 1
        targetInfo[iter_targets]['mag'] = mag[iter_targets]  # 使用 mag 数组中的相应值

    return targetInfo