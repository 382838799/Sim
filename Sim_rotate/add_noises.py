import numpy as np


# ================================================================
# 函数: 添加噪声
# ================================================================
# 旧版本的添加噪声，性能太差，不使用
# ----------------------------
def add_noisev1(mean, img, sigma):
    # 获取图像的尺寸
    h, w = img.shape

    # 初始化噪声矩阵
    noise = np.zeros((h, w))

    # 创建一个噪声图像副本
    noise_img = np.copy(img)

    # 为每个像素添加噪声
    for i in range(h):
        for j in range(w):
            # 生成正态分布噪声
            n = mean + sigma * np.random.randn()
            noise[i, j] = n
            noise_img[i, j] += n

    return noise_img, noise


# 新版的添加噪声函数
def add_noise(mean, img, sigma):
    # 获取图像的尺寸
    h, w = img.shape

    # 生成整个噪声矩阵
    noise = mean + sigma * np.random.randn(h, w)

    # 将噪声加到图像上
    noise_img = img + noise

    return noise_img, noise
