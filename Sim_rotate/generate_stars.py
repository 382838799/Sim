import math
import numpy as np
import time

from scipy.signal import convolve2d
from corr_gaussian_spots import corr_gaussian_spot
from motion import motion_blur_kernel
from read_skip_values import read_skip_values



# ================================================================
# 函数: hx 仿图
# ================================================================
def generate_image_double_log(starImg, imageSize, opts, starmat, starAngle, starLength):
    """生成恒星图像，使用预计算参数和运动模糊核"""
    staryNumber = len(np.ravel(starmat))
    # starCoordinate = np.zeros((staryNumber, 7))   # 先注掉，暂时没用
    starCoordinate = []

    topGray = 65535

    # 计算当前帧中所有星的最小星等和最大星等值
    # ---------------------------------
    magnitudes = np.array([star['magnitude'] for star in starmat])
    opts['minmagnitude'], opts['maxmagnitude'] = min(magnitudes), max(magnitudes)

    # 预计算系数
    # --------
    knodesize = (opts['minnodesize'] - opts['maxnodesize']) / (opts['maxmagnitude'] - opts['minmagnitude'])
    bnodesize = opts['maxnodesize'] - knodesize * opts['minmagnitude'] + 10
    kblurlevel = (opts['minblurlevel'] - opts['maxblurlevel']) / (opts['maxmagnitude'] - opts['minmagnitude'])
    bblurlevel = opts['maxblurlevel'] - kblurlevel * opts['minmagnitude']
    maxlam = topGray * opts['lammaxrate']
    minlam = opts['lamrate'] * maxlam
    klam = (np.log(minlam) - np.log(maxlam)) / (opts['maxmagnitude'] - opts['minmagnitude'])
    blam = np.log(maxlam) - klam * opts['minmagnitude']

    # 因为每一帧中所有星的运动模糊核都是相同的，因此提取出来
    # ---------------------------------------------
    starTWLen = starLength
    starTWDirec = starAngle
    motionkernel = motion_blur_kernel(starTWLen, starTWDirec)


    # 加载需要剔除的恒星
    # -----------------
    # skip_value = []
    skip_values = read_skip_values('/mnt/d/XT/AAA_Simulation/JG02/4/chapter001/Results/skip_values_4s_chapter001.txt')
    tolerance = 1e-4
    # ---------------------------------------------



    k1 = -1
    imgStarNumber = 0


    for j in range(staryNumber):

        mag = starmat[j]['magnitude']

        # 控制整体恒星星等******************************************************************
        if mag > 14.159:
            continue 

        # 过滤粘连恒星
        should_skip = any(abs(mag - val) <= tolerance for val in skip_values)
        if should_skip:
            continue    



        x = round(starmat[j]['x'])
        y = round(starmat[j]['y'])
        if not (0 <= x < imageSize and 0 <= y < imageSize):
            continue

        # 记录偏移量
        xoffset = starmat[j]['x'] - x
        yoffset = starmat[j]['y'] - y

        imgStarNumber = imgStarNumber + 1  # 记录每帧图像中恒星的数目

        mag = starmat[j]['magnitude']
        pointRadius = round(math.sqrt(knodesize * mag + bnodesize)) + 40
        sigma = kblurlevel * mag + bblurlevel
        maxgray = np.exp(klam * mag + blam)

        startX = round(x - pointRadius)
        startY = round(y - pointRadius)
        endX = round(x + pointRadius)
        endY = round(y + pointRadius)

        maxbounder = max(startX, startY, endX, endY)
        minbounder = min(startX, startY, endX, endY)

        if minbounder < 0 or maxbounder > imageSize-1:
            continue


        # 将高斯光斑中心移动到非整数坐标位置
        # ----------------------------
        crtNode = np.zeros((pointRadius * 2 + 1, pointRadius * 2 + 1))
        centerX = pointRadius
        centerY = pointRadius
        crtNode = corr_gaussian_spot(crtNode, (pointRadius * 2 + 1), sigma, (centerX, centerY), (xoffset, yoffset))


        # 给恒星添加运动模糊
        # ---------------
        crtNode = convolve2d(crtNode, motionkernel, mode='same', boundary='symm')


        # 归一化 + 拉伸？
        # ----
        maxcrtNode = np.max(crtNode)
        if maxcrtNode == 0:
            maxcrtNode = 1
        crtNode = crtNode / maxcrtNode
        crtNode = crtNode * maxgray

        # 存储 hx 数据（由于在查星表得到相面坐标时，就已经对大于图像尺寸的坐标进行了过滤，因此不考虑再此处继续存储了，先注掉）
        # -----------
        # k1 += 1
        # starCoordinate[k1, 0] = starmat[j]['x']  # 假设 starmat 是字典的列表
        # starCoordinate[k1, 1] = starmat[j]['y']
        # starCoordinate[k1, 2] = starmat[j]['magnitude']
        # starCoordinate[k1, 3] = sigma
        # starCoordinate[k1, 4] = maxgray
        # starCoordinate[k1, 5] = starTWLen
        # starCoordinate[k1, 6] = starTWDirec
        starCoordinate.append([starmat[j]['x'], starmat[j]['y'], starmat[j]['magnitude'], starmat[j]['ra'], starmat[j]['dec']])
        # ------------------------------------

        # 合并
        starImg[startX:endX+1, startY:endY+1] = starImg[startX:endX+1, startY:endY+1] + crtNode



    return starImg, starCoordinate
