import numpy as np

from scipy.signal import convolve2d
from corr_gaussian_spots import corr_gaussian_spot
from motion import motion_blur_kernel






# ================================================================
# 函数: mb 仿图，因为mb个数较少，对仿图产生的影响很微弱，因此不做修改
# ================================================================
def generate_target(starImg, opts, targetsInfo, frameNum, targetsNum):
    TargetCoordinate = []

    maxMagToGray = np.exp(-opts['minmagnitude'] / 10)
    minMagToGray = np.exp(-opts['maxmagnitude'] / 10)
    knodesize = (opts['maxnodesize'] - opts['minnodesize']) / (maxMagToGray - minMagToGray)
    bnodesize = opts['minnodesize'] - knodesize * minMagToGray

    maxlam = 65535 * opts['lammaxrate']
    minlam = opts['lamrate'] * maxlam

    klam = (maxlam - minlam) / (maxMagToGray - minMagToGray)
    blam = minlam - klam * minMagToGray

    k1 = -1
    for j in range(targetsNum):

        if (targetsInfo[j]['xcenter']) < 0 or (targetsInfo[j]['ycenter']) < 0:
            continue

        x_center = round(targetsInfo[j]['xcenter'])
        y_center = round(targetsInfo[j]['ycenter'])

        # 记录偏移量
        xoffset = targetsInfo[j]['xcenter'] - x_center
        yoffset = targetsInfo[j]['ycenter'] - y_center

        mag = np.exp(-targetsInfo[j]['mag'] / 10)
        pointRadius = round(knodesize * mag + bnodesize) + 20
        maxgray = klam * mag + blam
        sigma = np.sqrt(maxgray / 1000)
        startX = round(x_center - pointRadius)
        startY = round(y_center - pointRadius)
        endX = round(x_center + pointRadius)
        endY = round(y_center + pointRadius)

        maxbounder = max(startX, startY, endX, endY)
        minbounder = min(startX, startY, endX, endY)
        if minbounder < 0 or maxbounder > 1650-1:
            continue


        # 将高斯光斑中心移动到非整数坐标位置
        crtNode = np.zeros((pointRadius * 2 + 1, pointRadius * 2 + 1))
        centerX = pointRadius
        centerY = pointRadius
        crtNode = corr_gaussian_spot(crtNode, (pointRadius * 2 + 1), sigma, (centerX, centerY), (xoffset, yoffset))

        # 给目标添加运动模糊
        starTWLen = 3
        starTWDirec = 90
        motionkernel = motion_blur_kernel(starTWLen, starTWDirec)
        crtNode = convolve2d(crtNode, motionkernel, mode='same', boundary='symm')


        maxcrtNode = np.max(crtNode)
        if maxcrtNode == 0:
            maxcrtNode = 1
        crtNode = crtNode / maxcrtNode
        crtNode = crtNode * maxgray


        k1 += 1
        TargetCoordinate.append([targetsInfo[j]['xcenter'], targetsInfo[j]['ycenter']])

        # Add the target to the star image
        starImg[startX:endX+1, startY:endY+1] = starImg[startX:endX+1, startY:endY+1] + crtNode

    return starImg, TargetCoordinate
