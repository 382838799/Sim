import os
import numpy as np
import pandas as pd
import time


from target_info import get_targets_loc_info
from targets_and_stars_params import set_targets_generate_params, set_stars_generate_params_double_log
from generate_targets import generate_target
from generate_stars import generate_image_double_log
from add_noises import add_noise
from save_data import save_results

def process_frame(i, mbData, lenDirec, imageSize, mean, backVar, targetsNum, magRange,
                  targetRange, expTime, frameFrequency,
                  pathSaveImageCoor, pathSaveTargetsCoor, pathSaveStarsCoor,
                  folderPath, filename_format="chapter24_{:02d}", file_extension=".xlsx",):

    """
    处理单帧的函数
    :param i:
    :param mbData:
    :param lenDirec:
    :param imageSize:
    :param mean:
    :param backVar:
    :param targetsNum:
    :param magRange:
    :param targetRange:
    :param expTime:
    :param frameFrequency:
    :param pathSaveImageCoor:
    :param pathSaveTargetsCoor:
    :param pathSaveStarsCoor:
    :param folderPath:
    :param filename_format:
    :param file_extension:
    :return:
    """

    starImg = np.zeros((imageSize, imageSize))

    # 自定义恒星属性数据星点横坐标、纵坐标、星等、赤经、赤纬
    # ------------------------------------------------------
    xlsxName = f"{filename_format.format(i+1)}{file_extension}"  # 格式化文件名
    ipdfile = os.path.join(folderPath, xlsxName)

    ipd = pd.read_excel(ipdfile)

    # 获取相应列
    # --------
    SX = ipd.iloc[:, 7]  # 第6列（索引从0开始）
    SY = ipd.iloc[:, 8]  # 第7列
    Smag = ipd.iloc[:, 2]  # 第3列
    SRa = ipd.iloc[:, 0]  # 第1列
    SDec = ipd.iloc[:, 1]  # 第2列

    # 创建 DataFrame，并命名列
    # ----------------------
    catStarlist = pd.DataFrame({'x': SX, 'y': SY, 'magnitude': Smag, 'ra': SRa, 'dec': SDec})

    # 将 DataFrame 转换为字典（类似 MATLAB 中的 table2struct）
    # ----------------------------------------------------
    starmat = catStarlist.to_dict(orient='records')


    # 计算角度和长度
    # ------------
    starAngle =  -lenDirec[i, 1]
    starLength = (lenDirec[i, 0] / frameFrequency) * expTime       # 除以帧频乘以曝光时间

    # 加载 mb 相关数据
    # ---------------
    targetsInfo = [{} for _ in range(targetsNum)]
    targetsInfo = get_targets_loc_info(i, mbData, targetsNum, targetsInfo)


    # 给 mb 设置相关参数，并生成 mb 光斑
    # ------------------------------
    prpts = set_targets_generate_params(targetRange)
    starImg, targetCoordinate = generate_target(starImg, prpts, targetsInfo, i, targetsNum)


    # 给 hx 设置相关参数，并生成 hx 光斑
    # ------------------------------
    # start = time.time()
    prpts = set_stars_generate_params_double_log(magRange)
    starImg, starCoordinate = generate_image_double_log(starImg, imageSize, prpts, starmat, starAngle, starLength)
    # end = time.time()
    # execution_time = end - start
    # print(f"程序执行时间: {execution_time}秒")

    # 为星图添加噪声
    # -----------
    starImg, noise = add_noise(mean, starImg, backVar)


    # 输出数据
    # -------
    save_results(i, starImg, targetCoordinate, starCoordinate, pathSaveImageCoor, pathSaveTargetsCoor, pathSaveStarsCoor)


    print(f"已经完成第{i}帧")
