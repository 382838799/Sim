import os
import time
import scipy.io
import logging
import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.io import loadmat

from process_frame import process_frame   # 单帧图像生成函数
from save_gif import create_cropped_gif   # 生成 gif 图
from star_catalog_parquet_xyz_fixed import process_from_csv # 读星表函数


if __name__ == '__main__':

    # # ==============================查星表====================================
    # for i in range(106, 116):
    #     chapter = f'chapter{i:03d}'
    #     csv = f'{chapter}.csv'

    #     # 设置参数（恢复原来的参数设置方式）
    #     input_dir = '/mnt/d/XT/AAA_Simulation/GaiaDr3/Parquet/'  # 768个星表Parquet文件所在的文件夹路径 ***不需要变
    #     fits_output_dir = '/mnt/d/XT/AAA_Simulation/temp/'  # FITS文件输出目录 ***不需要变

    #     csv_file = f'/mnt/d/XT/AAA_Simulation/01/Point/{csv}'  # 保持CSV格式的输入参数文件
    #     excel_output_dir = f'/mnt/d/XT/AAA_Simulation/GY_01/{chapter}/StarProperties/'  # Excel文件输出目录

    #     if not os.path.exists(excel_output_dir):
    #         os.makedirs(excel_output_dir)
    #         logging.info(f"创建目录:{excel_output_dir}")

    #     # 开始处理XT
    #     start_time = datetime.datetime.now()
    #     logging.info(f"开始处理，时间: {start_time}")

    #     process_from_csv(
    #         csv_file,
    #         input_dir,
    #         fits_output_dir,
    #         excel_output_dir
    #     )

    #     end_time = datetime.datetime.now()
    #     logging.info(f"处理完成，时间: {end_time}")
    #     logging.info(f"总耗时: {end_time - start_time}")


    # # =========================================================================

    # logging.info("****************所有星表均已查询完毕*********************")

    # # ==========================================================================


    # # ==============================绘图====================================

    for number in range(1, 2):

        # 计时开始
        # ----------
        start_time = time.time()



        # 迭代参数
        # -------
        chapter = f'chapter{number:03d}'


        # 创建目录
        # -------
        # 输出图像和数据的一级目录
        pathSaveImage = f'/mnt/d/XT/AAA_Simulation/JG02/4/{chapter}/'  # 图片存储路径
        # 如果该目录不存在，则创建该目录
        if not os.path.exists(pathSaveImage):
            os.makedirs(pathSaveImage)

        # 图像和数据的二级目录
        pathSaveTargetsCoor = os.path.join(pathSaveImage, 'Targets')        # MB_IPD 存储路径
        pathSaveStarsCoor = os.path.join(pathSaveImage, 'Stars')            # HX_IPD 存储路径
        pathSaveImageCoor = os.path.join(pathSaveImage, 'Image')            # 图像   存储路径

        # 创建目录
        os.makedirs(pathSaveTargetsCoor, exist_ok=True)
        os.makedirs(pathSaveStarsCoor, exist_ok=True)
        os.makedirs(pathSaveImageCoor, exist_ok=True)


        # 参数初始化
        # --------
        imageSize = 1650                    # 图像尺寸
        mean = 1000                         # 背景均值
        backVar = 50                        # 背景方差
        frameNum = 40                       # 总帧数                  ****************************************************此处需要修改
        targetsNum = 12                     # MB个数                   *******
        magRange = [1,18]                 # HX星等范围
        targetRange= [7,14]                 # 拟合与星等关系直线用    [10,14];
        expTime = 4                       # 曝光时间                *********************
        frameFrequency = 5                 # 帧频                     *********************

        # 设置并行化参数
        parallelization_enabled = True  # 设置为 False 禁用并行化，True 启用并行化


        # 加载 mb 的坐标
        # -------------
        mbData = scipy.io.loadmat(f'/mnt/d/XT/AAA_Simulation/JG02/JG02Data/geotoxt/4/target{number:03d}.mat')
        mbData = mbData['data']['PIC']
        targetsNum = len(mbData[0,0]['X'][0,0][0])    # 自动计算目标个数
        


        # 加载每一帧的拉线方向和长度
        # ----------------------
        lenDirec = loadmat(f'/mnt/d/XT/AAA_Simulation/JG02/JG02Data/geohx/{number:03d}章.mat')
        lenDirec = lenDirec['hx_v_direct']
        frameNum = 75 #len(lenDirec)

        # 存放相面坐标数据的文件夹
        # --------------------
        folderPath = f'/mnt/d/XT/AAA_Simulation/JG02/4/{chapter}/StarProperties/'


        # 相面坐标的格式化命名
        # filename_format = f'GY01_{chapter}_{:03d}'
        filename_format = 'JG02_' + f'{chapter}_' + '{:03d}'


        file_extension = '.xlsx'


        if parallelization_enabled:
            # 使用并行处理
            # os.cpu_count() // 2 ：并行个数设置为最大核心数的一半
            with ProcessPoolExecutor(max_workers = os.cpu_count() // 2) as executor:
                process_frame_partial = partial(
                    process_frame,
                    mbData=mbData,
                    lenDirec=lenDirec,
                    imageSize=imageSize,
                    mean=mean,
                    backVar=backVar,
                    targetsNum=targetsNum,
                    magRange=magRange,
                    targetRange=targetRange,
                    expTime=expTime,
                    frameFrequency=frameFrequency,
                    pathSaveImageCoor=pathSaveImageCoor,
                    pathSaveTargetsCoor=pathSaveTargetsCoor,
                    pathSaveStarsCoor=pathSaveStarsCoor,
                    folderPath=folderPath,
                    filename_format=filename_format,
                    file_extension=file_extension
                )
                # 执行并行任务
                for _ in executor.map(process_frame_partial, range(frameNum)):
                    pass  # map 返回的结果不使用，仅确保任务完成
        else:
            # 单步执行
            for i in range(frameNum):
                process_frame(i, mbData, lenDirec, imageSize, mean, backVar, targetsNum, magRange, targetRange, expTime, frameFrequency,
                            pathSaveImageCoor, pathSaveTargetsCoor, pathSaveStarsCoor,
                            folderPath, filename_format, file_extension)



        # 生成 gif 图
        # ----------
        # 默认参数
        folder_path = f'/mnt/d/XT/AAA_Simulation/JG02/4/{chapter}/Image/'  # 替换为实际文件夹路径
        image_width = 1650  # 假设的图像宽度
        image_height = 1650  # 假设的图像高度
        num_files = frameNum  # 预期的.raw文件数量    # ******************************************************************************
        output_gif_path = f'/mnt/d/XT/AAA_Simulation/JG02/4/{chapter}/GIF/output.gif'  # 输出文件路径
        output_gif = f'/mnt/d/XT/AAA_Simulation/JG02/4/{chapter}/GIF/'  # 输出文件路径

        if not os.path.exists(output_gif):
            os.makedirs(output_gif)
            logging.info(f"创建目录:{output_gif}")
   

        # 使用默认参数调用函数
        result = create_cropped_gif(
            folder_path=folder_path,
            image_width=image_width,
            image_height=image_height,
            num_files=num_files,
            output_gif_path=output_gif_path
        )

        print(result)                    


        # 结束时间
        # ------
        end_time = time.time()
        # 计算并打印程序执行时间
        execution_time = end_time - start_time
        print(f"程序执行时间: {execution_time}秒")

    #     # ==============================================================================================================