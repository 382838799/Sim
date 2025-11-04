import os
import pandas as pd
import scipy.io
from scipy.spatial.distance import cdist
import numpy as np

def process_excel_files(pathSaveImage, mbdata_folder, targetsNum, num_files, dis_pixel):
    """
    提取所有帧中距离目标dis_pixel恒星的星等

    参数:
    pathSaveImage (str): Excel 文件所在的文件夹路径。
    mbdata_folder (str): 目标坐标文件所在的文件夹路径。
    targetsNum (int): 目标数量。
    num_files (int): Excel 文件数量。
    dis_pixel (float): 距离阈值（像素）。

    返回:
    list: 2D 列表，包含所有满足条件的恒星星等。
    """
    # 参数设置
    output_folder = os.path.join(pathSaveImage, 'Results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, 'filterd_mag.txt')

    file_prefix = 'JG02_chapter001_'   # Excel 文件名前缀                    # ************************************************
    file_suffix = '.xlsx'    # Excel 文件名后缀

    # 目标数据路径
    # -----------
    # mdh_filename = os.path.join(mbdata_folder, 'config', 'mdh小银河.mat')
    mdh_filename = './config/jgmb75.mat'                                     # ************************************************

    # 读取 [x2, y2] 坐标
    mat_data = scipy.io.loadmat(mdh_filename)
    mat_data = mat_data['data']['PIC']
    tarinfo_x_list = mat_data[0, 0]['X'][0, 0]
    tarinfo_y_list = mat_data[0, 0]['Y'][0, 0]

    result = np.array([item[0] for sublist in tarinfo_x_list for item in sublist])
    tarinfo_x_list = result.reshape(num_files, targetsNum)

    result = np.array([item[0] for sublist in tarinfo_y_list for item in sublist])
    tarinfo_y_list = result.reshape(num_files, targetsNum)
    # ----------------------------------------------------------------------------------


    # 初始化 2D 列表，用于存储所有满足条件的第三列值
    all_selected_values = []

    # 打开输出文件
    with open(output_filename, 'w', encoding='utf-8') as fid:
        fid.write('粘连的恒星星等计算结果\n')
        fid.write('==============================\n\n')

        for frame in range(1, num_files + 1):
            # 生成当前 Excel 文件名
            current_file_name = f'{file_prefix}{frame:03d}{file_suffix}'
            current_file_path = os.path.join(pathSaveImage, current_file_name)

            # 检查文件是否存在
            if not os.path.exists(current_file_path):
                print(f'文件 {current_file_path} 不存在，已跳过。')
                all_selected_values.append([])
                continue

            # 尝试读取 Excel 文件
            try:
                data1 = pd.read_excel(current_file_path)
            except Exception as e:
                print(f'读取文件 {current_file_path} 时出错：{e}，已跳过。')
                all_selected_values.append([])
                continue

            # 检查列数是否足够
            if data1.shape[1] < 7:
                print(f'文件 {current_file_path} 的列数少于 7 列，已跳过。')
                all_selected_values.append([])
                continue

            # 提取坐标和值
            x = data1.iloc[:, 7].values  # 第 6 列，索引 5
            y = data1.iloc[:, 8].values  # 第 7 列，索引 6
            values = data1.iloc[:, 2].values  # 第 3 列，索引 2

            # 获取当前帧的 [x2, y2] 坐标------------- 此处设置目标的坐标
            currentX2 = tarinfo_x_list[frame - 1]
            currentY2 = tarinfo_y_list[frame - 1]
            # currentX2 = 825
            # currentY2 = 825



            # 检查是否有有效的 [x2, y2] 坐标
            if len(currentX2) == 0:
                print(f'文件: {current_file_name}')
                print('对应 [x2, y2]: 无有效坐标')
                print(f'没有满足距离小于 {dis_pixel} 像素的值。\n')
                fid.write(f'文件: {current_file_name}\n')
                fid.write('对应 [x2, y2]: 无有效坐标\n')
                fid.write(f'没有满足距离小于 {dis_pixel} 像素的值。\n\n')
                fid.write('--------------------\n\n')
                all_selected_values.append([])
                continue

            # 创建坐标矩阵
            coords1 = np.column_stack((x, y))
            coords2 = np.column_stack((currentX2, currentY2))

            # 计算距离矩阵
            distance_matrix = cdist(coords1, coords2, 'euclidean')

            # 判断哪些点满足距离小于 dis_pixel 像素
            is_within_threshold = np.any(distance_matrix < dis_pixel, axis=1)

            # 提取满足条件的索引和值
            selected_indices = np.where(is_within_threshold)[0]
            selected_values = values[selected_indices]
            selected_x = x[selected_indices]
            selected_y = y[selected_indices]

            # 格式化选定值，保留 13 位小数
            formatted_values = [f'{val:.13f}' for val in selected_values]

            # 写入文件
            fid.write(f'文件: {current_file_name}\n')
            fid.write('对应 [x2, y2]:\n')
            for k in range(len(currentX2)):
                fid.write(f'  [{currentX2[k]:.4f}, {currentY2[k]:.4f}]\n')

            if len(selected_values) > 0:
                fid.write(f'满足距离小于 {dis_pixel} 像素的第 3 列值:\n')
                for j in range(len(selected_values)):
                    fid.write(f'x: {selected_x[j]:.4f}, y: {selected_y[j]:.4f}, value: {formatted_values[j]}\n')
            else:
                fid.write(f'没有满足距离小于 {dis_pixel} 像素的值。\n')

            fid.write('\n--------------------\n\n')

            # 将选定值添加到 2D 列表
            all_selected_values.append(selected_values.tolist())


            # 显示处理进度
            print(f'已处理文件 {frame}/{num_files}: {current_file_name}')

    print(f'所有文件已处理完成，结果已写入 {output_filename}')

    return all_selected_values


# 主程序入口
if __name__ == '__main__':

    pathSaveImage = '/mnt/d/XT/AAA_Simulation/JG02/10/chapter001/StarProperties/'                # ************************************************
    mbdata_folder = '/mnt/d/XT/AAA_Simulation/JG02/10/chapter001/StarProperties/'                # ************************************************
    targetsNum = 1
    num_files = 75                                                                      # ************************************************
    dis_pixel = 35                                                                      # ************************************************





    skip_values = process_excel_files(pathSaveImage, mbdata_folder, targetsNum, num_files, dis_pixel)

    with open('./config/skip_values_10s_chapter001_35.txt', 'w') as f:                      # ************************************************
        for value in skip_values:
            f.write(f"{value}\n")

    
    print("Skip values 已保存到 skip_values.txt")