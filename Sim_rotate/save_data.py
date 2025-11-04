import os
import numpy as np


# ================================================================
# 函数: 保存数据
# ================================================================
def save_results(i, star_img, target_coordinates, star_data, path_save_image_coor, path_save_targets_coor,
                 path_save_stars_coor, sign_need_output_image=1):
    # 生成文件名
    i_str = f'{i:04d}'
    
    # 整个星图顺时针旋转90°
    star_img = np.rot90(star_img, k=-1)

    # 如果需要保存图像数据
    if sign_need_output_image == 1:
        raw_name = os.path.join(path_save_image_coor, f'{i_str}.raw')
        with open(raw_name, 'wb') as bin_fid:
            # 将图像数据保存为小端格式
            bin_fid.write(star_img.astype(np.uint16).tobytes())

    # 保存目标坐标
    target_name = f'{i_str}target.ipd'
    target_name_full = os.path.join(path_save_targets_coor, target_name)
    with open(target_name_full, 'w') as ipd_fid:
        for coord in target_coordinates:
            # ipd_fid.write(f'{coord[0]} {coord[1]}\n')
            ipd_fid.write(f'{coord[1]} {1650-coord[0]}\n')

    # 保存星点坐标和数据
    star_name = f'{i_str}star.ipd'
    star_name_full = os.path.join(path_save_stars_coor, star_name)
    with open(star_name_full, 'w') as star_fid:
        # 可选：写入星点的数量信息（如果需要）
        star_fid.write(f"Star Num:{len(star_data)}\n")    # 写入恒星数量
        star_fid.write(f"Target Num:{len(target_coordinates)}\n\n\n")

        # 存储恒星
        for star in star_data:
            # 假设star是一个字典，包含x, y, magnitude, ra, dec   0:x,  1:y, 2:mag, 3:ra, 4:dec
            # star_fid.write(f'0000,{star["y"]},{1650-star["x"]},{star["magnitude"]},{star["ra"]},{star["dec"]}\n')
            star_fid.write(f'0000,{star[1]},{1650-star[0]},{star[2]},{star[3]},{star[4]}\n')

        # 存储目标
        for coord in target_coordinates:
            # ipd_fid.write(f'{coord[0]} {coord[1]}\n')
            star_fid.write(f'0001,{coord[1]},{1650-coord[0]}\n')
