import os
import numpy as np
from PIL import Image
import glob


def create_cropped_gif(folder_path, image_width, image_height, num_files, output_gif_path,
                       crop_size=1600, crop_offset=25, min_val=0, max_val=5000, frame_duration=1000):
    """
    从文件夹中读取RAW图像文件，裁剪中心部分，并创建GIF动画。

    参数:
    -----------
    folder_path : str
        包含RAW图像文件的文件夹路径
    image_width : int
        原始RAW图像的宽度
    image_height : int
        原始RAW图像的高度
    num_files : int
        文件夹中预期的RAW文件数量
    output_gif_path : str
        输出GIF的保存路径
    crop_size : int, 可选
        裁剪的正方形区域大小（默认：2000）
    crop_offset : int, 可选
        从图像边缘开始裁剪的偏移量（默认：1000）
    min_val : int, 可选
        对比度调整的最小值（默认：0）
    max_val : int, 可选
        对比度调整的最大值（默认：5000）
    frame_duration : int, 可选
        每帧的持续时间（单位：毫秒，默认：100）

    返回值:
    --------
    str
        如果GIF成功创建，则返回成功消息
    """
    # 获取文件夹中所有.raw文件
    file_list = sorted(glob.glob(os.path.join(folder_path, '*.raw')))

    # 检查文件数量是否与预期匹配
    if len(file_list) != num_files:
        raise ValueError(f'文件夹必须正好包含 {num_files} 个.raw文件，但找到 {len(file_list)} 个')

    # 创建一个空数组来存储裁剪后的图像
    images = np.zeros((len(file_list), crop_size, crop_size), dtype=np.uint16)

    # 读取并裁剪每张图像
    for i in range(len(file_list)):
        # 根据命名规则读取文件
        filename = os.path.join(folder_path, f'{i:04d}.raw')

        # 读取.raw文件
        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16, count=image_width * image_height)

        # 将数据重塑为图像尺寸
        img = data.reshape((image_height, image_width))

        # 裁剪中心区域
        img_cropped = img[crop_offset:crop_offset + crop_size, crop_offset:crop_offset + crop_size]

        # 存储裁剪后的图像
        images[i] = img_cropped

    # 创建GIF帧
    frames = []
    for i in range(len(images)):
        # 获取当前图像
        current_image = images[i]

        # 对比度调整：将图像标准化到0-255
        current_image_normalized = np.clip(current_image, min_val, max_val)
        current_image_normalized = ((current_image_normalized - min_val) /
                                    (max_val - min_val) * 255).astype(np.uint8)

        # 创建PIL图像
        frame = Image.fromarray(current_image_normalized, mode='L')
        frames.append(frame)

    # 保存为GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,  # 每帧的持续时间（单位：毫秒）
        loop=0  # 0表示无限循环
    )

    return f'GIF文件已成功创建于 {output_gif_path}'


if __name__ == '__main__':
    # 默认参数
    folder_path = '/mnt/d/XT/AAA_Simulation/GY/chapter3/Image/'  # 替换为实际文件夹路径
    image_width = 8900  # 假设的图像宽度
    image_height = 8900  # 假设的图像高度
    num_files = 40  # 预期的.raw文件数量
    output_gif_path = '/mnt/d/XT/AAA_Simulation/GY/chapter3/GIF/output.gif'  # 输出文件路径

    # 使用默认参数调用函数
    result = create_cropped_gif(
        folder_path=folder_path,
        image_width=image_width,
        image_height=image_height,
        num_files=num_files,
        output_gif_path=output_gif_path
    )

    print(result)