# 优化后的第三版，读取.parquet 而非 .csv，保持其他逻辑不变
import pandas as pd
import numpy as np
from astropy.io import fits
import re
import healpy as hp
import os
import logging
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 清理文件名，移除非法字符
def sanitize_filename(filename):
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='ignore')
    return re.sub(r'[<>:"/\\|?*]', '-', str(filename))


# 1. 读取大范围数据并保存为 FITS 文件 - 修改为读取Parquet文件
def filter_data_by_fov_and_mag(path, ra_cam, dec_cam, radius_deg, min_mag, max_mag, nside, output_dir, title=None):
    """逐个像素块读取数据，筛选符合条件的天区数据，结合星等范围"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 目标天区的角度（以弧度表示）
    target_ra_rad = np.radians(ra_cam)
    radius_rad = np.radians(radius_deg)
    vec = hp.ang2vec(np.radians(90 - dec_cam), target_ra_rad)

    try:
        # 使用 Healpix 的 query_disc 查找视场内的像素块索引
        pixel_indices = hp.query_disc(nside=nside, vec=vec, radius=radius_rad, inclusive=True, nest=True)
    except Exception as e:
        logging.error(f"Error in query_disc: {e}")
        return None

    if len(pixel_indices) == 0:
        logging.warning("No pixel indices found for the given coordinates and radius.")
        return None

    # 获取符合条件的像素块文件 - 修改为读取Parquet文件
    try:
        pixel_files = [f for f in os.listdir(path) if f.startswith('pixel_') and f.endswith('.parquet')]
    except Exception as e:
        logging.error(f"Error listing directory {path}: {e}")
        return None

    pixel_indices_set = set(pixel_indices)  # 转换为集合以加速查找

    relevant_files = []
    for pixel_file in pixel_files:
        # 提取文件名中的索引（像素块编号）
        try:
            pixel_index = int(pixel_file.split('_')[1].split('.')[0])
            if pixel_index in pixel_indices_set:
                relevant_files.append(pixel_file)
        except (IndexError, ValueError):
            continue

    if not relevant_files:
        logging.warning("No relevant pixel files found.")
        return None

    # 预分配存储空间
    cols = ['ra', 'dec', 'phot_g_mean_mag', 'pmra', 'pmdec', 'parallax', 'radial_velocity']
    all_data = {col: [] for col in cols}
    total_rows = 0

    for pixel_file in relevant_files:
        pixel_file_path = os.path.join(path, pixel_file)

        try:
            # 读取Parquet文件 - 修改此处
            df = pd.read_parquet(pixel_file_path, columns=cols)
            
            # 星等筛选
            mask = (df['phot_g_mean_mag'] >= min_mag) & (df['phot_g_mean_mag'] <= max_mag)
            if not mask.any():
                continue

            # 添加筛选后的数据
            for col in cols:
                all_data[col].append(df.loc[mask, col].values)
            total_rows += mask.sum()
        except Exception as e:
            logging.warning(f"Error reading {pixel_file}: {e}")
            continue

    if total_rows == 0:
        logging.warning("No data passed the filters.")
        return None

    # 合并数据
    for col in cols:
        if all_data[col]:
            all_data[col] = np.concatenate(all_data[col])
        else:
            all_data[col] = np.array([])

    # 保存为 FITS 文件 - 修复 NumPy 弃用警告
    try:
        # 创建结构化数组
        dtype = [(col, 'f8') for col in cols]
        # 使用正确的 NumPy API，避免使用已弃用的 np.core.records
        structured_data = np.array(list(zip(*[all_data[col] for col in cols])), dtype=dtype)

        hdu = fits.BinTableHDU(structured_data)
        output_filename = os.path.join(output_dir, f"gaia_stars_{sanitize_filename(title)}.fits")
        hdu.writeto(output_filename, overwrite=True)
        logging.info(f"Saved to FITS file: {output_filename} with {total_rows} stars")
        return output_filename
    except Exception as e:
        logging.error(f"Error saving FITS file: {e}")
        return None


# 2. 根据光行差修正计算小范围赤经赤纬
def apply_aberration_correction(ra, dec, pmra, pmdec, parallax, radial_velocity):
    """Apply aberration correction to star coordinates"""
    # 常量定义
    DAS2R = np.pi / (180.0 * 3600000.0)  # 毫角秒到弧度的转换常数
    VF = 0.21094502  # 速度常数

    # 地球和太阳系参数
    amprms = [
        9.4688,  # 时间间隔（儒略年）
        0.004328052823849642, -0.9373677410717631, -0.40616055697408754,  # 地球的日心坐标（天文单位 AU）
        0.004328052823849642/1.0188, -0.9373677410717631/1.0188, -0.40616055697408754/1.0188,  # 地球的日心方向（单位向量）
        0.0000000197,  # 太阳引力辐射常数 * 2 / （太阳-地球距离）
        0.00009779, 0.00000044, 0.00000019,  # 地球的日心速度（以光速单位 c 表示）
        0.9999,  # 1 - v^2 的平方根，其中 v = abv 的模
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 进动/章动矩阵（3x3）
    ]

    pmt = amprms[0]
    gr2e = amprms[7]
    ab1 = amprms[11]
    eb = np.array(amprms[1:4])  # barycentric position of Earth (AU)
    ehn = np.array(amprms[4:7])  # heliocentric direction of Earth (unit vector)
    abv = np.array(amprms[8:11])  # barycentric Earth velocity (units of c)

    # 单位转换
    pr_radians_per_year = pmra * DAS2R
    pd_radians_per_year = pmdec * DAS2R
    px_radians = parallax * DAS2R
    rv_in_au_per_year = radial_velocity * 1.0e-5
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    # 计算笛卡尔坐标
    cos_ra = np.cos(ra_rad)
    sin_ra = np.sin(ra_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)

    q = np.empty((len(ra), 3))
    q[:, 0] = cos_ra * cos_dec
    q[:, 1] = sin_ra * cos_dec
    q[:, 2] = sin_dec

    # 计算空间运动（radians per year）
    pxr = px_radians
    w = VF * rv_in_au_per_year * pxr

    em = np.empty_like(q)
    em[:, 0] = (-pr_radians_per_year * q[:, 1]) - (pd_radians_per_year * cos_ra * sin_dec) + (w * q[:, 0])
    em[:, 1] = (pr_radians_per_year * q[:, 0]) - (pd_radians_per_year * sin_ra * sin_dec) + (w * q[:, 1])
    em[:, 2] = (pd_radians_per_year * cos_dec) + (w * q[:, 2])

    # 计算地心方向
    p = np.empty_like(q)
    for i in range(3):
        p[:, i] = q[:, i] + (pmt * em[:, i]) - (pxr * eb[i])

    # 单位化
    norm = np.sqrt(np.sum(p * p, axis=1))
    norm = np.maximum(norm, 1e-10)  # 避免除以零
    pn = p / norm[:, None]

    # 光线偏转修正（限制在太阳盘内）
    pde = np.sum(pn * ehn, axis=1)
    pdep1 = 1.0 + pde
    pdep1 = np.maximum(pdep1, 1.0e-5)
    w = gr2e / pdep1

    p1 = np.empty_like(pn)
    for i in range(3):
        p1[:, i] = pn[:, i] + (w * (ehn[i] - pde * pn[:, i]))

    # 周年光行差修正
    p1dv = np.sum(p1 * abv, axis=1)
    p1dvp1 = p1dv + 1.0
    w = 1.0 + p1dv / (ab1 + 1.0)

    p2 = np.empty_like(p1)
    for i in range(3):
        p2[:, i] = (ab1 * p1[:, i] + w * abv[i]) / p1dvp1

    # 单位化
    p2_norm = np.sqrt(np.sum(p2 * p2, axis=1))
    p2_norm = np.maximum(p2_norm, 1e-10)  # 避免除以零
    p2 = p2 / p2_norm[:, None]

    # 将笛卡尔坐标转换回天球坐标（赤经和赤纬）
    ra_corrected = np.arctan2(p2[:, 1], p2[:, 0])
    ra_corrected = (np.degrees(ra_corrected) + 360) % 360  # 计算赤经
    dec_corrected = np.arcsin(np.clip(p2[:, 2], -1.0, 1.0))  # 裁剪值以避免NaN
    dec_corrected = np.degrees(dec_corrected)  # 计算赤纬

    return ra_corrected, dec_corrected


# 3. 读取 FITS 文件并将数据转换
def read_fits_to_df(fits_file):
    """读取FITS文件并提取关键字段"""
    try:
        with fits.open(fits_file) as hdulist:
            data = hdulist[1].data
        return data['ra'], data['dec'], data['phot_g_mean_mag'], data['pmra'], data['pmdec'], data['parallax'], data[
            'radial_velocity']
    except Exception as e:
        logging.error(f"Error reading FITS file {fits_file}: {e}")
        return None, None, None, None, None, None, None


# 4. 坐标系转换函数（例如相面坐标转换）
# 修复版本的 to_camera_and_pixel_coords 函数
def to_camera_and_pixel_coords(ra, dec, f, W, H, ra_cam, dec_cam, ra_camy, dec_camy, ra_camx, dec_camx, pixel_scale):
    """
    若传入多个星体，则逐个处理；若为单颗星，则将 ra, dec 提取为标量。
    修复：返回完整的数组，无效点用NaN填充，确保输出长度一致
    """
    if np.ndim(ra) > 0 and ra.size > 1:
        x_pixels = np.full(ra.shape, np.nan)  # 预分配全NaN数组
        y_pixels = np.full(ra.shape, np.nan)
        valid_flags = np.full(ra.shape, False, dtype=bool)
        
        for i, (r, d) in enumerate(zip(ra, dec)):
            xp, yp, valid = to_camera_and_pixel_coords(np.array([r]), np.array([d]), f, W, H,
                                                       ra_cam, dec_cam, ra_camy, dec_camy, ra_camx, dec_camx,
                                                       pixel_scale)
            if valid.size > 0 and valid[0]:
                x_pixels[i] = xp[0]
                y_pixels[i] = yp[0]
                valid_flags[i] = True
        
        return x_pixels, y_pixels, valid_flags

    # 单颗星处理：将 ra, dec 提取为标量
    ra = np.radians(ra)[0]
    dec = np.radians(dec)[0]

    # 转换相机三轴为弧度
    ra_cam = np.radians(ra_cam)
    dec_cam = np.radians(dec_cam)
    ra_camy = np.radians(ra_camy)
    dec_camy = np.radians(dec_camy)
    ra_camx = np.radians(ra_camx)
    dec_camx = np.radians(dec_camx)

    ## 计算星体在 J2000 下的单位向量
    hx_x = np.cos(dec) * np.cos(ra)
    hx_y = np.cos(dec) * np.sin(ra)
    hx_z = np.sin(dec)
    # 构造相机坐标系基向量（来自三个参考轴）
    z_x = np.cos(dec_cam) * np.cos(ra_cam)
    z_y = np.cos(dec_cam) * np.sin(ra_cam)
    z_z = np.sin(dec_cam)

    y_x = np.cos(dec_camy) * np.cos(ra_camy)
    y_y = np.cos(dec_camy) * np.sin(ra_camy)
    y_z = np.sin(dec_camy)

    x_x = np.cos(dec_camx) * np.cos(ra_camx)
    x_y = np.cos(dec_camx) * np.sin(ra_camx)
    x_z = np.sin(dec_camx)

    A = np.array([
        [x_x, y_x, z_x],
        [x_y, y_y, z_y],
        [x_z, y_z, z_z]
    ])

    # 计算星体在相机坐标系下的坐标
    hx_cam = np.linalg.inv(A) @ np.array([[hx_x], [hx_y], [hx_z]])

    # 转像面部分：使用 np.vstack 生成齐次坐标（保持原代码）
    projection_matrix = np.array([
        [f / 0.01, 0, W / 2, 0],
        [0, f / 0.01, H / 2, 0],
        [0, 0, 1, 0]
    ])

    hx_pos = projection_matrix @ np.vstack((hx_cam, np.array([[1]]))) / hx_cam[2]
    x_pixel = hx_pos[0]
    y_pixel = hx_pos[1]

    valid = (x_pixel >= 0) & (x_pixel <= W) & (y_pixel >= 0) & (y_pixel <= H)
    
    # 返回标量数组和布尔值
    if valid:
        return np.array([x_pixel[0]]), np.array([y_pixel[0]]), np.array([True])
    else:
        return np.array([np.nan]), np.array([np.nan]), np.array([False])


# 5. 将数据保存为 Excel 文件
# *********与不加转台的代码不同，本代码在此处交换了x,y***********************
def save_to_excel(ra, dec, phot_g_mean_mag, pmra, pmdec, parallax, radial_velocity, x, y, output_dir, filename):
    """将处理后的数据保存为Excel文件"""
    try:
        filename = sanitize_filename(filename)
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame({
            'ra (deg)': ra,
            'dec (deg)': dec,
            'phot_g_mean_mag': phot_g_mean_mag,
            'pmra (mas/year)': pmra,
            'pmdec (mas/year)': pmdec,
            'parallax (mas)': parallax,
            'radial_velocity (km/s)': radial_velocity,
            'x_pixel': y,
            'y_pixel': x
        })
        output_path = os.path.join(output_dir, filename)
        df.to_excel(output_path, index=False)
        logging.info(f"数据已保存至 {output_path}, 包含 {len(df)} 个星点")
        return output_path
    except Exception as e:
        logging.error(f"Error saving Excel file {filename}: {e}")
        return None


# 6. 从 CSV 文件读取数据并生成多组文件
def process_single_row(row, data, f, W, H, pixel_scale, min_mag, max_mag, input_dir, fits_output_dir, excel_output_dir):
    """处理CSV文件中的单行数据"""
    try:
        row_id = row
        # 确保使用str()处理可能的非字符串值
        title = str(data.iloc[row, 0])
        ra_cam = (data.iloc[row, 5] + 360) % 360  # 第六列数据
        dec_cam = data.iloc[row, 6]  # 第七列数据
        ra_camy = (data.iloc[row, 3] + 360) % 360  # 第四列数据
        dec_camy = data.iloc[row, 4]  # 第五列数据
        ra_camx = (data.iloc[row, 1] + 360) % 360  # 第二列数据
        dec_camx = data.iloc[row, 2]  # 第三列数据
        radius_deg = 16
        nside = 8

        logging.info(f"处理行 {row_id}: 目标 {title}, RA = {ra_cam}, DEC = {dec_cam}")

        # 1. 读取大范围数据并保存为 FITS 文件
        fits_file = filter_data_by_fov_and_mag(input_dir, ra_cam, dec_cam, radius_deg, min_mag, max_mag, nside,
                                               fits_output_dir, title)

        if fits_file is None:
            logging.warning(f"行 {row_id}: 无法生成FITS文件")
            return

        # 2. 读取 FITS 文件并转换为 DataFrame
        ra, dec, phot_g_mean_mag, pmra, pmdec, parallax, radial_velocity = read_fits_to_df(fits_file)

        if ra is None:  # 检查读取是否失败
            return

        # 3. 将 nan 值替换为适当的默认值
        pmra = np.nan_to_num(pmra, nan=0)
        pmdec = np.nan_to_num(pmdec, nan=0)
        parallax = np.nan_to_num(parallax, nan=0.00000001)
        radial_velocity = np.nan_to_num(radial_velocity, nan=0)

        # 4. 根据光行差修正计算新的赤经赤纬
        ra_new, dec_new = apply_aberration_correction(ra, dec, pmra, pmdec, parallax, radial_velocity)

        # 5. 转换坐标系并筛选有效点
        x_pixel, y_pixel, valid = to_camera_and_pixel_coords(ra_new, dec_new, f, W, H,
                                                        ra_cam, dec_cam,
                                                        ra_camy, dec_camy,
                                                        ra_camx, dec_camx,
                                                        pixel_scale)

        if not np.any(valid):
            logging.warning(f"行 {row_id}: 无有效星点")
            return

        # 6. 获取有效数据
        filtered_ra = ra_new[valid]
        filtered_dec = dec_new[valid]
        filtered_mag = phot_g_mean_mag[valid]
        filtered_pmra = pmra[valid]
        filtered_pmdec = pmdec[valid]
        filtered_parallax = parallax[valid]
        filtered_radial_velocity = radial_velocity[valid]
        filtered_x = x_pixel[valid]  # 确保只选择有效的像素坐标
        filtered_y = y_pixel[valid]

        # 7. 保存结果
        output_filename = f"{sanitize_filename(title)}.xlsx"
        save_to_excel(filtered_ra, filtered_dec, filtered_mag, filtered_pmra, filtered_pmdec,
                    filtered_parallax, filtered_radial_velocity, filtered_x, filtered_y,
                    excel_output_dir, output_filename)

    except Exception as e:
        logging.error(f"处理行 {row_id} 时发生错误: {e}")


# 7. 主函数：使用多进程处理 CSV 文件中的每一行
def process_from_csv(csv_file, input_dir, fits_output_dir, excel_output_dir, start_row=0, end_row=None, workers=None):
    """处理CSV文件中的所有行"""
    # 创建输出目录
    os.makedirs(fits_output_dir, exist_ok=True)
    os.makedirs(excel_output_dir, exist_ok=True)

    # 读取CSV文件
    try:
        data = pd.read_csv(csv_file)
        logging.info(f"已读取CSV文件，共 {len(data)} 行")
    except Exception as e:
        logging.error(f"读取CSV文件失败: {e}")
        return

    # 设置处理范围
    if end_row is None or end_row > len(data):
        end_row = len(data)

    if start_row >= end_row:
        logging.error(f"起始行 {start_row} 大于或等于结束行 {end_row}")
        return

    # 相机参数
    f = 418.3870309  # 相机焦距（mm）
    W, H = 8900, 8900  # 图像大小（像素）
    pixel_scale = 3.23  # 像素角度（arcsec/pixel）
    min_mag = 0.0  # 星等最小值
    max_mag = 15.0  # 星等最大值

    # 设置进程数
    if workers is None:
        workers = max(1, cpu_count() // 2)  # 默认使用一半的CPU核心

    logging.info(f"开始处理行 {start_row} 到 {end_row - 1}，使用 {workers} 个进程")

    # 使用偏函数设置固定参数
    process_func = partial(
        process_single_row,
        data=data,
        f=f, W=W, H=H,
        pixel_scale=pixel_scale,
        min_mag=min_mag,
        max_mag=max_mag,
        input_dir=input_dir,
        fits_output_dir=fits_output_dir,
        excel_output_dir=excel_output_dir
    )

    # 多进程处理
    try:
        with Pool(workers) as pool:
            pool.map(process_func, range(start_row, end_row))
    except Exception as e:
        logging.error(f"多进程处理时出错: {e}")


# 主程序入口
if __name__ == '__main__':
    # 设置参数（恢复原来的参数设置方式）
    csv_file = '/mnt/d/XT/AAA_Simulation/GY/chapter3/Point/chapter3.csv'  # 保持CSV格式的输入参数文件
    input_dir = '/mnt/d/XT/AAA_Simulation/GaiaDr3/Parquet/'  # 768个星表Parquet文件所在的文件夹路径
    fits_output_dir = '/mnt/d/XT/AAA_Simulation/temp/'  # FITS文件输出目录
    excel_output_dir = '/mnt/d/XT/AAA_Simulation/GY/chapter3/StarProperties/'  # Excel文件输出目录

    # 开始处理XT
    start_time = datetime.datetime.now()
    logging.info(f"开始处理，时间: {start_time}")

    process_from_csv(
        csv_file,
        input_dir,
        fits_output_dir,
        excel_output_dir
    )

    end_time = datetime.datetime.now()
    logging.info(f"处理完成，时间: {end_time}")
    logging.info(f"总耗时: {end_time - start_time}")


    # -----------------------------------------------------------------------------------------
    # 一次性读取多个文件夹
    for i in range(1, 3):
        chapter = f'chapter{i:03d}'
        csv = f'{chapter}.csv'

        # 设置参数（恢复原来的参数设置方式）
        input_dir = '/mnt/d/XT/AAA_Simulation/GaiaDr3/Parquet/'  # 768个星表Parquet文件所在的文件夹路径 ***不需要变
        fits_output_dir = '/mnt/d/XT/AAA_Simulation/temp/'  # FITS文件输出目录 ***不需要变

        csv_file = f'/mnt/d/XT/AAA_Simulation/01/Point/{csv}'  # 保持CSV格式的输入参数文件
        excel_output_dir = f'/mnt/d/XT/AAA_Simulation/GY_01/{chapter}/StarProperties/'  # Excel文件输出目录

        if not os.path.exists(excel_output_dir):
            os.makedirs(excel_output_dir)
            logging.info(f"创建目录:{excel_output_dir}")

        # 开始处理XT
        start_time = datetime.datetime.now()
        logging.info(f"开始处理，时间: {start_time}")

        process_from_csv(
            csv_file,
            input_dir,
            fits_output_dir,
            excel_output_dir
        )

        end_time = datetime.datetime.now()
        logging.info(f"处理完成，时间: {end_time}")
        logging.info(f"总耗时: {end_time - start_time}")