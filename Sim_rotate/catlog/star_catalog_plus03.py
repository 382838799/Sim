# 优化后的第三版，读取.parquet 而非 .csv，保持其他逻辑不变并解决冻结问题
import pandas as pd
import numpy as np
from astropy.io import fits
import re
import healpy as hp
import os
import logging
import datetime
import time
import gc
from multiprocessing import Pool, cpu_count
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 清理文件名，移除非法字符
def sanitize_filename(filename):
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='ignore')
    return re.sub(r'[<>:"/\\|?*]', '-', str(filename))


# 1. 读取大范围数据并保存为 FITS 文件 - 修改为读取Parquet文件并改进内存管理
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
        logging.info(f"Processing file: {pixel_file}")

        try:
            # 读取Parquet文件 - 添加内存管理
            df = pd.read_parquet(pixel_file_path, columns=cols)
            
            # 星等筛选
            mask = (df['phot_g_mean_mag'] >= min_mag) & (df['phot_g_mean_mag'] <= max_mag)
            if not mask.any():
                del df, mask
                gc.collect()
                continue

            # 只提取需要的数据
            filtered_df = df.loc[mask, cols]
            
            # 添加筛选后的数据
            for col in cols:
                all_data[col].append(filtered_df[col].values)
            total_rows += len(filtered_df)
            
            # 显式删除不再需要的数据帧以释放内存
            del df, filtered_df, mask
            gc.collect()
            
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
        
        # 清理内存
        del all_data, structured_data, hdu
        gc.collect()
        
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
        9.0,  # 时间间隔（儒略年）
        -0.1666765089229757, 0.8891312155913055, 0.3854482126946778,  # 地球的日心坐标（天文单位 AU）
        -0.1695049303130315, 0.904219351075994, 0.3919890862502201,  # 地球的日心方向（单位向量）
        0.00000002,  # 太阳引力辐射常数 * 2 / （太阳-地球距离）
        -0.000099155, -0.00001731, -0.0000075,  # 地球的日心速度（以光速单位 c 表示）
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

    # 清理内存
    del q, em, p, pn, p1, p2
    gc.collect()

    return ra_corrected, dec_corrected


# 3. 读取 FITS 文件并将数据转换
def read_fits_to_df(fits_file):
    """读取FITS文件并提取关键字段"""
    try:
        with fits.open(fits_file) as hdulist:
            data = hdulist[1].data
            # 复制数据到新数组而不是引用
            ra = np.copy(data['ra'])
            dec = np.copy(data['dec'])
            phot_g_mean_mag = np.copy(data['phot_g_mean_mag'])
            pmra = np.copy(data['pmra'])
            pmdec = np.copy(data['pmdec'])
            parallax = np.copy(data['parallax'])
            radial_velocity = np.copy(data['radial_velocity'])
        return ra, dec, phot_g_mean_mag, pmra, pmdec, parallax, radial_velocity
    except Exception as e:
        logging.error(f"Error reading FITS file {fits_file}: {e}")
        return None, None, None, None, None, None, None


# 4. 坐标系转换函数（例如相面坐标转换）
def to_camera_and_pixel_coords(ra, dec, f, W, H, ra_cam, dec_cam, pixel_scale):
    """将天球坐标转换为相机像素坐标"""
    try:
        # 将度转换为弧度
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        ra_cam_rad = np.radians(ra_cam)
        dec_cam_rad = np.radians(dec_cam)

        # 计算方向余弦
        cos_dec = np.cos(dec_rad)
        sin_dec = np.sin(dec_rad)
        cos_dec_cam = np.cos(dec_cam_rad)
        sin_dec_cam = np.sin(dec_cam_rad)
        cos_ra_diff = np.cos(ra_rad - ra_cam_rad)
        sin_ra_diff = np.sin(ra_rad - ra_cam_rad)

        # 计算投影坐标
        x_proj = cos_dec * sin_ra_diff
        y_proj = sin_dec * cos_dec_cam - sin_dec_cam * cos_dec * cos_ra_diff
        z_proj = sin_dec * sin_dec_cam + cos_dec * cos_dec_cam * cos_ra_diff

        # 计算相机坐标（只处理在相机前方的星点，即z_proj > 0）
        mask = z_proj > 0

        # 初始化结果数组
        x_pixel = np.zeros_like(ra)
        y_pixel = np.zeros_like(ra)
        valid = np.zeros_like(ra, dtype=bool)

        if np.any(mask):
            # 计算相机坐标
            x_cam = -f * x_proj[mask] / z_proj[mask]
            y_cam = -f * y_proj[mask] / z_proj[mask]

            # 转换为像素坐标
            x_pixel[mask] = x_cam / 0.01 + W / 2
            y_pixel[mask] = y_cam / 0.01 + H / 2

            # 确定哪些点在图像内
            in_frame = (x_pixel[mask] >= 0) & (x_pixel[mask] < W) & (y_pixel[mask] >= 0) & (y_pixel[mask] < H)

            # 获取实际有效的索引
            valid_indices = np.where(mask)[0][in_frame]
            valid[valid_indices] = True

        # 清理内存
        del x_proj, y_proj, z_proj, mask
        gc.collect()

        return x_pixel[valid], y_pixel[valid], valid
    except Exception as e:
        logging.error(f"Error in coordinate transformation: {e}")
        # 返回空结果
        return np.array([]), np.array([]), np.zeros_like(ra, dtype=bool)


# 5. 将数据保存为 Excel 文件
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
            'x_pixel': x,
            'y_pixel': y
        })
        output_path = os.path.join(output_dir, filename)
        df.to_excel(output_path, index=False)
        logging.info(f"数据已保存至 {output_path}, 包含 {len(df)} 个星点")
        
        # 清理内存
        del df
        gc.collect()
        
        return output_path
    except Exception as e:
        logging.error(f"Error saving Excel file {filename}: {e}")
        return None


# 6. 从 CSV 文件读取数据并生成多组文件
def process_single_row(row, data, f, W, H, pixel_scale, min_mag, max_mag, input_dir, fits_output_dir, excel_output_dir):
    """处理CSV文件中的单行数据，添加超时检测和错误处理"""
    start_time = time.time()
    timeout_seconds = 1800  # 30分钟超时
    
    try:
        row_id = row
        # 确保使用str()处理可能的非字符串值
        title = str(data.iloc[row, 0])  # 获取目标文件名
        ra_cam = (float(data.iloc[row, 1]) + 360) % 360  # 从CSV中获取ra_cam
        dec_cam = float(data.iloc[row, 2])  # 从CSV中获取dec_cam
        radius_deg = 20
        nside = 8

        logging.info(f"处理行 {row_id}: 目标 {title}, RA = {ra_cam}, DEC = {dec_cam}")

        # 检查超时
        if time.time() - start_time > timeout_seconds:
            logging.error(f"行 {row_id}: 处理超时")
            return

        # 1. 读取大范围数据并保存为 FITS 文件
        fits_file = filter_data_by_fov_and_mag(input_dir, ra_cam, dec_cam, radius_deg, min_mag, max_mag, nside,
                                               fits_output_dir, title)

        if fits_file is None:
            logging.warning(f"行 {row_id}: 无法生成FITS文件")
            return

        # 检查超时
        if time.time() - start_time > timeout_seconds:
            logging.error(f"行 {row_id}: 处理超时")
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

        # 检查超时
        if time.time() - start_time > timeout_seconds:
            logging.error(f"行 {row_id}: 处理超时")
            return

        # 4. 根据光行差修正计算新的赤经赤纬
        ra_new, dec_new = apply_aberration_correction(ra, dec, pmra, pmdec, parallax, radial_velocity)

        # 检查超时
        if time.time() - start_time > timeout_seconds:
            logging.error(f"行 {row_id}: 处理超时")
            return

        # 5. 转换坐标系并筛选有效点
        x_pixel, y_pixel, valid = to_camera_and_pixel_coords(ra_new, dec_new, f, W, H, ra_cam, dec_cam, pixel_scale)

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

        # 清理内存
        del ra, dec, pmra, pmdec, parallax, radial_velocity, ra_new, dec_new, valid
        gc.collect()

        # 检查超时
        if time.time() - start_time > timeout_seconds:
            logging.error(f"行 {row_id}: 处理超时")
            return

        # 7. 保存结果
        output_filename = f"{sanitize_filename(title)}.xlsx"
        save_to_excel(filtered_ra, filtered_dec, filtered_mag, filtered_pmra, filtered_pmdec,
                      filtered_parallax, filtered_radial_velocity, x_pixel, y_pixel,
                      excel_output_dir, output_filename)

        logging.info(f"行 {row_id}: 处理完成，耗时 {time.time() - start_time:.2f} 秒")

    except Exception as e:
        logging.error(f"处理行 {row_id} 时发生错误: {e}")


# 7. 主函数：使用多进程处理 CSV 文件中的每一行
def process_from_csv(csv_file, input_dir, fits_output_dir, excel_output_dir, start_row=0, end_row=None, workers=None):
    """处理CSV文件中的所有行，改进的批处理方式"""
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
    max_mag = 13.0  # 星等最大值

    # 设置进程数 - 限制进程数以防止资源耗尽
    if workers is None:
        workers = min(4, max(1, cpu_count() // 4))  # 限制最多4个进程

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

    # 小批量处理，每次处理少量行
    batch_size = 3  # 减小批处理大小
    for batch_start in range(start_row, end_row, batch_size):
        batch_end = min(batch_start + batch_size, end_row)
        logging.info(f"处理批次：行 {batch_start} 到 {batch_end - 1}")
        
        try:
            # 多进程处理该批次
            with Pool(workers) as pool:
                pool.map(process_func, range(batch_start, batch_end))
            
            # 强制清理内存
            gc.collect()
            
            logging.info(f"完成批次：行 {batch_start} 到 {batch_end - 1}")
            
            # 在批次之间添加短暂休息，让系统资源恢复
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"批处理错误 {batch_start}-{batch_end-1}: {e}")
            
            # 如果批处理失败，尝试逐行处理
            logging.info(f"尝试逐行处理批次 {batch_start}-{batch_end-1}")
            for row in range(batch_start, batch_end):
                try:
                    process_func(row)
                    # 每行之间也添加短暂休息
                    time.sleep(1)
                except Exception as e_row:
                    logging.error(f"处理行 {row} 时出错: {e_row}")


# 主程序入口
if __name__ == '__main__':
    # 设置参数
    csv_file = '/mnt/d/Simulation/5.0000000.csv'  # CSV 文件路径，即输入的相机指向
    input_dir = '/mnt/d/gaia/data_parquet/'  # Parquet 文件所在的文件夹路径
    fits_output_dir = '/mnt/d/Simulation/output/chapter24/xydata_parquet'  # FITS文件输出目录
    excel_output_dir = '/mnt/d/Simulation/output/chapter24/xydata_parquet'  # Excel文件输出目录

    # 开始处理
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