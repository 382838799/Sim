'''
#这段代码用于将奇偶数列区分开以便于滚动曝光
import pandas as pd

# 读取 Excel 文件
excel_file = '5.0000.xlsx'  # 替换为实际文件路径
df = pd.read_excel(excel_file)

# 提取奇数行（index从0开始，奇数行的index是0, 2, 4, ...）
odd_rows = df.iloc[::2]  # 从第1行开始，步长为2，选择奇数行
even_rows = df.iloc[1::2]  # 从第2行开始，步长为2，选择偶数行

# 保存为CSV文件
odd_rows.to_csv('odd_rows.csv', index=False)
even_rows.to_csv('even_rows.csv', index=False)

print("奇数行和偶数行已分别保存为 odd_rows.csv 和 even_rows.csv")
'''
# 这段代码用于读取完整星表并计算位置，输出数据
import pandas as pd
import numpy as np
from astroquery.gaia import Gaia
from astropy.table import QTable
from astropy.io import fits
import re
import healpy as hp
import time
import os
import logging
import datetime
from multiprocessing import Pool, cpu_count
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 清理文件名，移除非法字符
def sanitize_filename(filename):
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='ignore')
    return re.sub(r'[<>:"/\\|?*]', '-', filename)


# 1. 读取大范围数据并保存为 FITS 文件
def filter_data_by_fov_and_mag(path, ra_cam, dec_cam, radius_deg, min_mag, max_mag, nside, title=None):
    """逐个像素块读取数据，筛选符合条件的天区数据，结合星等范围"""
    filtered_data = []

    # 目标天区的角度（以弧度表示）
    target_ra_rad = np.radians(ra_cam)  # 使用 numpy 的矢量化操作，自动转换整个数组
    target_dec_rad = np.radians(dec_cam)  # 此处纬度是余纬
    radius_rad = np.radians(radius_deg)  # 视场半径，转换为弧度
    # vec = hp.ang2vec(np.radians(90 - dec_cam / 2), target_ra_rad)
    vec = hp.ang2vec(np.radians(90 - dec_cam), target_ra_rad)

    try:
        # 使用 Healpix 的 query_disc 查找视场内的像素块索引
        pixel_indices = hp.query_disc(nside=nside, vec=vec, radius=radius_rad, inclusive=True, nest=True)
    #   logging.info(f"Pixel indices found: {pixel_indices[:20]}...")  # 输出前20个索引以帮助调试
    except Exception as e:
        logging.error(f"Error in query_disc: {e}")
        pixel_indices = []

    # logging.info(f"Number of pixel indices found: {len(pixel_indices)}")  # 输出查询到的像素数量

    if len(pixel_indices) == 0:
        logging.warning("No pixel indices found for the given coordinates and radius.")

    # 获取符合条件的像素块文件
    pixel_files = [f for f in os.listdir(path) if f.startswith('pixel_') and f.endswith('.csv.gz')]
    # logging.info(f"Files in the directory: {pixel_files}")  # 输出当前文件夹中的文件列表

    relevant_files = []
    for pixel_file in pixel_files:
        # 提取文件名中的索引（像素块编号）
        pixel_index = int(pixel_file.split('_')[1].split('.')[0])
        # logging.debug(f"Checking pixel index: {pixel_index}")

        # 检查文件的像素块索引是否在查询返回的索引列表中
        if pixel_index in pixel_indices:
            relevant_files.append(pixel_file)

    # logging.info(f"Found {len(relevant_files)} relevant pixel files.")  # 输出符合条件的文件数量

    for pixel_file in relevant_files:
        pixel_file_path = os.path.join(path, pixel_file)
        #  logging.info(f"Reading file: {pixel_file_path}")

        # 只读取所需的列，减少内存消耗
        cols = ['ra', 'dec', 'phot_g_mean_mag', 'pmra', 'pmdec', 'parallax', 'radial_velocity']
        try:
            pixel_data = pd.read_csv(pixel_file_path, compression='gzip', usecols=cols)
        except ValueError as e:
            logging.warning(f"Missing columns in {pixel_file_path}: {e}")
            continue

        # 筛选符合星等范围的数据
        pixel_data_filtered = pixel_data[
            (pixel_data['phot_g_mean_mag'] >= min_mag) & (pixel_data['phot_g_mean_mag'] <= max_mag)]

        # 检查筛选后数据是否为空
        if pixel_data_filtered.empty:
            logging.warning(f"No data in {pixel_file} after applying magnitude filter.")
            continue

        # 将筛选后的数据添加到最终结果中
        filtered_data.append(pixel_data_filtered)

    if not filtered_data:
        logging.warning("No data passed the filters.")

    # 合并所有符合条件的数据
    if filtered_data:
        final_filtered_data = pd.concat(filtered_data, ignore_index=True)
    #    logging.info(f"Total {len(final_filtered_data)} stars after filtering.")
    else:
        final_filtered_data = pd.DataFrame()

    # 保存为 FITS 文件
    if not final_filtered_data.empty:
        dtype = [('ra', 'f8'), ('dec', 'f8'), ('phot_g_mean_mag', 'f8'),
                 ('pmra', 'f8'), ('pmdec', 'f8'), ('parallax', 'f8'),
                 ('radial_velocity', 'f8')]

        structured_data = np.core.records.fromarrays(
            [final_filtered_data[col].values for col in final_filtered_data.columns],
            dtype=dtype)

        hdu = fits.BinTableHDU(structured_data)
        output_filename = os.path.join(path, f"gaia_stars_{title}.fits")  # 保存到指定路径
        hdu.writeto(output_filename, overwrite=True)
        logging.info(f"Saved to FITS file: {output_filename}")

        # 返回生成的 FITS 文件路径
        return output_filename


# 2. 根据光行差修正计算小范围赤经赤纬
def apply_aberration_correction(ra, dec, pmra, pmdec, parallax, radial_velocity):
    DAS2R = np.pi / (180.0 * 3600000.0)  # 毫角秒到弧度的转换常数
    VF = 0.21094502  # 速度常数，具体值可以根据需要调整
    SRS = 1.97412574336e-8  # (AU/yr)
    c = 4.74057e-3  # 光速 (AU/yr)

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
    ra = ra * np.pi / 180
    dec = dec * np.pi / 180

    # 假设 ra 和 dec 是二维数组，每行代表一个天体的赤经和赤纬
    num_stars = len(ra)  # 天体的数量
    q = np.zeros((num_stars, 3))  # 初始化笛卡尔坐标数组
    q[:, 0] = np.cos(ra) * np.cos(dec)  # 计算 x 坐标
    q[:, 1] = np.sin(ra) * np.cos(dec)  # 计算 y 坐标
    q[:, 2] = np.sin(dec)  # 计算 z 坐标

    # 计算空间运动（radians per year）
    pxr = px_radians
    w = VF * rv_in_au_per_year * pxr
    em = np.zeros_like(q)  # 初始化空间运动数组
    em[:, 0] = (-pr_radians_per_year * q[:, 1]) - (pd_radians_per_year * np.cos(ra) * np.sin(dec)) + (w * q[:, 0])
    em[:, 1] = (pr_radians_per_year * q[:, 0]) - (pd_radians_per_year * np.sin(ra) * np.sin(dec)) + (w * q[:, 1])
    em[:, 2] = (pd_radians_per_year * np.cos(dec)) + (w * q[:, 2])

    # 计算地心方向
    p = np.zeros_like(q)  # 初始化地心方向数组
    for i in range(3):
        p[:, i] = q[:, i] + (pmt * em[:, i]) - (pxr * eb[i])

    # 单位化
    norm = np.linalg.norm(p, axis=1)
    pn = p / norm[:, None]

    # 光线偏转修正（限制在太阳盘内）
    pde = np.dot(pn, ehn)
    pdep1 = 1.0 + pde
    w = gr2e / np.maximum(pdep1, 1.0e-5)
    p1 = np.zeros_like(pn)
    for i in range(3):
        p1[:, i] = pn[:, i] + (w * (ehn[i] - pde * pn[:, i]))

    # 周年光行差修正
    p1dv = np.dot(p1, abv)
    p1dvp1 = p1dv + 1.0
    w = 1.0 + p1dv / (ab1 + 1.0)
    p2 = np.zeros_like(p1)
    for i in range(3):
        p2[:, i] = (ab1 * p1[:, i] + w * abv[i]) / p1dvp1

    # 将笛卡尔坐标转换回天球坐标（赤经和赤纬）
    p2_norm = np.linalg.norm(p2, axis=1)
    p2 = p2 / p2_norm[:, None]  # 归一化

    ra_corrected = np.arctan2(p2[:, 1], p2[:, 0])
    ra_corrected = (np.degrees(ra_corrected) + 360) % 360  # 计算赤经
    dec_corrected = np.arcsin(p2[:, 2])
    dec_corrected = np.degrees(dec_corrected)  # 计算赤纬

    return ra_corrected, dec_corrected


# 3. 读取 FITS 文件并将数据转换为 DataFrame
def read_fits_to_df(fits_file):
    with fits.open(fits_file) as hdulist:
        data = hdulist[1].data
    return data['ra'], data['dec'], data['phot_g_mean_mag'], data['pmra'], data['pmdec'], data['parallax'], data[
        'radial_velocity']


# 4. 坐标系转换函数（例如相面坐标转换）
def to_camera_and_pixel_coords(ra, dec, f, W, H, ra_cam, dec_cam, pixel_scale):
    ra = np.radians(ra)
    dec = np.radians(dec)
    ra_cam = np.radians(ra_cam)
    dec_cam = np.radians(dec_cam)

    pixel_scale_rad = pixel_scale * (1 / 3600) * np.pi / 180  # arcsec/pixel -> radians/pixel
    x_proj = np.cos(dec) * np.sin(ra - ra_cam)
    y_proj = np.sin(dec) * np.cos(dec_cam) - np.sin(dec_cam) * np.cos(dec) * np.cos(ra - ra_cam)
    z_proj = np.sin(dec) * np.sin(dec_cam) + np.cos(dec) * np.cos(dec_cam) * np.cos(ra - ra_cam)

    x_cam = -f * x_proj / z_proj
    y_cam = -f * y_proj / z_proj

    x_pixel = x_cam / 0.01 + W / 2
    y_pixel = y_cam / 0.01 + H / 2

    valid = (x_pixel >= 0) & (x_pixel <= W) & (y_pixel >= 0) & (y_pixel <= H)
    return x_pixel[valid], y_pixel[valid], valid


# 5. 将 DataFrame 保存为 Excel 文件
def save_to_excel(ra, dec, phot_g_mean_mag, pmra, pmdec, parallax, radial_velocity, x, y,
                  filename='filtered_stars.xlsx'):
    filename = sanitize_filename(filename)
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
    output_path = os.path.join(path, filename)  # 保存到指定路径
    df.to_excel(output_path, index=False)
    logging.info(f"数据已保存至 {output_path}")


# 6. 从 CSV 文件读取数据并生成多组文件
def process_single_row(row, data, f, W, H, pixel_scale, min_mag, max_mag):
    title = str(data.iloc[row, 0])  # 获取目标文件名
    ra_cam = (data.iloc[row, 1] + 360) % 360  # 从CSV中获取ra_cam
    dec_cam = data.iloc[row, 2]  # 从CSV中获取dec_cam
    radius_deg = 15
    nside = 8

    # 确保 ra_cam 和 dec_cam 是浮动类型
    ra_cam = float(ra_cam)
    dec_cam = float(dec_cam)

    # 1. 读取大范围数据并保存为 FITS 文件
    fits_file = filter_data_by_fov_and_mag(path, ra_cam, dec_cam, radius_deg, min_mag, max_mag, nside, title)

    # 2. 读取 FITS 文件并转换为 DataFrame
    ra, dec, phot_g_mean_mag, pmra, pmdec, parallax, radial_velocity = read_fits_to_df(fits_file)

    # 3. 将 pmra 和 pmdec 中为空的地方替换为 0
    pmra = np.nan_to_num(pmra, nan=0)
    pmdec = np.nan_to_num(pmdec, nan=0)
    parallax = np.nan_to_num(parallax, nan=0.00000001)
    radial_velocity = np.nan_to_num(radial_velocity, nan=0)

    # 4. 根据光行差修正计算新的赤经赤纬
    ra_new, dec_new = apply_aberration_correction(ra, dec, pmra, pmdec, parallax, radial_velocity)

    # 5. 筛选小范围内的恒星
    # ra_min, ra_max = ra_cam - 0.8, ra_cam + 0.8
    # dec_min, dec_max = dec_cam - 0.8, dec_cam + 0.8
    valid_indices_small_range = (ra_new <= 360)

    # 6. 转换坐标并计算星等强度
    ra_filtered = ra_new[valid_indices_small_range]
    dec_filtered = dec_new[valid_indices_small_range]
    phot_g_mean_mag_filtered = phot_g_mean_mag[valid_indices_small_range]
    pmra_filtered = pmra[valid_indices_small_range]
    pmdec_filtered = pmdec[valid_indices_small_range]
    parallax_filtered = parallax[valid_indices_small_range]
    radial_velocity_filtered = radial_velocity[valid_indices_small_range]

    # 7. 转换坐标系并筛选有效点
    x_pixel, y_pixel, valid = to_camera_and_pixel_coords(ra_filtered, dec_filtered, f, W, H, ra_cam, dec_cam,
                                                         pixel_scale)

    # 8. 重新筛选对应的星等
    filtered_ra = ra_filtered[valid]
    filtered_dec = dec_filtered[valid]
    filtered_mag = phot_g_mean_mag_filtered[valid]
    filtered_pmra = pmra_filtered[valid]
    filtered_pmdec = pmdec_filtered[valid]
    filtered_parallax = parallax_filtered[valid]
    filtered_radial_velocity = radial_velocity_filtered[valid]

    # 9. 保存结果
    output_filename = f"{sanitize_filename(title)}.xlsx"  # 使用 sanitize_filename 清理文件名

    save_to_excel(filtered_ra, filtered_dec, filtered_mag, filtered_pmra, filtered_pmdec, filtered_parallax,
                  filtered_radial_velocity, x_pixel, y_pixel, output_filename)


# 7. 主函数：使用多进程处理 CSV 文件中的每一行
def process_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    f = 418.3870309  # 相机焦距（mm）
    W, H = 8900, 8900  # 图像大小（像素）
    pixel_scale = 3.23  # 像素角度（arcsec/pixel）
    min_mag = 0.0  # 星等最小值
    max_mag = 13.0  # 星等最大值

    num_workers = cpu_count() // 4 + 2  # 获取 CPU 核心数
    with Pool(num_workers) as pool:
        pool.starmap(process_single_row,
                     [(i, data, f, W, H, pixel_scale, min_mag, max_mag) for i in range(0, data.shape[0])])


# 调用处理
if __name__ == '__main__':
    start = time.time()
    csv_file = '/mnt/d/Simulation/5.0000000.csv'  # CSV 文件路径，即输入的相机指向
    path = '/mnt/d/gaia/gaia_csv/'  # 768个星表文件所在的文件夹路径
    process_from_csv(csv_file)

    end = time.time()
    exetime = end-start
    print(f"程序执行时间: {exetime}秒")
