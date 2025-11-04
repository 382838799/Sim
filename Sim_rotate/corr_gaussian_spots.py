import numpy as np
from scipy import ndimage


# from numba import jit



# ================================================================
# 函数: 双三次插值（计算效率太慢，暂时不采用）
# ================================================================
# def cubic_weight(t, a=-0.5):
#     """
#     Compute the cubic interpolation weight for distance t.
#     Uses MATLAB's likely default a = -0.5 (Catmull-Rom spline).
#     """
#     abs_t = np.abs(t)
#     if abs_t <= 1:
#         return (a + 2) * abs_t ** 3 - (a + 3) * abs_t ** 2 + 1
#     elif abs_t <= 2:
#         return a * abs_t ** 3 - 5 * a * abs_t ** 2 + 8 * a * abs_t - 4 * a
#     else:
#         return 0
#
# def interpolate_bicubic(A, x, y, fill_value=0):
#     """
#     Perform bicubic interpolation at (x, y) in image A.
#     x, y are in world coordinates (1-based, matching MATLAB).
#     """
#     M, N = A.shape
#     i = np.floor(x)  # X-coordinate (horizontal)
#     j = np.floor(y)  # Y-coordinate (vertical)
#     dx = x - i
#     dy = y - j
#     value = 0.0
#
#     for m in range(-1, 3):
#         for n in range(-1, 3):
#             weight = cubic_weight(dx - m) * cubic_weight(dy - n)
#             # World coordinates k (row), l (col)
#             k = int(j + n)  # Row index in 1-based system
#             l = int(i + m)  # Column index in 1-based system
#             if 1 <= k <= M and 1 <= l <= N:
#                 # Convert to 0-based indexing for NumPy
#                 A_val = float(A[k - 1, l - 1])
#             else:
#                 A_val = float(fill_value)
#             value += A_val * weight
#     return value
#
#
# def imtranslate(A, translation, method='bicubic', output_view='same', fill_values=0):
#     """
#     Translate 2D image A by translation vector using bicubic interpolation.
#
#     Parameters:
#     - A: 2D NumPy array (image)
#     - translation: [TX, TY] vector (can be fractional)
#     - method: 'bicubic' (only this method is implemented)
#     - output_view: 'same' (only this option is implemented)
#     - fill_values: Scalar value for out-of-bounds pixels
#
#     Returns:
#     - B: Translated image with same size and dtype as A
#     """
#     # Validate inputs
#     if method != 'bicubic':
#         raise NotImplementedError("Only 'bicubic' method is implemented.")
#     if output_view != 'same':
#         raise NotImplementedError("Only 'same' output_view is implemented.")
#     if len(translation) != 2:
#         raise ValueError("Translation must be a 2-element vector for 2D images.")
#     if A.ndim != 2:
#         raise ValueError("Input A must be a 2D array.")
#
#     TX, TY = float(translation[0]), float(translation[1])
#     M, N = A.shape
#
#     # Check for integer translation (MATLAB optimization)
#     is_integer_translation = (TX == int(TX)) and (TY == int(TY))
#
#     if is_integer_translation:
#         # Integer translation: use direct indexing
#         TX_int = int(TX)
#         TY_int = int(TY)
#         B = np.full((M, N), fill_values, dtype=A.dtype)
#
#         # Compute row and column slices
#         if TY_int >= 0:
#             row_dst_start = TY_int
#             row_dst_end = M
#             row_src_start = 0
#             row_src_end = M - TY_int
#         else:
#             row_dst_start = 0
#             row_dst_end = M + TY_int
#             row_src_start = -TY_int
#             row_src_end = M
#
#         if TX_int >= 0:
#             col_dst_start = TX_int
#             col_dst_end = N
#             col_src_start = 0
#             col_src_end = N - TX_int
#         else:
#             col_dst_start = 0
#             col_dst_end = N + TX_int
#             col_src_start = -TX_int
#             col_src_end = N
#
#         # Copy overlapping region if valid
#         if row_dst_start < row_dst_end and col_dst_start < col_dst_end:
#             B[row_dst_start:row_dst_end, col_dst_start:col_dst_end] = \
#                 A[row_src_start:row_src_end, col_src_start:col_src_end]
#     else:
#         # Non-integer translation: use bicubic interpolation
#         B = np.zeros((M, N), dtype=np.float64)
#         for v in range(1, M + 1):  # 1-based Y (rows)
#             for u in range(1, N + 1):  # 1-based X (columns)
#                 x = u - TX  # Input X-coordinate
#                 y = v - TY  # Input Y-coordinate
#                 B[v - 1, u - 1] = interpolate_bicubic(A, x, y, fill_values)
#         # Cast to original dtype if not floating-point
#         if not np.issubdtype(A.dtype, np.floating):
#             B = B.astype(A.dtype)
#
#     return B
# --------------------------------------------------------------------------------



# ================================================================
# 函数: 实现高斯光斑偏移（双三次插值，高性能版本）
# ================================================================

# @jit(nopython=True)
# def cubic_weight(t, a=-0.5):
#     """
#     Compute the cubic interpolation weight for distance t.
#     Uses MATLAB's likely default a = -0.5 (Catmull-Rom spline).
#     """
#     abs_t = np.abs(t)
#     if abs_t <= 1:
#         return (a + 2) * abs_t ** 3 - (a + 3) * abs_t ** 2 + 1
#     elif abs_t <= 2:
#         return a * abs_t ** 3 - 5 * a * abs_t ** 2 + 8 * a * abs_t - 4 * a
#     else:
#         return 0


# @jit(nopython=True)
# def interpolate_bicubic_optimized(A, x, y, fill_value=0):
#     """
#     Perform bicubic interpolation at (x, y) in image A.
#     x, y are in world coordinates (1-based, matching MATLAB).
#     """
#     M, N = A.shape
#     i = np.floor(x)  # X-coordinate (horizontal)
#     j = np.floor(y)  # Y-coordinate (vertical)
#     dx = x - i
#     dy = y - j

#     # Pre-compute weights for reuse
#     wx = np.zeros(4)
#     wy = np.zeros(4)

#     for m in range(-1, 3):
#         wx[m + 1] = cubic_weight(dx - m)

#     for n in range(-1, 3):
#         wy[n + 1] = cubic_weight(dy - n)

#     value = 0.0
#     for n in range(-1, 3):
#         k = int(j + n)  # Row index in 1-based system
#         k_valid = 1 <= k <= M

#         for m in range(-1, 3):
#             weight = wx[m + 1] * wy[n + 1]
#             l = int(i + m)  # Column index in 1-based system

#             if k_valid and 1 <= l <= N:
#                 # Convert to 0-based indexing for NumPy
#                 A_val = A[k - 1, l - 1]
#             else:
#                 A_val = fill_value

#             value += A_val * weight

#     return value


# @jit(nopython=True)
# def imtranslate_optimized(A, translation, fill_values=0):
#     """
#     Optimized version of imtranslate function.
#     """
#     TX, TY = float(translation[0]), float(translation[1])
#     M, N = A.shape

#     # Check for integer translation
#     is_integer_translation = (TX == int(TX)) and (TY == int(TY))

#     if is_integer_translation:
#         # Integer translation: use direct indexing
#         TX_int = int(TX)
#         TY_int = int(TY)
#         B = np.full((M, N), fill_values, dtype=A.dtype)

#         # Compute row and column slices
#         if TY_int >= 0:
#             row_dst_start = TY_int
#             row_dst_end = M
#             row_src_start = 0
#             row_src_end = M - TY_int
#         else:
#             row_dst_start = 0
#             row_dst_end = M + TY_int
#             row_src_start = -TY_int
#             row_src_end = M

#         if TX_int >= 0:
#             col_dst_start = TX_int
#             col_dst_end = N
#             col_src_start = 0
#             col_src_end = N - TX_int
#         else:
#             col_dst_start = 0
#             col_dst_end = N + TX_int
#             col_src_start = -TX_int
#             col_src_end = N

#         # Copy overlapping region if valid
#         if row_dst_start < row_dst_end and col_dst_start < col_dst_end:
#             B[row_dst_start:row_dst_end, col_dst_start:col_dst_end] = \
#                 A[row_src_start:row_src_end, col_src_start:col_src_end]

#         return B
#     else:
#         # Non-integer translation: use optimized bicubic interpolation
#         B = np.zeros((M, N), dtype=A.dtype)

#         # Apply interpolation to each pixel using simple loops instead of mgrid
#         for v in range(1, M + 1):
#             y = v - TY  # Input Y-coordinate
#             for u in range(1, N + 1):
#                 x = u - TX  # Input X-coordinate
#                 B[v - 1, u - 1] = interpolate_bicubic_optimized(A, x, y, fill_values)

#         return B


# def imtranslate(A, translation, method='bicubic', output_view='same', fill_values=0):
#     """
#     Translate 2D image A by translation vector using bicubic interpolation.
#     This is a wrapper around the optimized implementation.

#     Parameters:
#     - A: 2D NumPy array (image)
#     - translation: [TX, TY] vector (can be fractional)
#     - method: 'bicubic' (only this method is implemented)
#     - output_view: 'same' (only this option is implemented)
#     - fill_values: Scalar value for out-of-bounds pixels

#     Returns:
#     - B: Translated image with same size and dtype as A
#     """
#     # Validate inputs
#     if method != 'bicubic':
#         raise NotImplementedError("Only 'bicubic' method is implemented.")
#     if output_view != 'same':
#         raise NotImplementedError("Only 'same' output_view is implemented.")
#     if len(translation) != 2:
#         raise ValueError("Translation must be a 2-element vector for 2D images.")
#     if A.ndim != 2:
#         raise ValueError("Input A must be a 2D array.")

#     # Convert A to float64 if not floating type to ensure accuracy during interpolation
#     orig_dtype = A.dtype
#     if not np.issubdtype(orig_dtype, np.floating):
#         A_float = A.astype(np.float64)
#         result = imtranslate_optimized(A_float, translation, fill_values)
#         return result.astype(orig_dtype)
#     else:
#         return imtranslate_optimized(A, translation, fill_values)

# ----------------------------------------------------------------










# ================================================================
# 函数: 实现高斯光斑偏移（双线性插值）
# ================================================================
def corr_gaussian_spot(img, filter_size, sigma, center, translation):
    # 将高斯光斑中心移动到非整数坐标位置

    # 参数:
    #   img: 原始图像
    #   filter_size: 高斯滤波器的大小
    #   sigma: 高斯滤波器的标准差
    #   center: 高斯光斑的中心位置 (行, 列)
    #   translation: 高斯光斑的平移量 (行方向平移量, 列方向平移量)

    # 返回值:
    #   img: 将一个高斯光斑移动到指定位置后的图像

    # 创建高斯滤波器
    x = np.arange(-filter_size // 2 + 1, filter_size // 2 + 1)
    y = np.arange(-filter_size // 2 + 1, filter_size // 2 + 1)
    x, y = np.meshgrid(x, y)
    gaussian_filter = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gaussian_filter /= np.sum(gaussian_filter)  # 归一化处理

    # 创建一个全零的图像来存放单个高斯光斑
    single_spot = np.zeros_like(img)

    # 将高斯滤波器放置在 single_spot 图像的中心位置
    half_size = (filter_size - 1) // 2
    single_spot[center[0] - half_size:center[0] + half_size + 1,
    center[1] - half_size:center[1] + half_size + 1] = gaussian_filter

    # 对单个高斯光斑进行平移。
    translated_spot = ndimage.shift(single_spot, translation, order = 1)  # oreder = 1 使用 双线性 插值, order = 3 双三次插值
    # translated_spot = imtranslate(single_spot, [translation[1], translation[0]])  # 双三次插值手动实现（但是性能太低了，暂不考虑使用）

    # 将平移后的光斑叠加到主图像中
    img = img + translated_spot


    return img





# GPU版本
# import cupy as cp
# from scipy import ndimage

# def corr_gaussian_spot_gpu(img, filter_size, sigma, center, translation):
#     # 将高斯光斑中心移动到非整数坐标位置

#     # 参数:
#     #   img: 原始图像
#     #   filter_size: 高斯滤波器的大小
#     #   sigma: 高斯滤波器的标准差
#     #   center: 高斯光斑的中心位置 (行, 列)
#     #   translation: 高斯光斑的平移量 (行方向平移量, 列方向平移量)

#     # 返回值:
#     #   img: 将一个高斯光斑移动到指定位置后的图像

#     # 将图像数据从CPU转移到GPU
#     img = cp.asarray(img)

#     # 创建高斯滤波器
#     x = cp.arange(-filter_size // 2 + 1, filter_size // 2 + 1)
#     y = cp.arange(-filter_size // 2 + 1, filter_size // 2 + 1)
#     x, y = cp.meshgrid(x, y)
#     gaussian_filter = cp.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
#     gaussian_filter /= cp.sum(gaussian_filter)  # 归一化处理

#     # 创建一个全零的图像来存放单个高斯光斑
#     single_spot = cp.zeros_like(img)

#     # 将高斯滤波器放置在 single_spot 图像的中心位置
#     half_size = (filter_size - 1) // 2
#     single_spot[center[0] - half_size:center[0] + half_size + 1,
#     center[1] - half_size:center[1] + half_size + 1] = gaussian_filter

#     # 对单个高斯光斑进行平移。
#     translated_spot = cp.asarray(ndimage.shift(cp.asnumpy(single_spot), translation, order=1))  # 使用双线性插值，首先转为NumPy数组，再转回GPU

#     # 将平移后的光斑叠加到主图像中
#     img = img + translated_spot

#     # 如果需要将结果转回CPU
#     img = cp.asnumpy(img)

#     return img
