import numpy as np


# ================================================================
# 函数: 运动模糊函数（跟matlab中的函数已完全对齐，无需修改）
# ================================================================
def motion_blur_kernel(Len, Direc):
    p2 = Len
    p3 = Direc

    length = max(1, p2)  # 避免覆盖内置的 len() 函数
    half = (length - 1) / 2  # rotate half length around center
    phi = np.mod(p3, 180) / 180 * np.pi

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    xsign = np.sign(cosphi)
    linewdt = 1

    # define mesh for the half matrix, eps takes care of the right size for 0 & 90 rotation
    sx = np.fix(half * cosphi + linewdt * xsign - length * np.finfo(float).eps)
    sy = np.fix(half * sinphi + linewdt - length * np.finfo(float).eps)
    x, y = np.meshgrid(np.arange(0, sx + xsign, xsign), np.arange(0, sy + 1))

    # define shortest distance from a pixel to the rotated line
    dist2line = y * cosphi - x * sinphi  # distance perpendicular to the line

    rad = np.sqrt(x ** 2 + y ** 2)
    # find points beyond the line's end-point but within the line width
    lastpix = np.where((rad >= half) & (np.abs(dist2line) <= linewdt))
    # distance to the line's end-point parallel to the line
    x2lastpix = half - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi)

    dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix ** 2)
    dist2line = linewdt + np.finfo(float).eps - np.abs(dist2line)
    dist2line[dist2line < 0] = 0  # zero out anything beyond line width

    # unfold half-matrix to the full size
    h = np.rot90(dist2line, 2)

    # 扩展 h 的大小，确保它足够容纳 dist2line 的内容
    h_expanded = np.zeros((len(h) + len(dist2line) - 1, len(h[0]) + len(dist2line[0]) - 1))

    # 将原矩阵 h 放置到扩展后的 h_expanded 的左上角
    h_expanded[:len(h), :len(h[0])] = h

    # 将 dist2line 放入扩展后的矩阵 h_expanded 的右下角
    h_expanded[len(h)-1:len(h)-1+len(dist2line), len(h[0])-1:len(h[0])-1+len(dist2line[0])] = dist2line

    # 对 h_expanded 进行归一化
    h_expanded = h_expanded / (np.sum(h_expanded) + np.finfo(float).eps * length * length)

    if cosphi > 0:
        h_expanded = np.flipud(h_expanded)

    return h_expanded
