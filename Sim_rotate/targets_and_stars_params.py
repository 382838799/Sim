# ================================================================
# 函数: 设置 mb 光斑的各项参数
# ================================================================
def set_targets_generate_params(magRange):
    prpts = {
        'minmagnitude': magRange[0],
        'maxmagnitude': magRange[1],

        # radius
        'minnodesize': 1,
        'maxnodesize': 1.5,

        # sigma
        'minblurlevel': 0.001,
        'maxblurlevel': 0.29,

        # 灰度值上限与下限
        'lammaxrate': 0.085,
        'lamrate': 0.05,

        # scaling low
        'scalinglow': 10
    }
    return prpts









# ================================================================
# 函数: 设置 hx 光斑的各项参数
# ================================================================
def set_stars_generate_params_double_log(magRange):
    prpts = {
        'minmagnitude': magRange[0],
        'maxmagnitude': magRange[1],

        # radius
        'minnodesize': 5,
        'maxnodesize': 50,

        # sigma
        'minblurlevel': 0.5,
        'maxblurlevel': 2.5,

        # 灰度值上限与下限
        'lammaxrate': 1,
        'lamrate': 0.05,

        # scaling low
        'scalinglow': 15
    }
    return prpts
