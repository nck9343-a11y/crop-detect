"""
田块尺度病害空间格局分析模块
===============================
输入：一组带坐标的检测结果（每个点可附带一个数值，如病斑数或严重度）
输出：该分布属于聚集 / 随机 / 均匀，以及显著的热点位置

四个统计量各自回答一个不同的问题：
  1. Clark-Evans 最近邻指数  -> 整体是聚集还是分散？（一个数）
  2. Ripley's L 函数         -> 在什么空间尺度上聚集？（一条曲线）
  3. Moran's I               -> 严重度高的地方是否挨在一起？（空间自相关）
  4. Getis-Ord Gi*           -> 具体哪几个位置是显著热点？（逐点）

所有显著性检验均采用蒙特卡洛置换检验，不依赖理论分布假设，
边界效应由模拟自动吸收，这一点比套用理论公式更稳妥。
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------- 工具

def _dist_matrix(coords):
    """返回 n x n 的两点距离矩阵。"""
    return squareform(pdist(np.asarray(coords, dtype=float)))


def _csr_sample(n, bounds, rng):
    """在给定矩形范围内生成 n 个完全随机分布(CSR)的点，用作零假设基准。"""
    xmin, ymin, xmax, ymax = bounds
    return np.column_stack([
        rng.uniform(xmin, xmax, n),
        rng.uniform(ymin, ymax, n),
    ])


def build_weights(coords, threshold=None, k=None, row_standardize=True):
    """
    构建空间权重矩阵 W —— 定义"谁和谁算邻居"。

    threshold: 距离阈值，两点距离小于它就算邻居（距离带法）
    k:         每个点取最近的 k 个作为邻居（K 近邻法）
    二者给其一。距离带法更贴合"病害在一定半径内传播"的物理意义，
    K 近邻法则保证每个点邻居数相同，点密度不均时更稳。
    """
    d = _dist_matrix(coords)
    n = len(coords)
    w = np.zeros((n, n))

    if threshold is not None:
        w = (d <= threshold).astype(float)
    elif k is not None:
        for i in range(n):
            order = np.argsort(d[i])
            w[i, order[1:k + 1]] = 1.0   # 跳过自身
    else:
        raise ValueError('threshold 与 k 必须给定其一')

    np.fill_diagonal(w, 0.0)

    if row_standardize:
        s = w.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0                  # 孤立点保持全 0，避免除零
        w = w / s
    return w


# ---------------------------------------------------------------- 1. 最近邻指数

def clark_evans(coords, bounds):
    """
    Clark-Evans 最近邻指数 R。

    R = 实测平均最近邻距离 / 随机分布下的期望值
      R < 1  聚集
      R ≈ 1  随机
      R > 1  均匀（相互排斥）

    返回 (R, z 值, p 值)。z 检验用的是 Donnelly 校正后的标准误。
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    xmin, ymin, xmax, ymax = bounds
    area = (xmax - xmin) * (ymax - ymin)

    d = _dist_matrix(coords)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    r_obs = nn.mean()

    density = n / area
    r_exp = 0.5 / np.sqrt(density)
    se = 0.26136 / np.sqrt(n * density)

    R = r_obs / r_exp
    z = (r_obs - r_exp) / se
    # 双侧 p 值，正态近似
    from math import erfc
    p = erfc(abs(z) / np.sqrt(2))
    return R, z, p


# ---------------------------------------------------------------- 2. Ripley's L

def ripley_l(coords, bounds, radii, n_sim=99, seed=0):
    """
    Ripley's L 函数（K 函数的方差稳定变换）。

    L(r) - r > 0  该尺度上聚集
    L(r) - r < 0  该尺度上均匀
    L(r) - r = 0  随机

    这是唯一能回答"在多大尺度上聚集"的统计量。
    比如 L 曲线在 r=2m 处峰值最高，说明病害斑块的特征尺寸约 2 米——
    这个数字可以直接换算成防治时的处理单元大小。

    用 n_sim 次 CSR 模拟构造置信包络，实测曲线跑出包络才算显著。
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    xmin, ymin, xmax, ymax = bounds
    area = (xmax - xmin) * (ymax - ymin)
    rng = np.random.default_rng(seed)

    def _l_curve(pts):
        d = _dist_matrix(pts)
        np.fill_diagonal(d, np.inf)
        m = len(pts)
        out = np.empty(len(radii))
        for i, r in enumerate(radii):
            cnt = (d <= r).sum()
            K = area * cnt / (m * m)
            out[i] = np.sqrt(K / np.pi)
        return out

    obs = _l_curve(coords)
    sims = np.array([_l_curve(_csr_sample(n, bounds, rng)) for _ in range(n_sim)])
    lo = np.percentile(sims, 2.5, axis=0)
    hi = np.percentile(sims, 97.5, axis=0)
    return obs, lo, hi


# ---------------------------------------------------------------- 3. Moran's I

def morans_i(coords, values, threshold=None, k=8, n_perm=999, seed=0):
    """
    Moran's I —— 数值型空间自相关。

    与前两个的区别：前两个只看"点在哪"，Moran's I 还看"点上的数值"。
    也就是回答：严重度高的植株是否倾向于挨在一起？

    I > 0  正相关（高值聚集、低值聚集）
    I ≈ E[I] = -1/(n-1)  无空间结构
    I < 0  负相关（高低相间，棋盘状）

    显著性用置换检验：固定坐标、随机打乱数值 n_perm 次，
    看实测 I 在模拟分布中的位置。
    """
    coords = np.asarray(coords, dtype=float)
    x = np.asarray(values, dtype=float)
    n = len(x)
    w = build_weights(coords, threshold=threshold, k=k)

    def _I(vals):
        z = vals - vals.mean()
        denom = (z ** 2).sum()
        if denom == 0:
            return 0.0
        num = (w * np.outer(z, z)).sum()
        s0 = w.sum()
        return (n / s0) * (num / denom)

    obs = _I(x)
    rng = np.random.default_rng(seed)
    sims = np.array([_I(rng.permutation(x)) for _ in range(n_perm)])

    expected = -1.0 / (n - 1)
    # 双侧 p 值：模拟值比实测更极端的比例
    p = (np.sum(np.abs(sims - expected) >= abs(obs - expected)) + 1) / (n_perm + 1)
    return obs, expected, p


# ---------------------------------------------------------------- 4. Getis-Ord Gi*

def getis_ord(coords, values, threshold=None, k=8):
    """
    Getis-Ord Gi* 局部热点统计量。

    前三个统计量给的是整体判断，Gi* 是逐点的：
    它告诉你具体哪几株是显著的高值聚集中心。

    返回每个点的 z 值：
      z > 1.96   显著热点（99% 置信用 2.58）
      z < -1.96  显著冷点
    输出可直接渲染成热力图，也可作为处方图的分区依据。
    """
    coords = np.asarray(coords, dtype=float)
    x = np.asarray(values, dtype=float)
    n = len(x)
    # Gi* 使用二值权重且包含自身
    w = build_weights(coords, threshold=threshold, k=k, row_standardize=False)
    np.fill_diagonal(w, 1.0)

    xbar = x.mean()
    s = np.sqrt((x ** 2).sum() / n - xbar ** 2)

    z_scores = np.empty(n)
    for i in range(n):
        wi = w[i]
        sw = wi.sum()
        num = (wi * x).sum() - xbar * sw
        den = s * np.sqrt((n * (wi ** 2).sum() - sw ** 2) / (n - 1))
        z_scores[i] = num / den if den > 0 else 0.0
    return z_scores


# ---------------------------------------------------------------- 汇总

def describe_pattern(coords, values=None, bounds=None, k=8, verbose=True):
    """
    一次性跑完全部统计量，并给出格局判定。
    values 为空时按等权处理（只分析点位置，不分析严重度）。
    """
    coords = np.asarray(coords, dtype=float)
    if bounds is None:
        bounds = (coords[:, 0].min(), coords[:, 1].min(),
                  coords[:, 0].max(), coords[:, 1].max())
    if values is None:
        values = np.ones(len(coords))

    R, z_ce, p_ce = clark_evans(coords, bounds)
    I, E_I, p_i = morans_i(coords, values, k=k)
    gi = getis_ord(coords, values, k=k)
    n_hot = int((gi > 1.96).sum())

    if R < 1 and p_ce < 0.05:
        verdict = '聚集分布'
    elif R > 1 and p_ce < 0.05:
        verdict = '均匀分布'
    else:
        verdict = '随机分布'

    if verbose:
        print(f'  点数 {len(coords)}')
        print(f'  Clark-Evans R = {R:.3f}  (z={z_ce:+.2f}, p={p_ce:.4f})  -> {verdict}')
        print(f'  Moran\'s I     = {I:+.3f}  (E={E_I:+.3f}, p={p_i:.4f})')
        print(f'  显著热点数     = {n_hot}')

    return {'R': R, 'p_ce': p_ce, 'morans_i': I, 'p_i': p_i,
            'gi_star': gi, 'n_hotspots': n_hot, 'verdict': verdict}