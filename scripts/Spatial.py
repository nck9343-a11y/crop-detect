"""
用合成数据验证空间统计方法是否真的能区分不同格局。

这一步的意义：如果连人为构造的理想数据都区分不出来，
真实数据必然更没戏。现在发现问题，比两个月后发现好。

模拟四种在葡萄园里真实存在的格局：
  A 随机     —— 品种因素、普遍性缺素
  B 聚集     —— 侵染性病害从发病中心扩散
  C 条带     —— 沿行分布，指向水肥不均或喷药遗漏
  D 边缘     —— 集中在田块边缘，指向外部传入或边行效应
"""

import numpy as np
from spatial_analysis import describe_pattern, ripley_l

BOUNDS = (0.0, 0.0, 50.0, 30.0)   # 50m x 30m 的地块
N = 200
rng = np.random.default_rng(42)


def pattern_random(n=N):
    """A 完全随机分布 (CSR)"""
    return np.column_stack([rng.uniform(0, 50, n), rng.uniform(0, 30, n)])


def pattern_clustered(n=N, n_parent=6, spread=2.0):
    """B 聚集分布：Thomas 过程 —— 若干发病中心，子代在中心附近散布"""
    parents = np.column_stack([rng.uniform(5, 45, n_parent),
                               rng.uniform(5, 25, n_parent)])
    idx = rng.integers(0, n_parent, n)
    pts = parents[idx] + rng.normal(0, spread, (n, 2))
    pts[:, 0] = np.clip(pts[:, 0], 0, 50)
    pts[:, 1] = np.clip(pts[:, 1], 0, 30)
    return pts


def pattern_striped(n=N, n_rows=5):
    """C 条带分布：集中在若干条平行行上（模拟沿垄或沿灌溉带）"""
    row_y = np.linspace(4, 26, n_rows)
    idx = rng.integers(0, n_rows, n)
    return np.column_stack([
        rng.uniform(0, 50, n),
        row_y[idx] + rng.normal(0, 0.6, n),
    ])


def pattern_edge(n=N, margin=5.0):
    """D 边缘分布：绝大多数点落在地块外围一圈"""
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(0, 50), rng.uniform(0, 30)
        near_edge = (x < margin or x > 50 - margin or
                     y < margin or y > 30 - margin)
        # 边缘区接受概率高，内部低
        if rng.random() < (0.95 if near_edge else 0.05):
            pts.append((x, y))
    return np.array(pts)


def severity_for(coords, mode):
    """
    为每个点附一个"严重度"数值，用于 Moran's I 与 Gi*。
    聚集/条带/边缘格局下，严重度也随位置变化；随机格局下严重度也随机。
    """
    if mode == 'random':
        return rng.uniform(1, 10, len(coords))
    if mode == 'clustered':
        # 离最近的高发中心越近，严重度越高
        center = coords.mean(axis=0)
        d = np.linalg.norm(coords - center, axis=1)
        return np.clip(10 - d / 3 + rng.normal(0, 1, len(coords)), 0.5, 10)
    if mode == 'striped':
        # 严重度随 y 周期变化
        return np.clip(5 + 4 * np.sin(coords[:, 1] * 1.1) + rng.normal(0, 0.8, len(coords)), 0.5, 10)
    if mode == 'edge':
        dx = np.minimum(coords[:, 0], 50 - coords[:, 0])
        dy = np.minimum(coords[:, 1], 30 - coords[:, 1])
        d = np.minimum(dx, dy)
        return np.clip(10 - d, 0.5, 10)
    raise ValueError(mode)


CASES = [
    ('A 随机分布  (预期: 随机, I≈0)', pattern_random, 'random'),
    ('B 聚集分布  (预期: 聚集, I>0)', pattern_clustered, 'clustered'),
    ('C 条带分布  (预期: 聚集, I>0)', pattern_striped, 'striped'),
    ('D 边缘分布  (预期: 聚集, I>0)', pattern_edge, 'edge'),
]


def main():
    print('=' * 62)
    print('空间格局分析方法验证  ——  50m x 30m 地块, 每种格局 200 个点')
    print('=' * 62)

    results = {}
    for title, gen, mode in CASES:
        print(f'\n{title}')
        print('-' * 62)
        pts = gen()
        sev = severity_for(pts, mode)
        results[title] = describe_pattern(pts, sev, BOUNDS, k=8)

    # Ripley's L：看聚集发生在什么尺度
    print('\n' + '=' * 62)
    print("Ripley's L 多尺度分析  (L(r)-r > 上包络 即该尺度显著聚集)")
    print('=' * 62)
    radii = np.array([0.5, 1, 2, 3, 5, 8])
    for title, gen, _ in CASES:
        pts = gen()
        obs, lo, hi = ripley_l(pts, BOUNDS, radii, n_sim=99)
        marks = ''.join('聚' if o > h else ('均' if o < l else '·')
                        for o, l, h in zip(obs, lo, hi))
        detail = '  '.join(f'{r:g}m:{o - r:+.2f}' for r, o in zip(radii, obs))
        print(f'\n{title[:12]}')
        print(f'  尺度标记 {marks}   (半径 {", ".join(f"{r:g}" for r in radii)} m)')
        print(f'  L(r)-r   {detail}')

    print('\n' + '=' * 62)
    print('结论汇总')
    print('=' * 62)
    for title, res in results.items():
        print(f'  {title[:12]:<14} 判定={res["verdict"]:<6} '
              f"R={res['R']:.2f}  I={res['morans_i']:+.3f}  热点={res['n_hotspots']}")


if __name__ == '__main__':
    main()