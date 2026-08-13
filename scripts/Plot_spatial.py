"""
空间格局分析可视化
==================
运行后在 results/spatial_out/ 目录下自动生成图片，用法与 ultralytics 一致：

    python plot_spatial.py

生成两张图：
    results/spatial_out/patterns.png    四种格局的散点图 + Gi* 热点渲染 + Ripley's L 曲线
    results/spatial_out/gi_hist.png     Gi* z 值分布，用于确认热点判定阈值是否合理

说明：图中标签一律使用英文。matplotlib 默认字体不含中文字形，
渲染中文会显示为方框，需额外配置字体文件，此处直接规避。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无界面后端，保证在任何环境下都能出图
import matplotlib.pyplot as plt

from spatial_analysis import getis_ord, ripley_l, describe_pattern
from Spatial import (BOUNDS, pattern_random, pattern_clustered,
                          pattern_striped, pattern_edge, severity_for)

OUT_DIR = r'D:\dev\crop-detect\results\spatial_out'
RADII = np.linspace(0.4, 10.0, 22)

# describe_pattern 返回的判定是中文，图中改用英文，避免字体缺字形显示成方框
VERDICT_EN = {'聚集分布': 'CLUSTERED', '随机分布': 'RANDOM', '均匀分布': 'REGULAR'}

CASES = [
    ('Random',    pattern_random,    'random',    'CSR / genetic factor'),
    ('Clustered', pattern_clustered, 'clustered', 'infection foci'),
    ('Striped',   pattern_striped,   'striped',   'row-wise / irrigation'),
    ('Edge',      pattern_edge,      'edge',      'external introduction'),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    xmin, ymin, xmax, ymax = BOUNDS

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    all_gi = {}

    for col, (name, gen, mode, subtitle) in enumerate(CASES):
        pts = gen()
        sev = severity_for(pts, mode)
        res = describe_pattern(pts, sev, BOUNDS, k=8, verbose=False)
        gi = res['gi_star']
        all_gi[name] = gi

        # ---------- 上排：散点图，颜色为 Gi* z 值 ----------
        ax = axes[0, col]
        vmax = max(3.0, np.abs(gi).max())
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=gi, cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax, s=38,
                        edgecolors='k', linewidths=0.4)
        # 显著热点额外加圈
        hot = gi > 1.96
        if hot.any():
            ax.scatter(pts[hot, 0], pts[hot, 1], s=150, facecolors='none',
                       edgecolors='darkred', linewidths=1.4, zorder=3)

        ax.set_xlim(xmin - 1, xmax + 1)
        ax.set_ylim(ymin - 1, ymax + 1)
        ax.set_aspect('equal')
        ax.set_title(f'{name}  ({subtitle})', fontsize=13, weight='bold')
        ax.set_xlabel('x (m)')
        if col == 0:
            ax.set_ylabel('y (m)')
        ax.text(0.02, 0.97,
                f"R = {res['R']:.2f}\nMoran's I = {res['morans_i']:+.3f}\n"
                f"hotspots = {res['n_hotspots']}\n"
                f"-> {VERDICT_EN.get(res['verdict'], res['verdict'])}",
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='gray'))
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label='Gi* z-score')

        # ---------- 下排：Ripley's L 曲线 ----------
        ax = axes[1, col]
        obs, lo, hi = ripley_l(pts, BOUNDS, RADII, n_sim=99, seed=1)
        ax.fill_between(RADII, lo - RADII, hi - RADII, color='0.82',
                        label='95% CSR envelope')
        ax.axhline(0, color='0.4', lw=0.9, ls='--')
        ax.plot(RADII, obs - RADII, color='#B02418', lw=2.2, label='observed')

        # 标出峰值位置——即病害斑块的特征尺度
        d = obs - RADII
        if d.max() > 0:
            rpk = RADII[np.argmax(d)]
            ax.axvline(rpk, color='#2E5395', lw=1.1, ls=':')
            ax.annotate(f'peak r = {rpk:.1f} m',
                        xy=(rpk, d.max()), xytext=(6, 6),
                        textcoords='offset points', fontsize=10,
                        color='#2E5395', weight='bold')

        ax.set_xlabel('radius r (m)')
        if col == 0:
            ax.set_ylabel('L(r) - r')
            ax.legend(fontsize=9, loc='upper left')
        ax.set_title("Ripley's L", fontsize=11)
        ax.grid(alpha=0.25)

    fig.suptitle('Spatial Pattern Analysis of Disease Distribution  '
                 '(50 x 30 m plot, n = 200)',
                 fontsize=16, weight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(OUT_DIR, 'patterns.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f'saved -> {p1}')

    # ---------- 第二张：Gi* 分布直方图 ----------
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6), sharey=True)
    for ax, (name, _, _, _) in zip(axes, CASES):
        gi = all_gi[name]
        ax.hist(gi, bins=28, color='#8FA8C8', edgecolor='white')
        ax.axvline(1.96, color='#B02418', ls='--', lw=1.4)
        ax.axvline(-1.96, color='#2E5395', ls='--', lw=1.4)
        ax.set_title(f'{name}   (hot: {int((gi > 1.96).sum())})', fontsize=12)
        ax.set_xlabel('Gi* z-score')
    axes[0].set_ylabel('count')
    fig.suptitle('Getis-Ord Gi* Distribution   '
                 '(dashed lines = +/- 1.96, the 0.05 significance threshold)',
                 fontsize=13, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(OUT_DIR, 'gi_hist.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f'saved -> {p2}')


if __name__ == '__main__':
    main()