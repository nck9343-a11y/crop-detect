"""
论文插图生成
=============
三张图，各承担一段论证：

    图 1  六类标注框相对面积的分布      -> 2.1 节，问题的提出
    图 2  训练占比与跨物种假阳性占比    -> 4.2 节，捷径的证据
    图 3  三组反事实的每图假阳性        -> 4.4 节，因果与两种机制

配色约定（全文三图一致）：
    橙色 #eb6834  固定表示被操纵／标注最粗的那一类（mosaic virus disease）
    蓝色 #2a78d6  其余类别与对照
颜色绑定的是实体而非排名，因此同一类别在三张图里始终同色。
该两色组合在白色纸面上经色觉障碍模拟校验：最差配对 ΔE 24.7（protan）、
常视 33.6，二者对纸面对比度均不低于 3:1。

期刊多为灰度印刷，故不依赖颜色单独承载信息：
图 1 以分面区分类别，图 2 以空心／实心区分两个量，图 3 以网格纹区分两个系列。

输出 PDF（矢量，投稿用）与 PNG（300 dpi，预览用）至 paper/figures/。

用法：
    python scripts/Make_figures.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

ROOT = Path(r'D:\dev\crop-detect')
OUT = ROOT / 'paper' / 'figures'
GRAPE = ROOT / 'datasets' / 'grape_public'

LANG = 'zh'          # 由命令行 --lang en 切换

L = {
 'zh': dict(
    xlabel1='标注框面积占全图比例（对数刻度）', median='中位', n='n = ',
    xlabel2='占比 / %', legend_train='训练集中的标注框占比',
    legend_pred='跨物种阴性图像上的假阳性占比', over='过表达 13.41×',
    area='框中位面积 {:.2f}%',
    ylabel3='每图假阳性框数', total='假阳性总数 {}',
    groups=['E1 原标注', '缩 mosaic 框', '缩 botrytis 框'],
 ),
 'en': dict(
    xlabel1='Box area as a fraction of the image (log scale)',
    median='median', n='n = ',
    xlabel2='Share / %', legend_train='Share of boxes in the training set',
    legend_pred='Share of false positives on cross-species images',
    over='13.41× over-representation',
    area='median box area {:.2f}%',
    ylabel3='False positives per image', total='total FP {}',
    groups=['E1 (original)', 'mosaic shrunk', 'botrytis shrunk'],
 ),
}

ORANGE, BLUE = '#eb6834', '#2a78d6'
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
TARGET = 'mosaic virus disease'

# Windows 自带黑体；缺失时回落到 matplotlib 默认（中文会显示为方块，此时需另行指定）
CJK = Path('C:/Windows/Fonts/simhei.ttf')


def setup():
    # 字体回落链：西文走 DejaVu Sans，中文回落到黑体。
    # 若只指定黑体，其等宽风格的西文字形在类别名（全为英文）上很难看。
    if CJK.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(CJK))
        cjk_name = FontProperties(fname=str(CJK)).get_name()
        plt.rcParams['font.family'] = ['DejaVu Sans', cjk_name]
    plt.rcParams.update({
        'axes.unicode_minus': False,     # 否则负号在中文字体下显示为方块
        'font.size': 9,
        'axes.edgecolor': MUTED,
        'axes.linewidth': 0.6,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'grid.color': GRID,
        'grid.linewidth': 0.5,
        'legend.frameon': False,
        'figure.dpi': 120,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    if LANG != 'zh':
        name = f'{name}_{LANG}'
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=300)
    plt.close(fig)
    print(f'  已生成 {name}.pdf / {name}.png')


def load_areas():
    """按类别收集标注框相对面积（YOLO 归一化坐标下即 w x h）。"""
    import yaml
    names = yaml.safe_load(open(GRAPE / 'data.yaml', encoding='utf-8'))['names']
    areas = defaultdict(list)
    for sp in ['train', 'valid', 'test']:
        for lbl in (GRAPE / sp / 'labels').glob('*.txt'):
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    areas[names[int(p[0])]].append(float(p[3]) * float(p[4]))
    return {k: np.array(v) for k, v in areas.items()}


# --------------------------------------------------------------------------
def fig1_granularity(areas):
    """六类标注框面积分布的小多图。分面承担类别识别，颜色只作强调。"""
    order = sorted(areas, key=lambda k: -np.median(areas[k]))
    fig, axes = plt.subplots(len(order), 1, figsize=(5.4, 5.2), sharex=True)
    bins = np.logspace(-4, 0, 46)

    for ax, cls in zip(axes, order):
        v = areas[cls]
        hit = cls == TARGET
        ax.hist(v, bins=bins, color=ORANGE if hit else BLUE,
                alpha=0.95 if hit else 0.75, linewidth=0)
        ax.axvline(0.10, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        ax.axvline(0.30, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        ax.set_xscale('log')
        ax.set_yticks([])
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        if ax is not axes[-1]:
            # 共享横轴时每个分面都会画自己的刻度，只保留最下一格
            ax.tick_params(axis='x', which='both', length=0)
        ax.text(0.012, 0.72, f'{cls}', transform=ax.transAxes,
                fontsize=8, color=INK, fontweight='bold' if hit else 'normal')
        ax.text(0.012, 0.30,
                f'{L[LANG]["median"]} {np.median(v) * 100:.2f}%    '
                f'{L[LANG]["n"]}{v.size}',
                transform=ax.transAxes, fontsize=7.5, color=MUTED)

    axes[0].text(0.10, 1.28, '10%', transform=axes[0].get_xaxis_transform(),
                 ha='center', fontsize=7, color=MUTED)
    axes[0].text(0.30, 1.28, '30%', transform=axes[0].get_xaxis_transform(),
                 ha='center', fontsize=7, color=MUTED)
    axes[-1].set_xlabel(L[LANG]['xlabel1'])
    axes[-1].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    axes[-1].set_xticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
    fig.align_labels()
    save(fig, 'fig1_granularity')


# --------------------------------------------------------------------------
def fig2_overexpression():
    """训练占比 -> 跨物种假阳性占比。空心=训练，实心=预测，形状承担量的区分。"""
    rows = [  # 类别, 训练框中位面积, 训练占比 %, 预测占比 %   （表 2、表 9）
        (TARGET,                 43.16, 4.9, 65.7),
        ('grape botrytis cinerea', 8.12, 12.9, 31.3),
        ('grape ulcer disease',    3.29, 12.9, 0.0),
        ('grape downy mildew',     2.39, 14.5, 2.4),
        ('grape powdery mildew',   2.01, 19.4, 0.0),
        ('grape black rot',        0.57, 35.3, 0.5),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for i, (cls, area, tr, pr) in enumerate(rows):
        y = len(rows) - 1 - i
        c = ORANGE if cls == TARGET else BLUE
        ax.plot([tr, pr], [y, y], color=c, lw=1.6, alpha=0.55,
                solid_capstyle='round', zorder=1)
        ax.scatter([tr], [y], s=42, facecolor='white', edgecolor=c,
                   linewidth=1.4, zorder=2)
        ax.scatter([pr], [y], s=42, color=c, zorder=3)
        ax.text(-3.2, y, cls, ha='right', va='center', fontsize=8,
                color=INK, fontweight='bold' if cls == TARGET else 'normal')
        ax.text(-3.2, y - 0.34, L[LANG]['area'].format(area), ha='right', va='center',
                fontsize=7, color=MUTED)
        if cls == TARGET:
            ax.annotate(L[LANG]['over'], xy=(pr, y), xytext=(pr - 4, y + 0.42),
                        fontsize=7.5, color=ORANGE, ha='right')

    ax.set_xlim(-2, 72)
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.set_yticks([])
    ax.set_xlabel(L[LANG]['xlabel2'])
    ax.xaxis.grid(True, lw=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)

    ax.scatter([], [], s=42, facecolor='white', edgecolor=MUTED, linewidth=1.4,
               label=L[LANG]['legend_train'])
    ax.scatter([], [], s=42, color=MUTED, label=L[LANG]['legend_pred'])
    # 放在中右侧的空白区：除首行外各行的哑铃都止于 x<36，此处不会压到数据。
    # 英文图例比中文长得多，若沿用 lower right 会压住末行。
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.34), fontsize=7.5,
              handletextpad=0.4, labelcolor=INK, borderpad=0.2)
    save(fig, 'fig2_overexpression')


# --------------------------------------------------------------------------
def fig3_counterfactual():
    """三组反事实的每图假阳性。网格纹使灰度印刷下两系列仍可区分。"""
    # 一律由原始计数导出，不从每图值反推——后者已四舍五入到三位小数，
    # 反推出的百分比会与表 13 差一个进位（botrytis 的 +3.53% 会算成 +3%）。
    # 计数取自 results/runs_shortcut/shortcut_{E1,E10b,E12}.json
    N_IMG = 5156
    groups = L[LANG]['groups']
    cnt_mosaic = [5379, 1805, 2987]
    cnt_botrytis = [3537, 3662, 556]
    totals = [9176, 5971, 3956]

    mosaic = [c / N_IMG for c in cnt_mosaic]
    botrytis = [c / N_IMG for c in cnt_botrytis]

    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    b1 = ax.bar(x - w / 2, mosaic, w, color=ORANGE, linewidth=0,
                label='mosaic virus disease')
    b2 = ax.bar(x + w / 2, botrytis, w, color=BLUE, linewidth=0,
                hatch='///', edgecolor='white', label='grape botrytis cinerea')

    for bars, vals in ((b1, mosaic), (b2, botrytis)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.022, f'{v:.3f}',
                    ha='center', fontsize=7.5, color=INK)

    for i in (1, 2):
        ax.text(i - w / 2, mosaic[i] + 0.075,
                f'{(cnt_mosaic[i] / cnt_mosaic[0] - 1) * 100:+.0f}%', ha='center',
                fontsize=7.5, color=ORANGE)
        ax.text(i + w / 2, botrytis[i] + 0.075,
                f'{(cnt_botrytis[i] / cnt_botrytis[0] - 1) * 100:+.0f}%', ha='center',
                fontsize=7.5, color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    for i, t in enumerate(totals):
        ax.text(i, -0.19, L[LANG]['total'].format(t), ha='center', fontsize=7.5,
                color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_ylabel(L[LANG]['ylabel3'])
    ax.set_ylim(0, 1.28)
    ax.yaxis.grid(True, lw=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(loc='upper right', fontsize=7.5, labelcolor=INK)
    save(fig, 'fig3_counterfactual')


def main():
    global LANG
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if '--lang' in sys.argv:
        LANG = sys.argv[sys.argv.index('--lang') + 1]
    print(f'语言：{LANG}')
    setup()
    if not CJK.exists():
        print('  [警告] 未找到中文字体 simhei.ttf，中文将显示为方块')
    print('生成论文插图：')
    fig1_granularity(load_areas())
    fig2_overexpression()
    fig3_counterfactual()
    print(f'\n输出目录 {OUT}')


if __name__ == '__main__':
    main()
