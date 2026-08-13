"""
标注粒度一致性的量化统计
=========================
论文第 4 节的现象建立在单一数据集上，这是最主要的外部效度短板。
本脚本给出一个不依赖人工判读的粒度度量，并在多个数据集上施加同一口径，
用以回答：类别之间标注粒度不一致，是这一份数据的偶然，还是普遍现象。

度量（全部基于 YOLO 归一化标注，相对面积 = w × h，无需读取图像）：

    m_c        类内标注框相对面积的中位数
    粒度跨度 R  max(m_c) / median_c(m_c)
               取类间中位数而非最小值作分母，避免被单个极细类别放大
    双峰占比    类内 >30% 与 <10% 两端各自的占比

判定：
    m_c >= 0.30  记为整叶级（占全图三成以上的框不可能是一个病斑）
    m_c <= 0.10  记为病斑级
    二者在同一数据集内共存 -> 该数据集存在类别间粒度不一致

阈值 0.30／0.10 是先验设定的，不由数据挑选；中间地带（0.10~0.30）不作判定，
以免把连续变化强行二分。

用法：
    python scripts/Granularity_stats.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r'D:\dev\crop-detect')
OUT = ROOT / 'results' / 'runs_unified' / 'granularity_stats.json'

DATASETS = [
    ('grape_public', ROOT / 'datasets' / 'grape_public', '主数据集（葡萄，6 类）'),
    ('fieldplant',   ROOT / 'datasets' / 'fieldplant',   'FieldPlant（木薯／玉米／番茄，27 类）'),
]

COARSE, FINE = 0.30, 0.10          # 整叶级 / 病斑级的判定阈值，先验设定
SPLITS = ['train', 'valid', 'test']


def load_names(root):
    y = yaml.safe_load(open(root / 'data.yaml', encoding='utf-8'))
    return y['names']


def collect(root):
    """遍历三个划分，按类别汇总每个框的相对面积。"""
    areas = defaultdict(list)
    for sp in SPLITS:
        d = root / sp / 'labels'
        if not d.exists():
            continue
        for lbl in d.glob('*.txt'):
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                areas[int(p[0])].append(float(p[3]) * float(p[4]))
    return areas


def describe(v):
    v = np.asarray(v)
    return {
        'n': int(v.size),
        'median': float(np.median(v)),
        'mean': float(v.mean()),
        'p10': float(np.percentile(v, 10)),
        'p90': float(np.percentile(v, 90)),
        'frac_gt30': float((v > COARSE).mean()),
        'frac_lt10': float((v < FINE).mean()),
    }


def analyse(key, root, note):
    names = load_names(root)
    areas = collect(root)
    if not areas:
        print(f'  [跳过] {key}: 未找到标注')
        return None

    per_class = {}
    for cid, v in areas.items():
        per_class[names[cid] if cid < len(names) else f'class_{cid}'] = describe(v)

    med = np.array([s['median'] for s in per_class.values()])
    span = float(med.max() / np.median(med))
    coarse = [n for n, s in per_class.items() if s['median'] >= COARSE]
    fine = [n for n, s in per_class.items() if s['median'] <= FINE]

    res = {
        'note': note,
        'n_classes': len(per_class),
        'n_boxes': int(sum(s['n'] for s in per_class.values())),
        'span_R': span,
        'median_of_class_medians': float(np.median(med)),
        'coarse_classes': coarse,
        'fine_classes': fine,
        'inconsistent': bool(coarse and fine),
        'per_class': per_class,
    }

    print(f'\n{"=" * 92}')
    print(f'{key}  —  {note}')
    print(f'{"=" * 92}')
    print(f'{"类别":<34}{"框数":>8}{"中位面积":>10}{"P90":>9}{">30%":>8}{"<10%":>8}')
    print('-' * 92)
    for n, s in sorted(per_class.items(), key=lambda kv: -kv[1]['median']):
        flag = ' 整叶级' if s['median'] >= COARSE else ('' if s['median'] > FINE else ' 病斑级')
        print(f'{n[:33]:<34}{s["n"]:>8}{s["median"] * 100:>9.2f}%{s["p90"] * 100:>8.2f}%'
              f'{s["frac_gt30"] * 100:>7.1f}%{s["frac_lt10"] * 100:>7.1f}%{flag}')
    print('-' * 92)
    print(f'  类间中位数 {np.median(med) * 100:.2f}%   最粗类 {med.max() * 100:.2f}%   '
          f'粒度跨度 R = {span:.1f}x')
    print(f'  整叶级类别: {coarse if coarse else "无"}')
    print(f'  粒度不一致: {"是" if res["inconsistent"] else "否"}')
    return res


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('=' * 92)
    print('标注粒度一致性统计（阈值：整叶级 >= 30%，病斑级 <= 10%）')
    print('=' * 92)

    out = {}
    for key, root, note in DATASETS:
        if not root.exists():
            print(f'  [跳过] {key}: {root} 不存在')
            continue
        r = analyse(key, root, note)
        if r:
            out[key] = r

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n已保存 -> {OUT}')


if __name__ == '__main__':
    main()
