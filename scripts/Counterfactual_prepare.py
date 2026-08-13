"""
E10 数据构造：捷径假说的反事实数据集
=====================================
目的：把"标注粒度"这一变量从"类别身份"中拆出来单独操纵。

问题：
    在原数据集中，mosaic virus disease 同时具备两个特殊性——
      (1) 它是一个特定的类别
      (2) 它的标注框是整叶级的（中位面积占比 43.16%，其余五类为 0.57%-8.12%）
    二者完全混淆，仅凭原数据无法判断捷径由哪一个造成。

构造两个反事实数据集：

    grape_cf_drop   剔除 mosaic 类，仅保留 5 类
                    -> 操纵"类别是否存在"，检验假阳性是否由该类承载

    grape_cf_shrink 保留 mosaic 类与全部样本，但把该类每个框裁到
                    中心 25% 面积（边长 x0.5），面积占比 43% -> 约 11%，
                    与 botrytis 同量级
                    -> 仅操纵"标注粒度"，类别身份、样本量、图像均不变

关于裁剪合法性：
    花叶病毒的症状为全叶弥漫性花斑，而非局部病斑，故中心裁剪后
    框内仍为真实症状组织，不引入错误标注。此前提对局部病斑类
    （如黑腐病的褐色圆斑）不成立，因此该操作只施加于 mosaic 类。

用法：
    python Counterfactual_prepare.py
"""

import argparse
import shutil
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

SRC = Path(r'D:\dev\crop-detect\datasets\grape_public')
DST_ROOT = Path(r'D:\dev\crop-detect\datasets')

TARGET_ID = 5          # mosaic virus disease 在原 names 列表中的索引
SHRINK = 0.5           # 边长缩放系数，面积变为原来的 0.25
SPLITS = ['train', 'valid', 'test']

# E12 安慰剂／复现对照：改缩 botrytis（训练框面积第二大，8.12%）。
# 该组同时检验两件事——
#   复现性  若粒度机制普适，botrytis 自身的过表达（2.42x）应下降
#   特异性  mosaic 的过表达应保持在 11.95x 附近不变；若它也随之下降，
#           说明 E10b 的效果只是"扰动标注"的泛泛结果，而非针对被改类别
PLACEBO_ID = 1


def load_names():
    return yaml.safe_load(open(SRC / 'data.yaml', encoding='utf-8'))['names']


def write_yaml(dst, names):
    """注意：原 data.yaml 的 train/val/test 写成 '../train/images'，
    与 path 拼接后会指向数据集目录之外，仅靠框架的回退搜索才能工作。
    此处写成正确的相对路径。"""
    (dst / 'data.yaml').write_text(
        f'path: "{dst.as_posix()}"\n'
        'train: train/images\n'
        'val: valid/images\n'
        'test: test/images\n'
        f'nc: {len(names)}\n'
        f'names: {names}\n',
        encoding='utf-8')


def shrink_box(parts, k=SHRINK):
    """YOLO 格式 cls cx cy w h（归一化）。中心不变，宽高各乘 k。"""
    cls, cx, cy, w, h = parts[0], *[float(v) for v in parts[1:5]]
    return f'{cls} {cx:.6f} {cy:.6f} {w * k:.6f} {h * k:.6f}'


def build(dst, mode, names, target_id=TARGET_ID):
    """mode: 'drop' 剔除目标类 | 'shrink' 缩小目标类的框"""
    if dst.exists():
        shutil.rmtree(dst)

    if mode == 'drop':
        keep = [n for i, n in enumerate(names) if i != target_id]
        # 剔除一类后，其后类别的编号要整体前移，否则标签错位
        remap = {i: (i if i < target_id else i - 1)
                 for i in range(len(names)) if i != target_id}
    else:
        keep, remap = list(names), {i: i for i in range(len(names))}

    stats = defaultdict(int)
    areas = []
    for sp in SPLITS:
        (dst / sp / 'images').mkdir(parents=True, exist_ok=True)
        (dst / sp / 'labels').mkdir(parents=True, exist_ok=True)
        for lbl in (SRC / sp / 'labels').glob('*.txt'):
            out = []
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                cid = int(p[0])
                if mode == 'drop' and cid == target_id:
                    stats['removed'] += 1
                    continue
                if mode == 'shrink' and cid == target_id:
                    line = shrink_box(p)
                    stats['shrunk'] += 1
                    areas.append(float(p[3]) * SHRINK * float(p[4]) * SHRINK)
                else:
                    line = ' '.join([str(remap[cid])] + p[1:5])
                out.append(line)
                stats['kept'] += 1

            # drop 模式下，若一张图的框全被剔除则整图不保留——
            # 留下无框图像会被当作纯背景负样本，改变训练分布，引入额外变量
            if not out:
                stats['img_dropped'] += 1
                continue

            img = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                c = SRC / sp / 'images' / (lbl.stem + ext)
                if c.exists():
                    img = c
                    break
            if img is None:
                continue
            shutil.copy(img, dst / sp / 'images' / img.name)
            (dst / sp / 'labels' / (lbl.stem + '.txt')).write_text('\n'.join(out))
            stats[f'img_{sp}'] += 1

    write_yaml(dst, keep)
    print(f'\n--- {dst.name}  ({mode}) ---')
    print(f'  类别数 {len(keep)}')
    for sp in SPLITS:
        print(f'  {sp:<6} {stats[f"img_{sp}"]} 张')
    print(f'  保留框 {stats["kept"]}', end='')
    if mode == 'drop':
        print(f'，剔除 {stats["removed"]} 个，因无框而弃掉 {stats["img_dropped"]} 张图')
    else:
        print(f'，缩小 {stats["shrunk"]} 个')
        if areas:
            print(f'  缩小后 [{target_id}] {names[target_id]} 的面积占比中位 '
                  f'{np.median(areas) * 100:.2f}%  (边长 x{SHRINK}，面积为原 {SHRINK ** 2:.0%})')
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['e10', 'e12'], default='e10',
                    help='e10 构造 drop+shrink（目标 mosaic）；e12 构造安慰剂组（目标 botrytis）')
    a = ap.parse_args()

    names = load_names()
    print('=' * 66)
    print('反事实数据集构造')
    print('=' * 66)

    if a.only == 'e10':
        print(f'  源数据集 {SRC.name}   目标类 [{TARGET_ID}] {names[TARGET_ID]}')
        build(DST_ROOT / 'grape_cf_drop', 'drop', names, TARGET_ID)
        build(DST_ROOT / 'grape_cf_shrink', 'shrink', names, TARGET_ID)
    else:
        print(f'  源数据集 {SRC.name}   安慰剂目标类 [{PLACEBO_ID}] {names[PLACEBO_ID]}')
        print(f'  预期：该类自身过表达下降，而 mosaic 的 11.95x 保持不变')
        build(DST_ROOT / 'grape_cf_placebo', 'shrink', names, PLACEBO_ID)
        return

    print('\n' + '=' * 66)
    print('下一步：分别以两份数据集重训 YOLO11n（其余超参与 E1 完全一致），')
    print('       再用 Shortcut_experiment.py 在 FieldPlant 上复测假阳性率。')
    print('  预期  shrink 组: 该类 AP 显著回落 + 假阳性率下降 -> 粒度是主因')
    print('        drop  组: 仅假阳性率下降              -> 该类承载了捷径')
    print('=' * 66)


if __name__ == '__main__':
    main()
