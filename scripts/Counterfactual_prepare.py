"""
E10 数据构造：捷径假说的反事实数据集
=====================================
目的：把"标注粒度"这一变量从"类别身份"中拆出来单独操纵。

问题：
    在原数据集中，mosaic virus disease 同时具备两个特殊性——
      (1) 它是一个特定的类别
      (2) 它的标注框是整叶级的（中位面积占比 43.16%，其余五类为 0.57%-8.12%）
    二者完全混淆，仅凭原数据无法判断捷径由哪一个造成。

构造三个反事实数据集：

    grape_cf_drop   剔除 mosaic 类，仅保留 5 类
                    -> 操纵"类别是否存在"，检验假阳性是否由该类承载

    grape_cf_shrink 保留 mosaic 类与全部样本，但把该类每个框裁到
                    中心 25% 面积（边长 x0.5），面积占比 43% -> 约 11%，
                    与 botrytis 同量级
                    -> 仅操纵"标注粒度"，类别身份、样本量、图像均不变

    grape_cf_expand 反向操作：把最细的 black rot（中位 0.57%）粗化到
                    整叶级。见下方"E13"。

关于裁剪合法性：
    花叶病毒的症状为全叶弥漫性花斑，而非局部病斑，故中心裁剪后
    框内仍为真实症状组织，不引入错误标注。此前提对局部病斑类
    （如黑腐病的褐色圆斑）不成立，因此该操作只施加于 mosaic 类。

E13（expand）：把细粒度类粗化，检验充分性
------------------------------------------
    此前三组（drop/shrink/placebo）都是"把粗的改细"，只证明了必要性：
    去掉粗粒度，捷径减弱。反方向未验证——细粒度类被改粗后，是否
    同样成为假阳性的汇点？若是，则粒度与捷径的关系是双向对称的。

    该组同时回应一条生物学质疑：黑腐病是局部褐色圆斑的真菌病，
    不存在"系统侵染故应标大框"的理由。它被粗化后若同样成为汇点，
    即说明机制只取决于标注粒度，与病害生物学无关。

    操作：取同一张图内全部 black rot 病斑框的外接框（union bbox），
    以单个框替代原有的全部病斑框。不含任何可调系数——
    这正是标注员在"整叶级"口径下会画的那个框。

    结果（全 3288 张图统计）：
        每图病斑框    中位 7 个（最大 34）  ->  1 个     mosaic 为 1 个
        中位面积占比  0.57%                ->  40.37%   mosaic 为 43.16%
        框数          4237                 ->  476      mosaic 为 593
        类别占比      35.3%                ->  5.8%     mosaic 为 4.9%
        union 框内真正是病斑的面积仅占 15.8%，其余为健康叶面。

    即黑腐病在粒度、每图框数、类别占比三项上同时成为 mosaic 的
    结构孪生，唯一剩余差别为病害生物学本身。

    预测（按 5.2 节的筛查量，而非按直觉）：
        原数据集   R = 15.20x   m(1)/m(2) = 5.32x   单一孤立的最粗类
        expand     R =  7.56x   m(1)/m(2) = 1.07x   两个并列的最粗类

    m(1)/m(2) 由 5.32 落到 1.07，意味着该数据集按其自身筛查量已属
    "无孤立汇点"那一型，即 FieldPlant 型。故预测的不是汇点整体转移到
    黑腐病，而是汇点在 mosaic 与黑腐病两个并列粗类之间分裂。
    若实测证实，则 m(1)/m(2) 由一个描述量升格为有预测力的量——
    这比单纯的双向对称更强。

    注意：粗化必然使框数下降（整叶级标注本就一叶一框），故该组的
    类别占比改变不是可消除的混淆，而是粒度的一部分。判据 A 的
    过表达倍数因分母变小会被机械抬高约 6 倍，故 Shortcut_experiment.py
    对该组须用 --prior-from 重算训练占比，并以原始预测占比为主指标。

用法：
    python Counterfactual_prepare.py            # e10: drop + shrink
    python Counterfactual_prepare.py --only e12 # 安慰剂组
    python Counterfactual_prepare.py --only e13 # 扩框组
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

# E13 扩框组：black rot 在原 names 列表中的索引（最细的类，中位 0.57%）
EXPAND_ID = 0


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


def union_box(boxes):
    """把同一张图内的多个框合并为其外接框。

    boxes 为 [(cx, cy, w, h), ...]，归一化坐标。返回同格式的单个框。
    个别框可能已略微越界，故对结果做一次 [0,1] 截断。
    """
    x1 = max(min(cx - w / 2 for cx, _, w, _ in boxes), 0.0)
    y1 = max(min(cy - h / 2 for _, cy, _, h in boxes), 0.0)
    x2 = min(max(cx + w / 2 for cx, _, w, _ in boxes), 1.0)
    y2 = min(max(cy + h / 2 for _, cy, _, h in boxes), 1.0)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def expand_file(lines, target_id):
    """E13：把一张图内目标类的全部病斑框替换为它们的外接框。

    非目标类的框原样保留。返回 (新的行列表, 新框面积或 None)。
    """
    keep, tgt = [], []
    for line in lines:
        p = line.split()
        if len(p) < 5:
            continue
        if int(p[0]) == target_id:
            tgt.append(tuple(float(v) for v in p[1:5]))
        else:
            keep.append(' '.join(p[:5]))
    if not tgt:
        return keep, None
    cx, cy, w, h = union_box(tgt)
    keep.append(f'{target_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return keep, w * h


def build(dst, mode, names, target_id=TARGET_ID):
    """mode: 'drop' 剔除目标类 | 'shrink' 缩小目标类的框
             | 'expand' 把目标类的病斑框合并为整叶级外接框
    """
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
            raw_lines = lbl.read_text().splitlines()

            # expand 需要一次看到整张图的框，故不走逐行分支
            if mode == 'expand':
                out, area = expand_file(raw_lines, target_id)
                if area is not None:
                    stats['merged_from'] += sum(
                        1 for ln in raw_lines
                        if ln.split() and int(ln.split()[0]) == target_id)
                    stats['merged_to'] += 1
                    areas.append(area)
                stats['kept'] += len(out)

            for line in ([] if mode == 'expand' else raw_lines):
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
    elif mode == 'expand':
        print(f'，其中 [{target_id}] {names[target_id]} 由 {stats["merged_from"]} '
              f'个病斑框合并为 {stats["merged_to"]} 个整叶框')
        if areas:
            a = np.array(areas)
            print(f'  合并后面积占比  中位 {np.median(a) * 100:.2f}%   '
                  f'均值 {a.mean() * 100:.2f}%   '
                  f'P10 {np.percentile(a, 10) * 100:.2f}%   '
                  f'P90 {np.percentile(a, 90) * 100:.2f}%')
            print(f'  参照 mosaic 原始标注：中位 43.16%  P10 5.31%  P90 64.83%')
    else:
        print(f'，缩小 {stats["shrunk"]} 个')
        if areas:
            print(f'  缩小后 [{target_id}] {names[target_id]} 的面积占比中位 '
                  f'{np.median(areas) * 100:.2f}%  (边长 x{SHRINK}，面积为原 {SHRINK ** 2:.0%})')
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['e10', 'e12', 'e13'], default='e10',
                    help='e10 构造 drop+shrink（目标 mosaic）；e12 安慰剂组（目标 botrytis）；'
                         'e13 扩框组（目标 black rot，粗化到整叶级）')
    a = ap.parse_args()

    names = load_names()
    print('=' * 66)
    print('反事实数据集构造')
    print('=' * 66)

    if a.only == 'e10':
        print(f'  源数据集 {SRC.name}   目标类 [{TARGET_ID}] {names[TARGET_ID]}')
        build(DST_ROOT / 'grape_cf_drop', 'drop', names, TARGET_ID)
        build(DST_ROOT / 'grape_cf_shrink', 'shrink', names, TARGET_ID)
    elif a.only == 'e12':
        print(f'  源数据集 {SRC.name}   安慰剂目标类 [{PLACEBO_ID}] {names[PLACEBO_ID]}')
        print(f'  预期：该类自身过表达下降，而 mosaic 的 11.95x 保持不变')
        build(DST_ROOT / 'grape_cf_placebo', 'shrink', names, PLACEBO_ID)
        return
    else:
        print(f'  源数据集 {SRC.name}   扩框目标类 [{EXPAND_ID}] {names[EXPAND_ID]}')
        print(f'  操作：同图内全部病斑框 -> 其外接框（整叶级），无可调系数')
        build(DST_ROOT / 'grape_cf_expand', 'expand', names, EXPAND_ID)
        print('\n' + '=' * 66)
        print('筛查量（5.2 节口径，Granularity_stats.screening）')
        print('=' * 66)
        print('  原数据集   R = 15.20x   m(1)/m(2) = 5.32x   -> 单一孤立的最粗类')
        print('  expand     R =  7.56x   m(1)/m(2) = 1.07x   -> 两个并列的最粗类')
        print('                                    mosaic 43.16% / black rot 40.37%')
        print('  故理论预测的不是汇点整体转移，而是汇点在两个并列粗类之间分裂——')
        print('  即本数据集按自身筛查量已属 FieldPlant 那一型。')
        print('\n' + '=' * 66)
        print('判据（3x3 决策表，九格全部有归属，见 Shortcut_experiment.py 顶部）')
        print('=' * 66)
        print('  主指标为原始预测占比。E1 基线：black rot 0.41%，mosaic 58.62%')
        print('    black rot >10% + mosaic 下降   -> H1 汇点转移或分裂，双向成立')
        print('    black rot 2-10%                -> H2 部分归属倾向，未夺取汇点')
        print('    black rot <2% + mosaic 不变    -> H3 证伪，4.4 节归属机制需重写')
        print('    其余组合已在决策表中逐格写明，不留空档')
        print('  注意：该组框数由 4237 降至 476，训练占比 35.3% -> 5.8%，')
        print('        复测须加 --prior-from 重算分母，否则倍数被机械抬高约 6 倍。')
        print('=' * 66)
        return

    print('\n' + '=' * 66)
    print('下一步：分别以两份数据集重训 YOLO11n（其余超参与 E1 完全一致），')
    print('       再用 Shortcut_experiment.py 在 FieldPlant 上复测假阳性率。')
    print('  预期  shrink 组: 该类 AP 显著回落 + 假阳性率下降 -> 粒度是主因')
    print('        drop  组: 仅假阳性率下降              -> 该类承载了捷径')
    print('=' * 66)


if __name__ == '__main__':
    main()
