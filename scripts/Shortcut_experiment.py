"""
E9: 跨物种阴性对照 —— 捷径学习的定量验证
==========================================
目的：把"标注粒度不一致诱导捷径学习"从推理链变成实验结论。

背景：
    E1 训练所用的葡萄数据集中，mosaic virus disease 一类的标注对象为整片叶片，
    其标注框中位面积占比 43.16%，为其余五类中位数的 5.3-76 倍。
    该类同时是样本最少（593 框）却 AP 最高（0.781）的类别。

    据此提出假设：模型学到的并非花叶病毒的视觉特征，而是一条捷径——
        "画面中存在占据大面积的、叶片形状的绿色区域" -> mosaic virus disease

    此前的证据仅为 10 张自采手机照片 + 肉眼判断，样本量与判据均不足。

本实验：
    FieldPlant 数据集包含 5156 张木薯／玉米／番茄图像，
    其中不可能存在葡萄花叶病毒。因此 E1 在这批图像上输出的
    任何 mosaic virus disease 都是假阳性。

    这批数据提供了两件此前没有的东西：
      1. 三个数量级的样本量（5156 vs 10）
      2. 客观判据（跨物种，真值必为阴性，无需人工判断症状）

判据（三条独立证据，需同时成立）：
    A 类别失衡   mosaic 在预测中的占比 >> 其在训练标注中的占比 4.9%
    B 剂量-反应   P(预测为 mosaic) 随图像中最大绿色连通区域占比单调上升
    C 框尺寸复现  预测为 mosaic 的框，其面积占比应复现训练标注的 ~43% 量级

    三条同时成立 -> 捷径机制得到定量验证
    仅 A 成立     -> 存在类别偏置，但未必是绿色大区域这条捷径
    均不成立      -> 该规则被证伪

实际结果为"仅 A 成立"：类别偏置显著，但绿色大区域这条具体规则未获支持，
故 4.3 节将其撤回。归属机制改由 4.4 节的反事实重训承担——
本脚本此后的用途即是复测各反事实组的假阳性分布。

用法：
    python Shortcut_experiment.py


================================ ENGLISH ================================

E9: cross-species negative control - quantitative test of shortcut learning

Purpose: turn "inconsistent annotation granularity induces shortcut learning"
from a chain of reasoning into an experimental result.

Background:
    In the grape dataset used to train E1, the class `mosaic virus disease` is
    annotated at whole-leaf level: median box area 43.16% of the image, 5.3x to
    76x the medians of the other five classes. It is simultaneously the rarest
    class (593 boxes) and the highest-scoring one (AP 0.781).

    Hypothesis: the model has not learnt the visual features of mosaic virus,
    but a shortcut -
        "a large, leaf-shaped green region fills the frame" -> mosaic virus disease

    The prior evidence was 10 self-collected phone photos judged by eye:
    inadequate in both sample size and criteria.

This experiment:
    FieldPlant contains 5156 cassava / maize / tomato images, in which grape
    mosaic virus cannot occur. Every `mosaic virus disease` box E1 emits on
    these images is therefore a false positive.

    This gives two things the earlier evidence lacked:
      1. three orders of magnitude more images (5156 vs 10)
      2. an objective criterion (cross-species: ground truth is negative by
         construction, no human judgement of symptoms required)

Criteria (three independent lines of evidence, all required):
    A  class imbalance   share of mosaic in predictions >> its 4.9% share of
                         the training annotations
    B  dose-response     P(predict mosaic) rises monotonically with the largest
                         connected green region in the image
    C  box-size echo     boxes predicted as mosaic should reproduce the ~43%
                         area share of the training annotations

    all three hold  -> the shortcut mechanism is quantitatively verified
    only A holds    -> a class bias exists, but not necessarily via large green
                       regions
    none holds      -> this specific rule is refuted

The actual outcome was "only A holds": the class bias is pronounced, but the
large-green-region rule was not supported, and Section 4.3 retracts it. The
attribution mechanism is carried instead by the counterfactual retraining of
Section 4.4; from that point on this script is used to re-measure the
false-positive distribution of each counterfactual group.

Usage:
    python Shortcut_experiment.py
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO

# ------------------------------------------------------------------ 配置

WEIGHTS = r'D:\dev\crop-detect\runs\detect\grape_n\weights\best.pt'   # E1
FIELDPLANT = Path(r'D:\dev\crop-detect\datasets\fieldplant')
OUT_DIR = Path(r'D:\dev\crop-detect\results\runs_shortcut')

CONF = 0.25          # 部署时的常用阈值，与 §4.2 的域偏移实验一致
IMGSZ = 640

# 训练集中各类标注框占比（来自 datasets/grape_public 全量统计），
# 用作"若无偏置，预测分布应大致相当"的参照基线。
#
# 该表对 E1/E10a/E10b/E12 均适用：drop 组按剩余类别重新归一化即可，
# shrink 组只改框的尺寸、不改框数。但 E13（扩框组）把 4237 个病斑框
# 合并成 476 个整叶框，占比由 35.3% 降至 5.8%，此表不再成立，
# 须用 --prior-from 从该组数据集现场重算。
TRAIN_PRIOR = {
    'grape black rot':        0.353,
    'grape powdery mildew':   0.194,
    'grape downy mildew':     0.145,
    'grape botrytis cinerea': 0.129,
    'grape ulcer disease':    0.129,
    'mosaic virus disease':   0.049,
}
TARGET = 'mosaic virus disease'

# 训练标注中该类框的中位面积占比，用于判据 C
TARGET_TRAIN_AREA = 0.4316

GREEN_BINS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 1.01]   # 判据 B 的分组


# ================================================================ E13 预注册判据
#                                                   E13 PRE-REGISTERED CRITERIA
#
# [EN] The criteria below were written into this file before the E13 group
#      finished training, and are derived from the screening statistics of
#      grape_cf_expand - not from any observed false-positive count. The
#      conclusion is fixed before the result is seen. Section 4.4 / Result 5 of
#      the paper rests on this block; readers checking that claim should read
#      BR_CUTS, MO_CUTS, E13_BASELINE and E13_TABLE below. The verdict these
#      yield is written to results/runs_shortcut/shortcut_E13.json under the
#      `prereg_verdict` key, alongside the raw false-positive counts.
#
#      Why this is not rigged: substituting the E1 baseline values into
#      E13_TABLE lands on ('<2%', '基本不变') = "H3 证伪" (H3 refuted). That is,
#      "nothing changed" necessarily falls on the refuting side, so the table
#      cannot manufacture a confirmation. All nine cells carry a pre-assigned
#      reading - leaving any cell blank would leave room for post-hoc
#      interpretation.
#
# 结论必须在看到结果之前定死。以下判据于训练完成前写入本文件，
# 依据是 grape_cf_expand 的筛查量，而非任何实测的假阳性数字。
#
# 该数据集的筛查量（Granularity_stats.screening 同一口径）：
#     原数据集   R = 15.20x   m1/m2 = 5.32x   -> 单一孤立的最粗类
#     expand     R =  7.56x   m1/m2 = 1.07x   -> 两个并列的最粗类
#                             (mosaic 43.16% / black rot 40.37%)
#
# 关键：按 5.2 节的筛查量，expand 组属于"有若干并列粗类、无孤立汇点"
# 那一型——即 FieldPlant 型。故理论预测的不是"汇点整体转移到黑腐病"，
# 而是汇点在两个并列粗类之间分裂。若实测证实，则 m1/m2 由描述量升格为
# 有预测力的量，这比单纯的双向对称更强。
#
# [EN] By the screening statistics of Section 5.2, the expand group belongs to
#      the "several equally coarse classes, no isolated sink" regime - the
#      FieldPlant type. The prediction is therefore NOT that the sink migrates
#      wholesale to black rot, but that it splits between the two equally coarse
#      classes. Confirmation would promote m1/m2 from a descriptive statistic to
#      a predictive one, which is a stronger claim than plain two-way symmetry.
#
# 主指标为原始预测占比：expand 组的框数由 4237 降至 476，训练占比
# 35.3% -> 5.8%，过表达倍数的分母缩小 6.1 倍会机械抬高倍数。
#
# [EN] The primary measure is the raw predicted share, not the over-representation
#      ratio: the expand group's box count drops from 4237 to 476 and its share of
#      the annotations from 35.3% to 5.8%, so a denominator 6.1x smaller would
#      inflate the ratio mechanically.
#
# E1 基线（results/runs_shortcut/shortcut_E1.json，9176 个假阳性框）：
#     black rot    38 / 9176 =  0.41%
#     mosaic     5379 / 9176 = 58.62%
E13_BASELINE = {'grape black rot': 0.0041, 'mosaic virus disease': 0.5862}

# 分档阈值。黑腐病三档由 2%/10% 切分；mosaic 三档为
#   显著下降 <40%   中等下降 40%~53.6%（较基线降 5 个百分点以上）   基本不变 >=53.6%
#
# [EN] Band thresholds. Black rot is split at 2% / 10%. Mosaic's three bands are
#      显著下降 (marked drop, <40%), 中等下降 (moderate drop, 40%-53.6%, i.e. at
#      least 5 percentage points below baseline), 基本不变 (essentially unchanged,
#      >=53.6%).
BR_CUTS = (0.02, 0.10)
MO_CUTS = (0.40, 0.536)

# 3x3 决策表。九格全部有归属——留空档等于给事后解释留口子。
#
# [EN] The 3x3 decision table, keyed (black-rot band, mosaic band). All nine cells
#      carry a pre-assigned reading. English glosses of each verdict:
#        H1 成立            = H1 holds: the sink migrates with granularity
#        H1 成立（分裂型）  = H1 holds, split form: the two equally coarse classes
#                             share the sink - the expected outcome for this group
#        未预见 A / B       = unforeseen A / B: outcomes the hypotheses did not
#                             anticipate, each with a prescribed follow-up check
#        H2 部分成立        = H2 partly holds: granularity biases attribution but
#                             cannot capture the sink
#        H3 证伪            = H3 refuted: granularity is not sufficient to create
#                             a sink; the attribution mechanism of 4.4 must be
#                             rewritten
#      The `reading` strings below are written verbatim into shortcut_E13.json;
#      they are left in Chinese so that re-running this script reproduces the
#      published artifact byte for byte.
E13_TABLE = {
    ('>10%', '显著下降'): (
        'H1 成立',
        '汇点随粒度转移；双向对称成立。m1/m2 由 5.32 降至 1.07 所预测的'
        '"无孤立汇点"得到独立验证，筛查量具有预测力。'),
    ('>10%', '中等下降'): (
        'H1 成立（分裂型）',
        '两个并列粗类分摊汇点——正是 m1/m2 = 1.07 所预测的形态，'
        '亦即 FieldPlant 型。这是本组的首选预期结果。'),
    ('>10%', '基本不变'): (
        '未预见 A',
        '黑腐病夺得汇点而 mosaic 未让位。须检查假阳性总量（fp_per_image）'
        '是否上升：若上升，则汇点并非零和，E12 中"缩 botrytis 使 mosaic 反升"'
        '的竞争性解读需要修正。'),
    ('2-10%', '显著下降'): (
        '未预见 B',
        'mosaic 让出汇点而黑腐病未接手。须查汇点是否流向 botrytis——'
        'E10b 曾出现该现象（botrytis 2.99x -> 4.75x）。'),
    ('2-10%', '中等下降'): (
        'H2 部分成立',
        '粒度可造成部分归属倾向，但不足以夺取汇点。'),
    ('2-10%', '基本不变'): (
        'H2 部分成立',
        '粒度可造成部分归属倾向，但不足以夺取汇点；'
        '须检查是否因 40.37% 仍略低于 43.16%。'),
    ('<2%', '显著下降'): (
        '未预见 B',
        'mosaic 让出汇点而黑腐病未接手，须查汇点流向何处。'),
    ('<2%', '中等下降'): (
        'H3 证伪',
        '粒度不足以造成汇点，4.4 节的归属机制要重写。'),
    ('<2%', '基本不变'): (
        'H3 证伪',
        '粒度不足以造成汇点，4.4 节的归属机制要重写。'),
}


def e13_verdict(pred_share):
    """按预注册的 3x3 表给出判定。pred_share 为各类的原始预测占比。"""
    br = pred_share.get('grape black rot', 0.0)
    mo = pred_share.get('mosaic virus disease', 0.0)
    br_band = '>10%' if br > BR_CUTS[1] else ('2-10%' if br >= BR_CUTS[0] else '<2%')
    mo_band = ('显著下降' if mo < MO_CUTS[0]
               else ('中等下降' if mo < MO_CUTS[1] else '基本不变'))
    name, text = E13_TABLE[(br_band, mo_band)]
    return {'black_rot_share': br, 'mosaic_share': mo,
            'black_rot_baseline': E13_BASELINE['grape black rot'],
            'mosaic_baseline': E13_BASELINE['mosaic virus disease'],
            'black_rot_band': br_band, 'mosaic_band': mo_band,
            'verdict': name, 'reading': text}


def prior_from_dataset(root):
    """从数据集现场统计各类标注框占比，替代硬编码的 TRAIN_PRIOR。

    用于框数被操纵过的组（E13）——此时占比的分母变了，沿用旧表会使
    过表达倍数被机械抬高，得出虚假的结论。
    """
    import yaml
    from collections import Counter

    root = Path(root)
    names = yaml.safe_load(open(root / 'data.yaml', encoding='utf-8'))['names']
    cnt = Counter()
    for sp in ['train', 'valid', 'test']:
        d = root / sp / 'labels'
        if not d.exists():
            continue
        for lbl in d.glob('*.txt'):
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    cnt[names[int(p[0])]] += 1
    total = sum(cnt.values())
    if not total:
        raise SystemExit(f'未能从 {root} 统计到任何标注框')
    return {n: cnt[n] / total for n in names}, total


# ------------------------------------------------------------------ 绿色区域度量

def green_metrics(img_bgr, max_side=512):
    """
    计算两个量，用于刻画"捷径触发条件"的强弱：
        green_frac  绿色像素占全图比例
        blob_frac   最大绿色连通区域占全图比例  <- 更贴近"一片大叶子"

    用 blob_frac 而非 green_frac 作为主自变量：草地背景也能有很高的
    绿色占比，但它是碎的；"整片叶片"的特征是单个大连通块。
    """
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # OpenCV 的 H 取值 0-179，绿色大致落在 25-95
    mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([95, 255, 255]))
    # 开运算去掉椒盐噪点，避免把碎草地算成连通块
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    total = mask.size
    green_frac = float(mask.sum() / 255 / total)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blob_frac = 0.0
    if n > 1:
        # stats[0] 是背景，取其余中面积最大者
        blob_frac = float(stats[1:, cv2.CC_STAT_AREA].max() / total)
    return green_frac, blob_frac


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser(description='跨物种阴性对照')
    ap.add_argument('--weights', default=WEIGHTS, help='待测权重')
    ap.add_argument('--out', default=str(OUT_DIR), help='输出目录')
    ap.add_argument('--tag', default='E1', help='本组的标签，用于结果文件命名')
    ap.add_argument('--reduce', type=int, default=4, choices=[1, 2, 4],
                    help='JPEG 解码降采样倍数。FieldPlant 为 13MP 原图，'
                         '全分辨率解码 67ms/张而 1/4 仅 20ms，'
                         '且图像最终统一缩至 640 送入模型，故不损失有效信息。'
                         '各组必须取相同值，否则解码链路不同、结果不可比。')
    ap.add_argument('--prior-from', default=None, metavar='DATASET_DIR',
                    help='从该数据集现场统计训练标注占比，替代硬编码的 TRAIN_PRIOR。'
                         '框数被操纵过的组（如 E13 扩框组）必须使用。')
    ap.add_argument('--verdict', choices=['e13'], default=None,
                    help='套用预注册的判定表。e13 用本文件顶部的 3x3 表。')
    args = ap.parse_args()
    decode_flag = {1: cv2.IMREAD_COLOR,
                   2: cv2.IMREAD_REDUCED_COLOR_2,
                   4: cv2.IMREAD_REDUCED_COLOR_4}[args.reduce]

    weights, out_dir, tag = args.weights, Path(args.out), args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in FIELDPLANT.rglob('*')
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not imgs:
        raise SystemExit(f'未找到图像: {FIELDPLANT}')

    print('=' * 70)
    print('E9  跨物种阴性对照 —— 捷径学习的定量验证')
    print('=' * 70)
    print(f'  组别     {tag}')
    print(f'  权重     {Path(weights).name}')
    print(f'  阴性集   FieldPlant  {len(imgs)} 张 (木薯/玉米/番茄)')
    print(f'  真值     全部为阴性——该数据集不含葡萄，故任何葡萄病害输出均为假阳性')
    print(f'  阈值     conf={CONF}  (与 §4.2 域偏移实验一致)')

    model = YOLO(weights)
    names = model.names
    # drop 组的模型不含 mosaic 类，此时判据 B/C 无从谈起，
    # 观察重点转为假阳性总量与类别分布是否发生转移。
    has_target_cls = TARGET in names.values()
    print(f'  类别     {list(names.values())}')
    if not has_target_cls:
        print(f'  注意     该模型不含 "{TARGET}"，将只报告假阳性总量与分布')

    records = []
    box_cls = Counter()          # 逐框的类别计数
    img_with_det = 0

    print(f'\n  推理中 ...')
    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p), decode_flag)
        if img is None:
            continue
        gf, bf = green_metrics(img)

        # 直接把已解码的数组交给模型，避免让 ultralytics 再解码一次原图。
        # FieldPlant 为 3456x3808 的手机原图，单次 JPEG 解码约 150-300ms，
        # 重复解码会使全程耗时翻倍。
        r = model.predict(img, imgsz=IMGSZ, conf=CONF,
                          verbose=False, device=0)[0]
        H, W = r.orig_shape
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append({
                'cls': names[int(b.cls[0])],
                'conf': float(b.conf[0]),
                'area_frac': (x2 - x1) * (y2 - y1) / (W * H),
            })
            box_cls[names[int(b.cls[0])]] += 1

        if dets:
            img_with_det += 1
        top = max(dets, key=lambda d: d['conf']) if dets else None
        records.append({
            'file': p.name, 'green_frac': gf, 'blob_frac': bf,
            'n_det': len(dets),
            'top_cls': top['cls'] if top else None,
            'top_conf': top['conf'] if top else 0.0,
            'top_area': top['area_frac'] if top else 0.0,
            'has_target': any(d['cls'] == TARGET for d in dets),
            'target_areas': [d['area_frac'] for d in dets if d['cls'] == TARGET],
        })
        if i % 500 == 0:
            print(f'    {i}/{len(imgs)}')

    n = len(records)
    total_boxes = sum(box_cls.values())

    # ---------------- 判据 A：类别失衡 ----------------
    print('\n' + '=' * 70)
    print('判据 A  预测类别分布  vs  训练标注分布')
    print('=' * 70)
    print(f'{"类别":<26}{"预测框数":>10}{"预测占比":>10}{"训练占比":>10}{"过表达倍数":>12}')
    print('-' * 70)
    # drop 组的模型只有 5 类，其训练占比需在剩余类别上重新归一化，
    # 否则过表达倍数会被系统性低估，与 6 类模型无法比较。
    prior_table = TRAIN_PRIOR
    screen = None
    if args.prior_from:
        prior_table, n_ann = prior_from_dataset(args.prior_from)
        print(f'  [训练占比现场重算自 {Path(args.prior_from).name}，共 {n_ann} 个框]')
        # 筛查量与假阳性结果必须同时落盘：事后补算就不算预注册了
        from Granularity_stats import screening
        screen = screening(Path(args.prior_from))
        if screen:
            print(f'  [该数据集筛查量  R = {screen["span_R"]:.2f}x   '
                  f'm(1)/m(2) = {screen["top_gap"]:.2f}x   '
                  f'整叶级类别 {screen["coarse_classes"]}]')
    present = {c: p for c, p in prior_table.items() if c in names.values()}
    norm = sum(present.values())
    prior_used = {c: p / norm for c, p in present.items()}

    over, pred_share = {}, {}
    for c, prior in sorted(prior_used.items(), key=lambda kv: -kv[1]):
        cnt = box_cls.get(c, 0)
        share = cnt / total_boxes if total_boxes else 0.0
        ratio = share / prior if prior else 0.0
        over[c] = ratio
        pred_share[c] = share
        mark = '  <<<' if c == TARGET else ''
        print(f'{c:<26}{cnt:>10}{share:>9.1%}{prior:>10.1%}{ratio:>11.2f}x{mark}')
    print('-' * 70)
    print(f'  有输出的图像 {img_with_det}/{n} ({img_with_det / n:.1%})，共 {total_boxes} 个假阳性框')
    A_ok = over.get(TARGET, 0) > 2.0

    # ---------------- 判据 B：剂量-反应 ----------------
    print('\n' + '=' * 70)
    print('判据 B  P(预测含 mosaic) 随最大绿色连通区域占比的变化')
    print('=' * 70)
    print(f'{"blob_frac 区间":<20}{"图像数":>8}{"含mosaic":>10}{"比例":>10}')
    print('-' * 70)
    curve = []
    for lo, hi in zip(GREEN_BINS[:-1], GREEN_BINS[1:]):
        sub = [r for r in records if lo <= r['blob_frac'] < hi]
        if not sub:
            continue
        k = sum(r['has_target'] for r in sub)
        rate = k / len(sub)
        curve.append({'lo': lo, 'hi': hi, 'n': len(sub), 'rate': rate})
        bar = '#' * int(rate * 40)
        print(f'{f"[{lo:.2f}, {hi:.2f})":<20}{len(sub):>8}{k:>10}{rate:>9.1%}  {bar}')
    print('-' * 70)
    # 单调性检验：相邻区间比例是否总体上升（允许个别回落）
    rates = [c['rate'] for c in curve]
    mono = sum(b >= a for a, b in zip(rates, rates[1:]))
    B_ok = len(rates) >= 3 and rates[-1] > rates[0] * 1.5 and mono >= len(rates) - 2
    if len(rates) >= 2:
        print(f'  最低组 {rates[0]:.1%}  ->  最高组 {rates[-1]:.1%}'
              f'   (相邻区间上升 {mono}/{len(rates) - 1} 次)')

    # ---------------- 判据 C：框尺寸复现 ----------------
    print('\n' + '=' * 70)
    print('判据 C  预测框面积占比  vs  训练标注的面积占比')
    print('=' * 70)
    areas = defaultdict(list)
    for r in records:
        for a in r['target_areas']:
            areas[TARGET].append(a)
    ta = areas.get(TARGET, [])
    C_ok = False
    if ta:
        med = float(np.median(ta))
        C_ok = med > 0.20
        print(f'  预测为 {TARGET} 的框: {len(ta)} 个')
        print(f'    面积占比中位 {med:.1%}   均值 {np.mean(ta):.1%}   P90 {np.percentile(ta, 90):.1%}')
        print(f'  训练标注中该类中位面积占比 {TARGET_TRAIN_AREA:.1%}')
        print(f'  -> 模型在完全陌生的物种上，复现了训练标注的框尺寸特征'
              if C_ok else '  -> 预测框尺寸与训练标注不匹配')
    else:
        print(f'  未产生任何 {TARGET} 预测框。')

    # ---------------- 结论 ----------------
    print('\n' + '=' * 70)
    print('结论')
    print('=' * 70)
    print(f'  A 类别失衡    {"成立" if A_ok else "不成立"}  '
          f'(mosaic 过表达 {over.get(TARGET, 0):.2f} 倍)')
    print(f'  B 剂量-反应   {"成立" if B_ok else "不成立"}')
    print(f'  C 框尺寸复现  {"成立" if C_ok else "不成立"}')
    print('-' * 70)
    if A_ok and B_ok and C_ok:
        print('  三条证据同时成立：捷径机制得到定量验证。')
        print('  该规则可由推理链升级为实验结论，样本量 5156，判据客观。')
    elif A_ok:
        print('  存在显著类别偏置，但绿色大区域这条具体捷径未被完整证实，')
        print('  需检查 blob_frac 是否恰当刻画了触发条件。')
    else:
        print('  假设未获支持，4.4 节的归属机制需要重写。')

    # ---------------- E13：预注册的 3x3 判定 ----------------
    vd = None
    if args.verdict == 'e13':
        vd = e13_verdict(pred_share)
        print('\n' + '=' * 70)
        print('E13 预注册判定（判据于训练完成前写定，见本文件顶部）')
        print('=' * 70)
        print(f'  black rot 预测占比  {vd["black_rot_baseline"]:.2%} -> '
              f'{vd["black_rot_share"]:.2%}   [{vd["black_rot_band"]}]')
        print(f'  mosaic    预测占比  {vd["mosaic_baseline"]:.2%} -> '
              f'{vd["mosaic_share"]:.2%}   [{vd["mosaic_band"]}]')
        if screen:
            print(f'  数据集筛查量  R = {screen["span_R"]:.2f}x   '
                  f'm(1)/m(2) = {screen["top_gap"]:.2f}x')
        print('-' * 70)
        print(f'  判定：{vd["verdict"]}')
        print(f'  {vd["reading"]}')

    res_path = out_dir / f'shortcut_{tag}.json'
    json.dump({
        'config': {'tag': tag, 'weights': weights, 'conf': CONF, 'n_images': n,
                   'classes': list(names.values())},
        'summary': {'images_with_det': img_with_det, 'total_fp_boxes': total_boxes,
                    'fp_per_image': total_boxes / n if n else 0.0},
        'class_counts': dict(box_cls),
        'pred_share': pred_share,          # 原始预测占比——不受分母操纵影响
        'prior_used': prior_used,          # 实际使用的训练占比（可能已现场重算）
        'prior_source': args.prior_from or 'hardcoded',
        'over_representation': over,
        # 筛查量与判定和假阳性数字同时落盘，供 5.2 节引用
        'screening': ({k: screen[k] for k in
                       ['span_R', 'top_gap', 'median_of_class_medians',
                        'top_median', 'second_median', 'coarse_classes',
                        'fine_classes', 'n_boxes']} if screen else None),
        'prereg_verdict': vd,
        'dose_response': curve,
        'target_pred_areas': {'n': len(ta),
                              'median': float(np.median(ta)) if ta else None,
                              'mean': float(np.mean(ta)) if ta else None},
        'criteria': {'A': A_ok, 'B': B_ok, 'C': C_ok},
    }, open(res_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(records, open(out_dir / f'per_image_{tag}.json', 'w', encoding='utf-8'),
              ensure_ascii=False)
    print(f'\n  已保存 -> {res_path}')


if __name__ == '__main__':
    main()
