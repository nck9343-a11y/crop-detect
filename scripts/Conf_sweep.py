"""
低阈值复测：区分"硬零"与"阈值效应"
====================================
E13（扩框组）在 FieldPlant 5156 张图上对 black rot 输出了 0 个框，
而该类在分布内 mAP50 = 0.990，为全模型最好的一类。
这个组合需要排除一种平凡解释：框其实存在，只是都低于 conf=0.25。

两种情形的含义完全不同：

    硬零      模型在跨物种图上根本不激活该类
              -> 粗化确实没有造出汇点，判定为证伪
    阈值效应  低置信度框大量存在，只是都低于 0.25
              -> 粗化压低了置信度而非消除倾向，结论须改写

做法：把阈值降到 0.001 复测并对照 E1，报告框数、置信度分布与框面积。
为控制耗时按固定间隔抽样（可复现，非随机）。

实测结果（STRIDE=6，860 张）：
    E1   black rot  207 个框   最高置信度 0.630   >=0.25 的 11 个
    E13  black rot   33 个框   最高置信度 0.009   >=0.25 的  0 个
即为硬零——阈值降到千分之一仍无实质输出。

用法：
    python scripts/Conf_sweep.py
    python scripts/Conf_sweep.py --stride 3 --conf 0.0005
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FIELDPLANT = Path(r'D:\dev\crop-detect\datasets\fieldplant')
IMGSZ = 640

WEIGHTS = {
    'E1':  r'D:\dev\crop-detect\runs\detect\grape_n\weights\best.pt',
    'E13': r'D:\dev\crop-detect\runs\detect\E13_expand\weights\best.pt',
}
TARGETS = ['grape black rot', 'mosaic virus disease']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=6,
                    help='每 N 张取 1 张。固定间隔而非随机，便于复现')
    ap.add_argument('--conf', type=float, default=0.001)
    a = ap.parse_args()

    imgs = sorted([p for p in FIELDPLANT.rglob('*')
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])[::a.stride]
    print(f'抽样 {len(imgs)} 张（每 {a.stride} 张取 1），conf={a.conf}\n')

    for tag, w in WEIGHTS.items():
        model = YOLO(w)
        names = model.names
        cnt, confs, areas = Counter(), defaultdict(list), defaultdict(list)

        for i, p in enumerate(imgs, 1):
            # 解码降采样倍数与 Shortcut_experiment.py 的默认值一致，否则不可比
            img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4)
            if img is None:
                continue
            r = model.predict(img, imgsz=IMGSZ, conf=a.conf,
                              verbose=False, device=0)[0]
            H, W = r.orig_shape
            for b in r.boxes:
                c = names[int(b.cls[0])]
                cnt[c] += 1
                confs[c].append(float(b.conf[0]))
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                areas[c].append((x2 - x1) * (y2 - y1) / (W * H))
            if i % 300 == 0:
                print(f'  {tag} {i}/{len(imgs)}')

        total = sum(cnt.values())
        print(f'\n===== {tag}  共 {total} 个框（conf>={a.conf}）=====')
        print(f'{"类别":<26}{"框数":>8}{"占比":>9}{"conf中位":>10}'
              f'{"conf P99":>10}{">=0.25":>9}')
        for c, n in cnt.most_common():
            cv = np.array(confs[c])
            print(f'{c:<26}{n:>8}{n / total:>8.1%}{np.median(cv):>10.3f}'
                  f'{np.percentile(cv, 99):>10.3f}{(cv >= 0.25).sum():>9}')

        for t in TARGETS:
            cv = np.array(confs.get(t, []))
            print(f'\n  [{t}]')
            if cv.size == 0:
                print(f'    conf>={a.conf} 下仍为 0 个框 —— 硬零')
                continue
            av = np.array(areas[t])
            print(f'    {cv.size} 个框   conf 中位 {np.median(cv):.4f}   '
                  f'最大 {cv.max():.4f}   >=0.25 的有 {(cv >= 0.25).sum()} 个')
            print(f'    面积占比 中位 {np.median(av):.1%}   '
                  f'P90 {np.percentile(av, 90):.1%}')
        print()


if __name__ == '__main__':
    main()
