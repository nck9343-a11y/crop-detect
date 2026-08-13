"""
E11 多种子重复
===============
论文核心论证之一是"组间 1 个百分点的差异属评估噪声"。原稿据以支撑该判断的
是一处观察——E2、E3 相对 E1 的变化在验证集与测试集上方向相反。该观察有力，
但仍是间接证据：它说明差异不稳定，却没有给出"同一配置重复运行会波动多少"。

本实验直接测定该波动：同一配置换随机种子重复训练，报告均值与标准差。
若同配置的标准差与组间差异同量级，则"1 个百分点属噪声"这一判断即有直接依据。

设计：
    E1 (YOLO11n@640) 的 seed=0 已存在（runs/detect/grape_n），此处补 seed 1、2。
    E2、E3 同理各补两个种子。其余超参与原组逐项一致。

    共 6 次训练，每次约 30 分钟。按 E1 -> E2 -> E3 顺序执行，
    中途中断亦可用已完成的部分（E1 的三个种子是最关键的一组，故排在最前）。

用法：
    python Seed_repeat.py            # 全部 6 组
    python Seed_repeat.py --only E1  # 只跑 E1 的两个种子
"""

import argparse
import time
from pathlib import Path
from ultralytics import YOLO

DATA = r'D:\dev\crop-detect\datasets\grape_public\data.yaml'

# 与 runs/detect/grape_n/args.yaml 逐项对齐，仅 seed 与 imgsz/model 按组变化
BASE = dict(
    # workers 由 4 降为 2：每个 worker 进程都要 import torch 并预留约 1 GB
    # 提交内存，本机 16 GB 内存下多进程叠加会触发 WinError 1455（页面文件不足）。
    # 训练本身是 GPU 瓶颈，降低 workers 对速度影响很小。
    data=DATA, epochs=50, batch=8, workers=2,
    optimizer='auto', lr0=0.01, lrf=0.01,
    momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
    deterministic=True, pretrained=True,
    close_mosaic=10, amp=True, patience=100,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, erasing=0.4,
    project=r'D:\dev\crop-detect\runs\detect',
)

# (组名, 预训练权重, imgsz, 已存在的种子)
GROUPS = [
    ('E1', r'D:\dev\crop-detect\weights\yolo11n.pt', 640, 0),
    ('E2', r'D:\dev\crop-detect\weights\yolo11s.pt', 640, 0),
    ('E3', r'D:\dev\crop-detect\weights\yolo11n.pt', 960, 0),
]
NEW_SEEDS = [1, 2]
# 第二批：n=3 时 E1 vs E3 的差异 p 落在 0.065-0.109，方向一致但检验力不足。
# 按现有均值与标准差外推，n=5 可使 AP_small 与 AP_medium 达到 p<0.05。
EXTRA_SEEDS = [3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='只跑指定组，如 E1')
    ap.add_argument('--extra', action='store_true',
                    help='跑第二批种子（3、4），把每组补到 n=5')
    a = ap.parse_args()

    seeds = EXTRA_SEEDS if a.extra else NEW_SEEDS
    todo = [g for g in GROUPS if a.only is None or g[0] == a.only]
    print('=' * 66)
    print('E11  多种子重复')
    print(f"  待训 {len(todo) * len(seeds)} 组: {[g[0] for g in todo]} x seed{seeds}")
    print('=' * 66)

    for name, weights, imgsz, _ in todo:
        for seed in seeds:
            run = f'{name}_seed{seed}'
            if (Path(BASE['project']) / run / 'results.csv').exists():
                print(f'\n  [跳过] {run} 已存在')
                continue
            print(f'\n>>> {run}   {weights} @ {imgsz}, seed={seed}')
            t0 = time.time()
            YOLO(weights).train(name=run, imgsz=imgsz, seed=seed, **BASE)
            print(f'    耗时 {(time.time() - t0) / 60:.1f} 分钟')

    print('\n' + '=' * 66)
    print('训练完毕。用 Eval_unified.py 的口径评估各种子，再算均值与标准差。')
    print('=' * 66)


if __name__ == '__main__':
    main()
