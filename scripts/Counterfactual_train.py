"""
E10 训练：反事实重训
=====================
在 Counterfactual_prepare.py 构造的两份数据集上重训 YOLO11n，
超参与 E1 逐项一致（seed=0, deterministic=True, 50 轮, batch 8, imgsz 640），
唯一差别为标注本身。

    grape_cf_shrink  主实验。图像、框数、类别数与原数据集完全相同，
                     仅 mosaic 类的框面积由 43.16% 缩至 10.79%。
                     这是"标注粒度"的单变量对照。

    grape_cf_drop    辅助。剔除 mosaic 类。注意该组训练图像由 2631 降至
                     2354（少 10.5%），存在样本量混淆，故只作旁证。

    grape_cf_expand  E13，反方向。把最细的 black rot（中位 0.57%）合并为
                     整叶级外接框（中位 40.37%）。前三组都是"把粗改细"，
                     该组把细改粗，用于验证粒度与捷径关系的双向对称性。

训练完成后用 Shortcut_experiment.py 在 FieldPlant 上复测假阳性率，
并用 Eval_unified.py 复测测试集指标。

用法：
    python Counterfactual_train.py               # E10a + E10b
    python Counterfactual_train.py --only e13    # E13 扩框组
"""

import time
from pathlib import Path
from ultralytics import YOLO

# 与 runs/detect/grape_n/args.yaml 逐项对齐
BASE = dict(
    epochs=50, batch=8, imgsz=640, workers=4,
    optimizer='auto', lr0=0.01, lrf=0.01,
    momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
    seed=0, deterministic=True, pretrained=True,
    close_mosaic=10, amp=True, patience=100,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, erasing=0.4,
    project=r'D:\dev\crop-detect\runs\detect',
)

RUNS = {
    'e10': [
        ('E10b_shrink', r'D:\dev\crop-detect\datasets\grape_cf_shrink\data.yaml'),
        ('E10a_drop',   r'D:\dev\crop-detect\datasets\grape_cf_drop\data.yaml'),
    ],
    'e13': [
        ('E13_expand',  r'D:\dev\crop-detect\datasets\grape_cf_expand\data.yaml'),
    ],
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list(RUNS), default='e10')
    runs = RUNS[ap.parse_args().only]
    print('=' * 66)
    print('E10  反事实重训')
    print('=' * 66)
    for name, data in runs:
        if not Path(data).exists():
            print(f'  [跳过] 数据集不存在: {data}')
            continue
        print(f'\n>>> {name}\n    {data}')
        t0 = time.time()
        YOLO(r'D:\dev\crop-detect\weights\yolo11n.pt').train(data=data, name=name, **BASE)
        print(f'    耗时 {(time.time() - t0) / 60:.1f} 分钟')

    print('\n' + '=' * 66)
    print('下一步：')
    print('  1. 改 Shortcut_experiment.py 的 WEIGHTS 指向新权重，复测假阳性率')
    print('  2. 用 Eval_unified.py 复测测试集分布内指标')
    print('  判据：shrink 组若 mosaic 过表达倍数由 13.41x 显著回落，')
    print('        则标注粒度对该捷径具有因果作用。')
    print('=' * 66)


if __name__ == '__main__':
    main()
