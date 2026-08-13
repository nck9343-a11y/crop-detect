"""
E5: RF-DETR 微调  —— 葡萄病害 6 类
=====================================
与 E1-E3 的对齐说明（对比实验必须控制变量）：
    数据集   同一份 grape 数据（v1），同一划分 2631/329/328
    epochs   50，与 E1-E3 一致
    等效批量 batch_size=4 x grad_accum_steps=2 = 8，与 E1-E3 的 batch=8 对齐
    分辨率   RF-DETR 由模型结构固定，无法自由设定，此项无法完全对齐，需在报告中注明

用法：
    python train_rfdetr.py

产物默认写入 results/runs_rfdetr/ 目录。
"""

import os
import time
import torch
from rfdetr import RFDETRNano

DATASET = r'D:\dev\crop-detect\datasets\grape_coco'
OUTPUT = r'D:\dev\crop-detect\results\runs_rfdetr\grape_nano'


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    print('=' * 60)
    print('E5  RF-DETR Nano  微调')
    print(f'  数据集   {DATASET}')
    print(f'  输出     {OUTPUT}')
    print(f'  显卡     {torch.cuda.get_device_name(0)}')
    print('=' * 60)

    model = RFDETRNano()

    t0 = time.time()
    model.train(
        dataset_dir=DATASET,
        epochs=50,
        batch_size=4,          # DETR 系显存开销大于 YOLO，先保守
        grad_accum_steps=2,    # 4 x 2 = 8，等效批量与 E1-E3 对齐
        lr=1e-4,               # DETR 常用初始学习率
        output_dir=OUTPUT,
    )
    print(f'\n训练耗时 {(time.time() - t0) / 3600:.3f} 小时')
    print(f'权重与日志见 {OUTPUT}')


if __name__ == '__main__':
    main()