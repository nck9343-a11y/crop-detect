"""
E5 评估：RF-DETR 在 test 集上的表现
=====================================
与 E1-E3 口径对齐：一律报告 test 集指标（训练与调参全程未使用）。

用法：
    python eval_rfdetr.py

注意：RF-DETR 的输入分辨率由模型结构决定（384/544），无法像 YOLO 那样
自由设定为 640，此项无法与 E1-E3 完全对齐，需在报告中注明。
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from rfdetr import RFDETRNano

CKPT = r'D:\dev\crop-detect\results\runs_rfdetr\grape_nano\checkpoint_best_regular.pth'
TEST_DIR = Path(r'D:\dev\crop-detect\datasets\grape_coco\test')
ANN = TEST_DIR / '_annotations.coco.json'
OUT = Path(r'D:\dev\crop-detect\results\runs_rfdetr\test_predictions.json')


def main():
    coco = json.load(open(ANN, encoding='utf-8'))
    images = coco['images']
    cats = {c['id']: c['name'] for c in coco['categories']}
    print(f'test 集: {len(images)} 张图, {len(coco["annotations"])} 个标注框')
    print(f'类别: {list(cats.values())}\n')

    model = RFDETRNano(pretrain_weights=CKPT)

    results = []
    for i, im in enumerate(images, 1):
        path = TEST_DIR / im['file_name']
        if not path.exists():
            continue
        det = model.predict(Image.open(path), threshold=0.001)
        for box, conf, cid in zip(det.xyxy, det.confidence, det.class_id):
            x1, y1, x2, y2 = [float(v) for v in box]
            results.append({
                'image_id': im['id'],
                'category_id': int(cid),
                'bbox': [x1, y1, x2 - x1, y2 - y1],   # COCO 用 xywh
                'score': float(conf),
            })
        if i % 50 == 0:
            print(f'  已处理 {i}/{len(images)}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, 'w'), ensure_ascii=False)
    print(f'\n预测结果已保存 -> {OUT}  (共 {len(results)} 个框)')

    # ---- 用 pycocotools 计算标准 COCO 指标 ----
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(ANN))
    dt = gt.loadRes(str(OUT))
    ev = COCOeval(gt, dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    print('\n' + '=' * 58)
    print('对齐 E1-E3 的口径')
    print('=' * 58)
    print(f'  mAP50     = {ev.stats[1]:.4f}')
    print(f'  mAP50-95  = {ev.stats[0]:.4f}')

    # ---- 逐类别 AP ----
    print('\n各类别 AP50-95:')
    precisions = ev.eval['precision']          # [T, R, K, A, M]
    for k, cid in enumerate(sorted(cats)):
        p = precisions[:, :, k, 0, -1]
        p = p[p > -1]
        ap = np.mean(p) if p.size else float('nan')
        print(f'  {cats[cid]:<26} {ap:.4f}')


if __name__ == '__main__':
    main()