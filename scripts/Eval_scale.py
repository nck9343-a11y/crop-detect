"""
E8: 分尺度对照评估  —— 检验"小目标瓶颈可由分辨率缓解"这一假设
================================================================
背景：
    E5 的分尺度数据显示 small 目标 AP 远低于 large（0.161 vs 0.598），
    据此提出假设：性能瓶颈位于小目标检测，增加有效像素可缓解。

    E7 采用切图推理检验该假设，结果全线下降。但事后分析表明该实验无效：
    原图尺寸（640）与模型输入尺寸相同，切片后再放大属于纯上采样，
    不产生新信息，且引入了训练时未见过的尺度，属于方法误用。

本实验：
    E3 使用 imgsz=960 训练，是"合法增加有效像素"的唯一一组实验，
    但此前只比较了整体 mAP，从未查看其分尺度表现。

    本脚本以完全相同的评估管线（同测试集、同 pycocotools、同置信度阈值）
    评估 E1 与 E3，唯一差别为训练与推理分辨率。

判据：
    E3 的 AP_small 显著高于 E1  -> 假设成立，分辨率是小目标的有效手段
    两者基本持平                -> 假设不成立，瓶颈另有原因

用法：
    python eval_scale.py
"""

import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ------------------------------------------------------------------ 配置

TEST_DIR = Path(r'D:\dev\crop-detect\datasets\grape_coco\test')
ANN = TEST_DIR / '_annotations.coco.json'
OUT_DIR = Path(r'D:\dev\crop-detect\results\runs_scale')

CONF = 0.01          # 各组统一，低阈值以完整计算 PR 曲线
CAT_OFFSET = 1       # YOLO 类别从 0 起，COCO 中 id 0 为占位符，真实类别从 1 起

MODELS = [
    ('E1  YOLO11n @640', r'D:\dev\crop-detect\runs\detect\grape_n\weights\best.pt', 640),
    ('E3  YOLO11n @960', r'D:\dev\crop-detect\runs\detect\grape_n_960-2\weights\best.pt', 960),
]


# ------------------------------------------------------------------ 评估

def predict_all(weights, imgsz, images):
    """对测试集逐张推理，输出 COCO 格式的预测列表。"""
    model = YOLO(weights)
    preds = []
    for i, im in enumerate(images, 1):
        path = TEST_DIR / im['file_name']
        if not path.exists():
            continue
        r = model.predict(str(path), imgsz=imgsz, conf=CONF,
                          verbose=False, device=0)[0]
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            preds.append({
                'image_id': int(im['id']),
                'category_id': int(box.cls[0]) + CAT_OFFSET,
                'bbox': [x1, y1, x2 - x1, y2 - y1],   # COCO 用 xywh
                'score': float(box.conf[0]),
            })
        if i % 100 == 0:
            print(f'    {i}/{len(images)}')
    return preds


def evaluate(pred_path, tag):
    gt = COCO(str(ANN))
    dt = gt.loadRes(str(pred_path))
    ev = COCOeval(gt, dt, 'bbox')
    ev.evaluate(); ev.accumulate()
    print(f'\n----- {tag} -----')
    ev.summarize()
    s = ev.stats
    return {
        'mAP50_95': float(s[0]), 'mAP50': float(s[1]),
        'AP_small': float(s[3]), 'AP_medium': float(s[4]), 'AP_large': float(s[5]),
        'AR_100': float(s[8]), 'AR_small': float(s[9]),
        'AR_medium': float(s[10]), 'AR_large': float(s[11]),
    }


# ------------------------------------------------------------------ 主流程

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco_raw = json.load(open(ANN, encoding='utf-8'))
    images = coco_raw['images']

    ws = [im['width'] for im in images]
    hs = [im['height'] for im in images]
    print('=' * 66)
    print('E8  分尺度对照评估')
    print('=' * 66)
    print(f'  测试集   {len(images)} 张图, {len(coco_raw["annotations"])} 个标注框')
    print(f'  原图尺寸 中位 {int(np.median(ws))} x {int(np.median(hs))}')
    print(f'  置信阈值 {CONF}  (各组一致)')

    results = {}
    for name, weights, imgsz in MODELS:
        if not Path(weights).exists():
            print(f'\n  [跳过] 权重不存在: {weights}')
            continue
        print(f'\n  推理中: {name}')
        preds = predict_all(weights, imgsz, images)
        pth = OUT_DIR / f'pred_{name.split()[0]}.json'
        json.dump(preds, open(pth, 'w'))
        print(f'    共 {len(preds)} 个框 -> {pth.name}')
        results[name] = evaluate(pth, name)

    if len(results) < 2:
        print('\n可用结果不足两组，无法对比。')
        return

    # ---------------- 对比表 ----------------
    names = list(results)
    a, b = results[names[0]], results[names[1]]

    print('\n' + '=' * 66)
    print('对比结果   ' + names[0] + '   vs   ' + names[1])
    print('=' * 66)
    print(f'{"指标":<14}{names[0].split()[0]:>12}{names[1].split()[0]:>12}{"变化":>12}')
    print('-' * 66)
    for label, key in [
        ('mAP50', 'mAP50'), ('mAP50-95', 'mAP50_95'),
        ('AP small', 'AP_small'), ('AP medium', 'AP_medium'), ('AP large', 'AP_large'),
        ('AR small', 'AR_small'), ('AR medium', 'AR_medium'), ('AR large', 'AR_large'),
    ]:
        d = b[key] - a[key]
        mark = '  <<<' if key == 'AP_small' else ''
        print(f'{label:<14}{a[key]:>12.4f}{b[key]:>12.4f}{d:>+12.4f}{mark}')

    print('-' * 66)
    ds = b['AP_small'] - a['AP_small']
    rel = ds / a['AP_small'] * 100 if a['AP_small'] > 0 else 0
    print(f'  小目标 AP 变化: {ds:+.4f}  (相对 {rel:+.1f}%)')
    if ds > 0.02:
        print('  -> 假设成立：提高分辨率可有效改善小目标检测。')
        print('     后续应据此反推无人机 GSD 要求与飞行高度。')
    elif ds > 0.005:
        print('  -> 方向正确但幅度有限，分辨率并非主要制约因素。')
    else:
        print('  -> 假设不成立：增加有效像素未改善小目标检测。')
        print('     瓶颈更可能在标注质量、小目标标注一致性或类别可分性。')
        print('     建议下一步：抽检最小的若干标注框，人工核查其边界是否可靠。')

    json.dump(results, open(OUT_DIR / 'scale_comparison.json', 'w'),
              ensure_ascii=False, indent=2)
    print(f'\n  已保存 -> {OUT_DIR / "scale_comparison.json"}')


if __name__ == '__main__':
    main()