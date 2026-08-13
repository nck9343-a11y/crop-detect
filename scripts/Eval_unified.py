"""
统一评估口径 —— 重算论文表 3
=============================
问题：
    原表 3 中 E1/E2/E3 的指标来自 Ultralytics 内置评估器，
    E5 的指标来自 pycocotools。二者并非同一口径：
    以同一份 E1 权重为例，Ultralytics 给出 mAP50=0.845，
    pycocotools 给出 0.8357，相差 0.9 个百分点——
    而全文核心结论建立在"四组极差仅 1.1 个百分点"之上。
    评估器差异与待论证的差异同量级，该比较不成立。

    此外原表遗漏了 E4（RT-DETR-l）。该组因 batch=4 与其余组的
    batch=8 不一致而未纳入，但编号跳号会引出疑问，且其结果
    （32M 参数仍打平）恰好支持主论点，故此处一并评估并如实标注。

本脚本：
    五组模型、同一测试集、同一 pycocotools 评估器、同一置信度阈值，
    输出可直接替换论文表 3 的完整结果，并附分尺度与逐类别指标。

用法：
    python Eval_unified.py
"""

import json
import contextlib
import io
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

TEST_DIR = Path(r'D:\dev\crop-detect\datasets\grape_coco\test')
ANN = TEST_DIR / '_annotations.coco.json'
OUT_DIR = Path(r'D:\dev\crop-detect\results\runs_unified')

CONF = 0.01        # 低阈值以完整计算 PR 曲线，各组一致
CAT_OFFSET = 1     # YOLO 类别从 0 起；COCO 中 id 0 为占位符，真实类别从 1 起

R = r'D:\dev\crop-detect\runs\detect'
MODELS = [
    # 编号,  名称,             类型,      权重,                              分辨率, 批量
    ('E1', 'YOLO11n @640',   'yolo',   rf'{R}\grape_n\weights\best.pt',      640, 8),
    ('E2', 'YOLO11s @640',   'yolo',   rf'{R}\grape_s\weights\best.pt',      640, 8),
    ('E3', 'YOLO11n @960',   'yolo',   rf'{R}\grape_n_960-2\weights\best.pt', 960, 8),
    ('E4', 'RT-DETR-l @640', 'yolo',   rf'{R}\grape_rtdetr\weights\best.pt',  640, 4),
    ('E5', 'RF-DETR-N @384', 'rfdetr',
     r'D:\dev\crop-detect\results\runs_rfdetr\grape_nano\checkpoint_best_regular.pth', 384, 8),
]


def predict_yolo(weights, imgsz, images):
    """Ultralytics 系模型（YOLO11 与 RT-DETR 共用同一接口）。"""
    from ultralytics import YOLO
    model = YOLO(weights)
    preds = []
    for i, im in enumerate(images, 1):
        p = TEST_DIR / im['file_name']
        if not p.exists():
            continue
        r = model.predict(str(p), imgsz=imgsz, conf=CONF, verbose=False, device=0)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            preds.append({'image_id': int(im['id']),
                          'category_id': int(b.cls[0]) + CAT_OFFSET,
                          'bbox': [x1, y1, x2 - x1, y2 - y1],
                          'score': float(b.conf[0])})
        if i % 100 == 0:
            print(f'    {i}/{len(images)}')
    return preds


def predict_rfdetr(ckpt, images):
    """RF-DETR 直接输出 COCO 类别 id，无需偏移（已比对高置信度预测数与真值验证）。"""
    from PIL import Image
    from rfdetr import RFDETRNano
    model = RFDETRNano(pretrain_weights=ckpt)
    preds = []
    for i, im in enumerate(images, 1):
        p = TEST_DIR / im['file_name']
        if not p.exists():
            continue
        det = model.predict(Image.open(p), threshold=0.001)
        for box, conf, cid in zip(det.xyxy, det.confidence, det.class_id):
            x1, y1, x2, y2 = [float(v) for v in box]
            preds.append({'image_id': int(im['id']), 'category_id': int(cid),
                          'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(conf)})
        if i % 100 == 0:
            print(f'    {i}/{len(images)}')
    return preds


def evaluate(pred_path, cats):
    """静默运行 COCOeval，返回整体、分尺度与逐类别指标。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gt = COCO(str(ANN))
        dt = gt.loadRes(str(pred_path))
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats

    # precision 的 K 维按 ev.params.catIds 排列，其中包含 COCO 里 id=0 的
    # 占位类（无任何标注）。必须以 catIds 为准逐一对应，
    # 若改用过滤后的类别字典去 enumerate，整列会错位一格。
    per_class = {}
    prec = ev.eval['precision']          # [T, R, K, A, M]
    for k, cid in enumerate(ev.params.catIds):
        if cid not in cats:              # 跳过占位类
            continue
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        per_class[cats[cid]] = float(np.mean(p)) if p.size else float('nan')

    return {'mAP50_95': float(s[0]), 'mAP50': float(s[1]), 'mAP75': float(s[2]),
            'AP_small': float(s[3]), 'AP_medium': float(s[4]), 'AP_large': float(s[5]),
            'AR_100': float(s[8]), 'AR_small': float(s[9]),
            'AR_medium': float(s[10]), 'AR_large': float(s[11]),
            'per_class': per_class}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.load(open(ANN, encoding='utf-8'))
    images = raw['images']
    cats = {c['id']: c['name'] for c in raw['categories'] if c['id'] != 0}

    print('=' * 78)
    print('统一评估口径  ——  五组模型 / 同一测试集 / pycocotools')
    print('=' * 78)
    print(f'  测试集 {len(images)} 张, {len(raw["annotations"])} 个标注框, 阈值 conf={CONF}')

    results = {}
    for eid, name, kind, w, imgsz, batch in MODELS:
        if not Path(w).exists():
            print(f'\n  [跳过] {eid} 权重不存在: {w}')
            continue
        pth = OUT_DIR / f'pred_{eid}.json'
        if pth.exists():
            # 预测结果已存在则直接复用，便于改动评估逻辑后快速重算，
            # 无需重复推理（推理约占本脚本全部耗时的 95%）。
            print(f'\n  复用已有预测 {eid}  {name}')
            preds = json.load(open(pth))
        else:
            print(f'\n  推理 {eid}  {name}')
            preds = (predict_rfdetr(w, images) if kind == 'rfdetr'
                     else predict_yolo(w, imgsz, images))
            json.dump(preds, open(pth, 'w'))
        r = evaluate(pth, cats)
        r.update({'name': name, 'imgsz': imgsz, 'batch': batch})
        results[eid] = r
        print(f'    {len(preds)} 个框  ->  mAP50 {r["mAP50"]:.4f}  '
              f'mAP50-95 {r["mAP50_95"]:.4f}')

    if not results:
        print('\n无可用结果。')
        return

    # ---------------- 表 3（统一口径） ----------------
    print('\n' + '=' * 78)
    print('表 3（统一 pycocotools 口径）')
    print('=' * 78)
    print(f'{"编号":<5}{"模型":<18}{"分辨率":>7}{"批量":>6}'
          f'{"mAP50":>9}{"mAP50-95":>10}{"AP_s":>8}{"AP_m":>8}{"AP_l":>8}')
    print('-' * 78)
    for eid in sorted(results):
        r = results[eid]
        flag = ' *' if r['batch'] != 8 else ''
        print(f'{eid:<5}{r["name"]:<18}{r["imgsz"]:>7}{r["batch"]:>6}'
              f'{r["mAP50"]:>9.4f}{r["mAP50_95"]:>10.4f}'
              f'{r["AP_small"]:>8.4f}{r["AP_medium"]:>8.4f}{r["AP_large"]:>8.4f}{flag}')
    print('-' * 78)
    m50 = [r['mAP50'] for r in results.values()]
    m5095 = [r['mAP50_95'] for r in results.values()]
    print(f'  mAP50    极差 {max(m50) - min(m50):.4f}  '
          f'({min(m50):.4f} ~ {max(m50):.4f})')
    print(f'  mAP50-95 极差 {max(m5095) - min(m5095):.4f}  '
          f'({min(m5095):.4f} ~ {max(m5095):.4f})')
    print('  * 该组等效批量与其余组不一致，非严格单变量对照')

    # ---------------- 分尺度（全部组，补 §5.3 的局限） ----------------
    print('\n' + '=' * 78)
    print('分尺度性能（原论文仅在 E5 上给出，此处补全全部架构）')
    print('=' * 78)
    print(f'{"编号":<5}{"AP_small":>10}{"AP_medium":>10}{"AP_large":>10}{"large/small":>13}')
    print('-' * 78)
    for eid in sorted(results):
        r = results[eid]
        ratio = r['AP_large'] / r['AP_small'] if r['AP_small'] > 0 else float('nan')
        print(f'{eid:<5}{r["AP_small"]:>10.4f}{r["AP_medium"]:>10.4f}'
              f'{r["AP_large"]:>10.4f}{ratio:>12.2f}x')

    # ---------------- 逐类别 ----------------
    print('\n' + '=' * 78)
    print('逐类别 AP50-95')
    print('=' * 78)
    names = sorted(cats.values())
    print(f'{"类别":<26}' + ''.join(f'{e:>10}' for e in sorted(results)))
    print('-' * 78)
    for n in names:
        print(f'{n:<26}' + ''.join(
            f'{results[e]["per_class"].get(n, float("nan")):>10.4f}'
            for e in sorted(results)))

    json.dump(results, open(OUT_DIR / 'table3_unified.json', 'w'),
              ensure_ascii=False, indent=2)
    print(f'\n  已保存 -> {OUT_DIR / "table3_unified.json"}')


if __name__ == '__main__':
    main()
