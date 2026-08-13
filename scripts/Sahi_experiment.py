"""
E7: 切图推理（SAHI）验证实验
================================
目的：验证"性能瓶颈位于小目标检测"这一假设。

原理：
    E5 的分尺度评估显示 small 目标 AP 仅 0.161，large 目标达 0.598。
    但这是"观察"，不是"验证"——小目标表现差，也可能是别的原因。

    切图推理把大图切成若干小块分别送入模型再合并结果。
    对模型而言，每个小块被放大到输入尺寸，等效于把小目标放大了。

    若假设成立 -> 小目标 AP 应显著上升
    若假设不成立 -> 小目标 AP 基本不动，需另找原因

实验设计（单一变量）：
    同一模型权重、同一测试集、同一评估器（pycocotools）、同一置信度阈值，
    唯一差别是标准推理 vs 切图推理。

依赖：
    pip install sahi

用法：
    python sahi_experiment.py
"""

import json
import time
import numpy as np
from pathlib import Path
from PIL import Image

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ------------------------------------------------------------------ 配置

WEIGHTS = r'D:\dev\crop-detect\runs\detect\grape_n\weights\best.pt'   # E1 的权重
TEST_DIR = Path(r'D:\dev\crop-detect\datasets\grape_coco\test')
ANN = TEST_DIR / '_annotations.coco.json'
OUT_DIR = Path(r'D:\dev\crop-detect\results\runs_sahi')

CONF = 0.01          # 两组统一，低阈值以便完整计算 PR 曲线
SLICE = 320          # 切片边长，需明显小于原图边长才有意义
OVERLAP = 0.2        # 切片重叠率，防止目标被切断

# YOLO 类别编号从 0 开始，COCO 标注中 id 0 是 supercategory 占位，
# 真实类别从 1 开始，故偏移量为 +1。脚本会自行校验该假设。
CAT_OFFSET = 1


# ------------------------------------------------------------------ 工具

def report_image_sizes(images):
    """切图推理是否有意义，取决于原图是否明显大于切片尺寸。"""
    ws = [im['width'] for im in images]
    hs = [im['height'] for im in images]
    print(f'  图像尺寸: 宽 {min(ws)}~{max(ws)} (中位 {int(np.median(ws))}), '
          f'高 {min(hs)}~{max(hs)} (中位 {int(np.median(hs))})')
    if max(ws) <= SLICE or max(hs) <= SLICE:
        print(f'  [警告] 原图尺寸不大于切片尺寸 {SLICE}，切图推理不会产生效果。')
        print(f'         请调小 SLICE 后重试。')
    return int(np.median(ws)), int(np.median(hs))


def verify_category_mapping(model, cats):
    """核对 YOLO 的类别顺序与 COCO 的类别 id 是否按 CAT_OFFSET 对齐。"""
    yolo_names = model.category_mapping   # {'0': 'name', ...}
    print('\n  类别映射校验:')
    ok = True
    for k in sorted(yolo_names, key=int):
        yolo_id = int(k)
        coco_id = yolo_id + CAT_OFFSET
        coco_name = cats.get(coco_id, '<缺失>')
        flag = '✓' if yolo_names[k] == coco_name else '✗'
        if flag == '✗':
            ok = False
        print(f'    YOLO {yolo_id} "{yolo_names[k]}"  ->  COCO {coco_id} "{coco_name}"  {flag}')
    if not ok:
        raise SystemExit('类别映射不一致，请检查 CAT_OFFSET 设置。')
    return ok


def to_coco(result, image_id):
    """
    把 SAHI 的预测结果转成 COCO 评估所需格式，并做类别编号偏移。

    注意：SAHI 返回的坐标与置信度为 numpy.float32，标准 json 模块无法序列化，
    因此此处显式转换为 Python 原生类型，只保留 pycocotools 需要的四个字段。
    """
    out = []
    for d in result.to_coco_predictions(image_id=image_id):
        out.append({
            'image_id': int(image_id),
            'category_id': int(d['category_id']) + CAT_OFFSET,
            'bbox': [float(v) for v in d['bbox']],
            'score': float(d['score']),
        })
    return out


def evaluate(gt_path, pred_path, tag):
    """用 pycocotools 计算标准 COCO 指标，返回关键数值。"""
    gt = COCO(str(gt_path))
    dt = gt.loadRes(str(pred_path))
    ev = COCOeval(gt, dt, 'bbox')
    ev.evaluate(); ev.accumulate()
    print(f'\n----- {tag} -----')
    ev.summarize()
    s = ev.stats
    return {
        'mAP50_95': float(s[0]), 'mAP50': float(s[1]), 'mAP75': float(s[2]),
        'AP_small': float(s[3]), 'AP_medium': float(s[4]), 'AP_large': float(s[5]),
        'AR_100': float(s[8]),
        'AR_small': float(s[9]), 'AR_medium': float(s[10]), 'AR_large': float(s[11]),
    }


# ------------------------------------------------------------------ 主流程

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coco_raw = json.load(open(ANN, encoding='utf-8'))
    images = coco_raw['images']
    cats = {c['id']: c['name'] for c in coco_raw['categories']}

    print('=' * 64)
    print('E7  切图推理验证实验')
    print('=' * 64)
    print(f'  权重     {Path(WEIGHTS).name}')
    print(f'  测试集   {len(images)} 张图, {len(coco_raw["annotations"])} 个标注框')
    print(f'  切片     {SLICE} x {SLICE}, 重叠 {OVERLAP:.0%}')
    print(f'  置信阈值 {CONF}  (两组一致)')
    report_image_sizes(images)

    model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=WEIGHTS,
        confidence_threshold=CONF,
        device='cuda:0',
    )
    verify_category_mapping(model, cats)

    baseline, sliced = [], []
    t_base = t_slice = 0.0

    print(f'\n  开始推理 ...')
    for i, im in enumerate(images, 1):
        path = TEST_DIR / im['file_name']
        if not path.exists():
            continue
        img = Image.open(path).convert('RGB')

        # --- A 组：标准推理（整图一次前向） ---
        t0 = time.time()
        r = get_prediction(img, model)
        t_base += time.time() - t0
        baseline += to_coco(r, im['id'])

        # --- B 组：切图推理（切片 + 整图，结果合并去重） ---
        t0 = time.time()
        r = get_sliced_prediction(
            img, model,
            slice_height=SLICE, slice_width=SLICE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
            perform_standard_pred=True,   # 同时保留整图预测，避免漏掉大目标
            verbose=0,
        )
        t_slice += time.time() - t0
        sliced += to_coco(r, im['id'])

        if i % 50 == 0:
            print(f'    {i}/{len(images)}')

    p_base = OUT_DIR / 'pred_baseline.json'
    p_slice = OUT_DIR / 'pred_sliced.json'
    json.dump(baseline, open(p_base, 'w'))
    json.dump(sliced, open(p_slice, 'w'))
    print(f'\n  标准推理 {len(baseline)} 个框, 耗时 {t_base:.1f}s '
          f'({t_base / len(images) * 1000:.1f} ms/图)')
    print(f'  切图推理 {len(sliced)} 个框, 耗时 {t_slice:.1f}s '
          f'({t_slice / len(images) * 1000:.1f} ms/图)')

    a = evaluate(ANN, p_base, 'A 标准推理')
    b = evaluate(ANN, p_slice, 'B 切图推理')

    # ---------------- 对比表 ----------------
    print('\n' + '=' * 64)
    print('对比结果')
    print('=' * 64)
    print(f'{"指标":<14}{"标准推理":>12}{"切图推理":>12}{"变化":>12}')
    print('-' * 64)
    rows = [
        ('mAP50', 'mAP50'), ('mAP50-95', 'mAP50_95'),
        ('AP small', 'AP_small'), ('AP medium', 'AP_medium'), ('AP large', 'AP_large'),
        ('AR@100', 'AR_100'), ('AR small', 'AR_small'),
    ]
    for label, key in rows:
        d = b[key] - a[key]
        mark = '  <<<' if key == 'AP_small' else ''
        print(f'{label:<14}{a[key]:>12.4f}{b[key]:>12.4f}{d:>+12.4f}{mark}')

    print('-' * 64)
    ds = b['AP_small'] - a['AP_small']
    if ds > 0.03:
        print(f'  结论: 小目标 AP 提升 {ds:+.4f}，假设成立——'
              f'瓶颈确实位于小目标检测。')
        print(f'        切图推理可作为可落地的优化手段。')
    elif ds > 0.005:
        print(f'  结论: 小目标 AP 提升 {ds:+.4f}，方向正确但幅度有限，'
              f'需调整切片尺寸后复测。')
    else:
        print(f'  结论: 小目标 AP 变化 {ds:+.4f}，假设不成立——'
              f'瓶颈另有原因，需重新排查。')
        print(f'        可能方向: 标注质量、目标本身可分性、类别定义。')

    json.dump({'config': {'slice': SLICE, 'overlap': OVERLAP, 'conf': CONF},
               'baseline': a, 'sliced': b},
              open(OUT_DIR / 'comparison.json', 'w'), ensure_ascii=False, indent=2)
    print(f'\n  结果已保存 -> {OUT_DIR / "comparison.json"}')


if __name__ == '__main__':
    main()