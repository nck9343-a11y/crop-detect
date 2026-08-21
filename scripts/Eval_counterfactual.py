"""
反事实组的分布内评估 —— 统一到 pycocotools 口径
================================================
问题：
    表 10 原先的逐类别 AP 取自 Ultralytics 内置评估器，而表 3、表 7
    取自 pycocotools。同一份 E1 权重在两套评估器下相差约 1-2 个百分点
    （2.3 节已记录该差值），致使表 7 与表 10 的 E1 列数字对不上。
    论文既已声明统一口径，此处将反事实组一并纳入同一评估器重算。

    反事实数据集只有 YOLO 格式标注，故先由 YOLO 标签构造 COCO 标注，
    图像 id 与类别 id 沿用 grape_coco/test，保证与表 7 逐项可比。

三个评估配置：
    E1 / 原标注 GT       表 7 的基线，直接复用 results/runs_unified/pred_E1.json
    CF / CF GT           反事实主组，各自在其标注体系下评估
    E1 / CF GT           诊断项。以原标注训练的模型在改动后 GT 上的表现，
                         用于观察"改标注使任务变难/变易"这一效度威胁的量级。
                         注意该配置存在标注体系错配，不作为论文结果报告。

两个反事实组：
    shrink (E10b)  mosaic 的框缩至中心 25% 面积
    expand (E13)   black rot 每图的病斑框合并为整叶级外接框

    expand 组读数须留意：该类 GT 由 4237 个小框变为 476 个大框，
    目标数减少而单目标变大，两者都使 AP 上升，与是否学到症状特征无关。
    故该组的分布内 AP 只用于说明"模型确实按新标注体系学到了东西"，
    不作为捷径论证的证据——捷径证据来自 FieldPlant 上的假阳性分布。

用法：
    python Eval_counterfactual.py                  # shrink 组（E10b）
    python Eval_counterfactual.py --group expand   # expand 组（E13）
"""

import json
import contextlib
import io
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(r'D:\dev\crop-detect')
BASE_ANN = ROOT / 'datasets' / 'grape_coco' / 'test' / '_annotations.coco.json'
OUT_DIR = ROOT / 'results' / 'runs_unified'   # 结果目录后来挪到 results/ 下

GROUPS = {
    'shrink': dict(tag='E10b', dataset='grape_cf_shrink',
                   run='E10b_shrink', changed='mosaic virus disease'),
    'expand': dict(tag='E13',  dataset='grape_cf_expand',
                   run='E13_expand',  changed='grape black rot'),
}

CONF = 0.01        # 与 Eval_unified.py 一致
CAT_OFFSET = 1     # YOLO 类别从 0 起，COCO 中 id 0 为占位类


def build_cf_ann(out_path, cf_test):
    """由 grape_cf_shrink 的 YOLO 标签构造 COCO 标注。

    图像 id、file_name、类别 id 全部沿用原 COCO 标注，只替换标注框，
    使新旧两份 GT 之间除框的尺寸外无任何差异。
    """
    raw = json.load(open(BASE_ANN, encoding='utf-8'))
    out = {k: raw[k] for k in raw if k != 'annotations'}

    anns, aid = [], 1
    missing = 0
    for im in raw['images']:
        lbl = cf_test / 'labels' / (Path(im['file_name']).stem + '.txt')
        if not lbl.exists():
            missing += 1
            continue
        W, H = im['width'], im['height']
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cid, cx, cy, w, h = int(p[0]), *[float(v) for v in p[1:5]]
            bw, bh = w * W, h * H
            anns.append({'id': aid, 'image_id': im['id'],
                         'category_id': cid + CAT_OFFSET,
                         'bbox': [(cx * W) - bw / 2, (cy * H) - bh / 2, bw, bh],
                         'area': bw * bh, 'iscrowd': 0})
            aid += 1
    out['annotations'] = anns
    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'  构造 CF 标注：{len(raw["images"]) - missing} 张图，{len(anns)} 个框'
          f'（原 {len(raw["annotations"])} 个）')
    if missing:
        print(f'  [注意] {missing} 张图在 CF 测试集中无标签文件')
    return out_path


def predict(weights, images, img_dir, out_path):
    if out_path.exists():
        print(f'  复用已有预测 {out_path.name}')
        return out_path
    from ultralytics import YOLO
    model = YOLO(str(weights))
    preds = []
    for i, im in enumerate(images, 1):
        p = img_dir / im['file_name']
        if not p.exists():
            continue
        r = model.predict(str(p), imgsz=640, conf=CONF, verbose=False, device=0)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            preds.append({'image_id': int(im['id']),
                          'category_id': int(b.cls[0]) + CAT_OFFSET,
                          'bbox': [x1, y1, x2 - x1, y2 - y1],
                          'score': float(b.conf[0])})
        if i % 100 == 0:
            print(f'    {i}/{len(images)}')
    json.dump(preds, open(out_path, 'w', encoding='utf-8'))
    print(f'  {len(preds)} 个预测框 -> {out_path.name}')
    return out_path


def evaluate(ann_path, pred_path, cats):
    """与 Eval_unified.evaluate 完全一致。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gt = COCO(str(ann_path))
        dt = gt.loadRes(str(pred_path))
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats

    per_class = {}
    prec = ev.eval['precision']
    for k, cid in enumerate(ev.params.catIds):
        if cid not in cats:
            continue
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        per_class[cats[cid]] = float(np.mean(p)) if p.size else float('nan')
    return {'mAP50_95': float(s[0]), 'mAP50': float(s[1]), 'per_class': per_class}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=list(GROUPS), default='shrink')
    group = ap.parse_args().group
    g = GROUPS[group]

    cf_test = ROOT / 'datasets' / g['dataset'] / 'test'
    cf_w = ROOT / 'runs' / 'detect' / g['run'] / 'weights' / 'best.pt'
    tag = g['tag']

    raw = json.load(open(BASE_ANN, encoding='utf-8'))
    cats = {c['id']: c['name'] for c in raw['categories'] if c['id'] != 0}
    names = sorted(cats.values())

    print('=' * 78)
    print(f'反事实组分布内评估（统一 pycocotools 口径）  组别 {tag} / {group}')
    print(f'  被改动的类别：{g["changed"]}')
    print('=' * 78)

    cf_ann = build_cf_ann(OUT_DIR / f'ann_cf_{group}_test.json', cf_test)

    pred_e1 = OUT_DIR / 'pred_E1.json'
    pred_cf = predict(cf_w, raw['images'], cf_test / 'images',
                      OUT_DIR / f'pred_{tag}_{group}.json')

    runs = {
        'E1 / 原标注 GT':            evaluate(BASE_ANN, pred_e1, cats),
        f'{group} / {group} GT':     evaluate(cf_ann, pred_cf, cats),
        f'E1 / {group} GT':          evaluate(cf_ann, pred_e1, cats),
    }

    print('\n' + '=' * 78)
    print('逐类别 AP50-95')
    print('=' * 78)
    print(f'{"类别":<26}' + ''.join(f'{k:>20}' for k in runs))
    print('-' * 78)
    for n in names:
        print(f'{n:<26}' + ''.join(f'{r["per_class"][n]:>20.4f}'
                                   for r in runs.values()))
    print('-' * 78)
    for k, r in runs.items():
        print(f'{k:<26}  mAP50 {r["mAP50"]:.4f}   mAP50-95 {r["mAP50_95"]:.4f}')

    print('\n  表 10 用（相对基线 E1 / 原标注 GT 的变化）')
    a = runs['E1 / 原标注 GT']['per_class']
    b = runs[f'{group} / {group} GT']['per_class']
    for n in names:
        mark = '  <- 被改动' if n == g['changed'] else ''
        print(f'    {n:<26}{a[n]:>9.4f}{b[n]:>9.4f}'
              f'{(b[n] / a[n] - 1) * 100:>9.1f}%{mark}')
    if group == 'expand':
        print('\n  注意：black rot 的 GT 由 4237 个小框变为 476 个大框，'
              '目标变少变大本身即抬高 AP，')
        print('        该行不能作为"学到症状特征"的证据，'
              '捷径证据在 FieldPlant 的假阳性分布。')

    # shrink 组沿用原文件名（论文已引用），expand 组另存
    out_path = OUT_DIR / ('counterfactual_unified.json' if group == 'shrink'
                          else f'counterfactual_unified_{tag}.json')
    json.dump({k: v for k, v in runs.items()}, open(out_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'\n  已保存 -> {out_path}')


if __name__ == '__main__':
    main()
