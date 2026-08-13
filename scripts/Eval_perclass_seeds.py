"""
逐类别 AP 的种子间波动
======================
表 10 用"其余五类变动均在 ±5% 以内"论证反事实操作的特异性，
但反事实重训只跑了一个种子（seed=0），±5% 究竟是零效应还是噪声，
原先没有参照。本脚本给出该参照。

不需要重新训练：E1 的五个种子已在 results/runs_unified/pred_E1_seed{0..4}.json
留有预测结果，直接以同一 pycocotools 口径复算逐类别 AP 即可。

用法：
    python Eval_perclass_seeds.py
"""

import json
import contextlib
import io
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANN = Path(r'D:\dev\crop-detect\datasets\grape_coco\test\_annotations.coco.json')
OUT_DIR = Path(r'D:\dev\crop-detect\results\runs_unified')
PREDS = [OUT_DIR / f'pred_E1_seed{i}.json' for i in range(5)]


def per_class_ap(pred_path, cats):
    """与 Eval_unified.evaluate 完全一致的逐类别 AP50-95 计算。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gt = COCO(str(ANN))
        dt = gt.loadRes(str(pred_path))
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()

    out = {}
    prec = ev.eval['precision']          # [T, R, K, A, M]
    for k, cid in enumerate(ev.params.catIds):
        if cid not in cats:              # 跳过 COCO 的 id=0 占位类
            continue
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        out[cats[cid]] = float(np.mean(p)) if p.size else float('nan')
    return out


def main():
    raw = json.load(open(ANN, encoding='utf-8'))
    cats = {c['id']: c['name'] for c in raw['categories'] if c['id'] != 0}
    names = sorted(cats.values())

    rows = []
    for p in PREDS:
        if not p.exists():
            print(f'  [跳过] {p.name} 不存在')
            continue
        rows.append(per_class_ap(p, cats))
        print(f'  已算 {p.name}')

    print('\n' + '=' * 86)
    print(f'逐类别 AP50-95 的种子间波动（YOLO11n@640, n={len(rows)}）')
    print('=' * 86)
    print(f'{"类别":<26}{"均值":>9}{"标准差":>9}{"极差":>9}'
          f'{"相对标准差":>12}{"相对极差":>11}')
    print('-' * 86)

    summary = {}
    for n in names:
        v = np.array([r[n] for r in rows])
        rng = float(v.max() - v.min())
        summary[n] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)),
                      'range': rng,
                      'rel_std': float(v.std(ddof=1) / v.mean()),
                      'rel_range': rng / float(v.mean()),
                      'values': [float(x) for x in v]}
        s = summary[n]
        print(f'{n:<26}{s["mean"]:>9.4f}{s["std"]:>9.4f}{s["range"]:>9.4f}'
              f'{s["rel_std"] * 100:>11.1f}%{s["rel_range"] * 100:>10.1f}%')
    print('-' * 86)
    worst = max(summary.values(), key=lambda s: s['rel_range'])
    print(f'  最大相对极差 {worst["rel_range"] * 100:.1f}%  '
          f'——单种子逐类别 AP 的比较需以此为噪声下界')

    json.dump(summary, open(OUT_DIR / 'perclass_seed_variance.json', 'w'),
              ensure_ascii=False, indent=2)
    print(f'\n  已保存 -> {OUT_DIR / "perclass_seed_variance.json"}')


if __name__ == '__main__':
    main()
