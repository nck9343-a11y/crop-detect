import random, shutil
from pathlib import Path

SRC = Path(r'D:\dev\crop-detect\datasets\fieldplant')
DST = Path(r'D:\dev\crop-detect\datasets\tomato3')

# 原始编号 -> 新编号。索引来自 data.yaml 的 names 列表顺序
OLD2NEW = {21: 0, 23: 1, 24: 2}
NEW_NAMES = ['Tomato_Brown_Spots', 'Tomato_blight_leaf', 'Tomato_healthy']

random.seed(0)  # 固定随机种子，保证结果可复现

# 1. 扫描，挑出含目标类别的图片
pairs = []
for lbl in (SRC / 'train' / 'labels').glob('*.txt'):
    kept = []
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        if cid in OLD2NEW:
            kept.append(' '.join([str(OLD2NEW[cid])] + parts[1:]))
    if not kept:
        continue
    img = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
        cand = SRC / 'train' / 'images' / (lbl.stem + ext)
        if cand.exists():
            img = cand
            break
    if img:
        pairs.append((img, kept))

print(f'筛选出 {len(pairs)} 张含目标类别的图片')

# 2. 打乱后按 8:1:1 划分
random.shuffle(pairs)
n = len(pairs)
splits = {
    'train': pairs[:int(n * .8)],
    'valid': pairs[int(n * .8):int(n * .9)],
    'test':  pairs[int(n * .9):],
}

# 3. 复制文件
for split, items in splits.items():
    (DST / split / 'images').mkdir(parents=True, exist_ok=True)
    (DST / split / 'labels').mkdir(parents=True, exist_ok=True)
    for img, lines in items:
        shutil.copy(img, DST / split / 'images' / img.name)
        (DST / split / 'labels' / (img.stem + '.txt')).write_text('\n'.join(lines))
    print(f'{split}: {len(items)} 张')

# 4. 写 data.yaml
(DST / 'data.yaml').write_text(
    f'path: "{DST.as_posix()}"\n'
    'train: train/images\n'
    'val: valid/images\n'
    'test: test/images\n'
    f'nc: {len(NEW_NAMES)}\n'
    f'names: {NEW_NAMES}\n',
    encoding='utf-8'
)
print(f'\n完成 -> {DST}')