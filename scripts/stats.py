from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(r'D:\dev\crop-detect\datasets\grape_public')
names = yaml.safe_load(open(ROOT / 'data.yaml', encoding='utf-8'))['names']

total = Counter()
for split in ['train', 'valid', 'test']:
    d = ROOT / split / 'labels'
    if not d.exists():
        print(f'[跳过] {split} 目录不存在')
        continue
    c = Counter()
    files = list(d.glob('*.txt'))
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    total += c
    print(f'\n=== {split} ===  图片 {len(files)} 张,目标框 {sum(c.values())} 个')

print('\n=== 全部类别（按目标框数量降序）===')
for idx, n in total.most_common():
    print(f'{n:7d}   {names[idx]}')

missing = [names[i] for i in range(len(names)) if total[i] == 0]
if missing:
    print(f'\n[警告] 以下类别没有任何标注: {missing}')