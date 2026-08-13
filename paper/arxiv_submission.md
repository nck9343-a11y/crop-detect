# 投稿备忘：chinaXiv 与 arXiv

生成于 2026-08-13。正文见 `论文.md`（中文）与 `paper_en.md`（英文）。

---

## 一、文件对照

| 文件 | 用途 |
|---|---|
| `论文.md` / `论文.docx` | 中文正文，投 chinaXiv 用 |
| `paper_en.md` / `paper_en.docx` | 英文正文 |
| `arxiv_bilingual.docx` | 双语合订本，**不用于投稿**，留作自用或给他人通读 |
| `figures/fig*_en.*` | 英文标注版插图（英文正文引用的是这一套） |

重新生成：

```bash
python scripts/Make_figures.py              # 中文插图
python scripts/Make_figures.py --lang en    # 英文插图
python scripts/Md_to_docx.py paper/paper_en.md paper/paper_en.docx        # arXiv 用
python scripts/Md_to_docx.py paper/论文.md paper/论文.docx                  # ChinaXiv 用
python scripts/Md_to_docx.py paper/paper_en.md paper/论文.md paper/arxiv_bilingual.docx   # 自用合订
```

导出 PDF：用 Word 打开 docx → 另存为 PDF。本机没有 LaTeX 与 pandoc，
这是唯一可用的路径；arXiv 接受纯 PDF 投稿（"不接受由 TeX 源生成的 PDF"
那一条不适用，因为本文未使用 TeX）。

---

## 二、投稿形态：两个平台各投单语版本

**英文版 → arXiv，中文版 → ChinaXiv，各自单语，都是正常形态。**

arXiv 自 2026-02-11 起要求"所有投稿必须有完整英文版"
（https://info.arxiv.org/help/faq/multilang.html ）。该要求在正文本身即为英文时
自动满足——需要把两种语言合订、且英文在前的，只是正文为非英文的情形。
因此英文单投无需任何特殊处理。

不采用双语合订的理由：篇幅翻倍，同一内容在一份 PDF 里出现两遍，
读者与 moderator 都不便，而且 arXiv 上极少有人这样做。
`arxiv_bilingual.docx` 保留自用即可。

**建议两版互相交叉引用。** 同一工作以两种语言发布于两个预印本平台是常见做法，
但宜在各自的 Comments 字段（或正文脚注）注明另一语言版本的所在，例如：

- arXiv 英文版 Comments：`A Chinese version of this work is available at ChinaXiv: [ID/链接]`
- ChinaXiv 中文版备注：`本文英文版见 arXiv:[ID]`

这不是任一平台的硬性规定，但可避免"重复发布"的误会，也方便读者找到对应版本。
先投的那一版拿到编号后，补进后投的那一版即可。

关于 AI 翻译：Nature 与 Slator 的报道称 arXiv 接受 AI 翻译，只要忠实于原文；
但**官方帮助页未就此作出规定**。本文英文版由 AI 辅助完成，已作全篇数字比对
（两版数字零差异，差异仅为记数法，如 1/4 → quarter、14 万 → 140 000），
但**忠实性的最终责任在作者**，投稿前须逐段通读。

---

## 三、arXiv 投稿表单可直接粘贴的内容

### Title

```
Shortcut Learning in a Public Grape Disease Dataset: The Causal Role of Inconsistent Annotation Granularity, and Its Limits
```

### Abstract

```
Public datasets are the main data source for research on agricultural disease
detection, and their fitness for use is usually judged from the metrics reported
on them - yet those metrics say nothing about whether the annotation scheme is
internally consistent. Taking one public grape disease detection dataset (3288
images, 11995 boxes, 6 classes) as the object of study, this paper asks which
intrinsic properties of public data determine whether a detection system built on
it is actually usable. Five single-variable controls spanning model capacity from
2.58M to 32M parameters, input resolution from 384 to 960, and one change of
detection paradigm yield a test-set mAP50 range of only 1.54 percentage points
under a single evaluation protocol - the same order as seed-to-seed variation;
scale-stratified evaluation locates the bottleneck at small objects consistently
across all five architectures. The principal finding lies on the data side: one
class is annotated predominantly at whole-leaf level (median box area 43.16% of
the image) while the other five are annotated at lesion level (0.57% to 8.12%).
Using 5156 cross-species images containing no grape as a negative control, 65.7%
of the 12221 false-positive boxes fall into that one class, an over-representation
of 13.41x relative to its share of the training annotations, in the direction
opposite to a class-frequency prior. Single-variable counterfactual retraining
establishes causality: shrinking only that class's boxes lowers its
in-distribution AP from 0.7609 to 0.4552 while the other five classes change by no
more than seed-to-seed variation, and cross-species false positives fall by 66% at
the same time; a placebo control confirms the effect is specific to the
manipulated class. Removing the class outright does not solve the problem - the
false positives transfer to the next most coarsely annotated class. Annotation
granularity accounts for roughly half of the over-representation; the remainder is
unexplained. We further propose a granularity screening statistic requiring
neither images nor training, and use it to separate two regimes, uniformly coarse
and uneven across classes, of which only the latter is dangerous. The failure mode
reported here is invisible to in-distribution evaluation and therefore constitutes
a blind spot in judging whether a public dataset is fit for use.
```

（摘要已去除非 ASCII 字符：× 写作 x，破折号写作 -。）

### Categories

- Primary: **cs.CV** (Computer Vision and Pattern Recognition)
- Cross-list: **cs.LG** (Machine Learning)

### Comments

```
3 figures, 16 tables. A Chinese version of this work is available at ChinaXiv:
[编号，投出后填入]. Code and evaluation artifacts: [仓库地址，推送后填入]
```

### License

建议选 **CC BY 4.0**——与所用数据集一致，且允许他人复用。
注意 arXiv 要求"an irrevocable license to distribute the work"，选定后不可更改。

---

## 四、投稿前逐项确认

- [ ] **填上作者姓名与单位**（`论文.md` 与 `paper_en.md` 顶部各有一处占位）
- [ ] **确认 arXiv endorsement**：首次投稿或投新分类可能需要背书。
      机构邮箱（`.edu.cn` 等）通常可免。**这一步应在其他工作之前完成**——
      若背书拿不到，arXiv 路线不成立
- [ ] **chinaXiv 实名认证**：需机构邮箱与手机号
- [ ] 通读英文版，确认翻译忠实（责任在作者，见第二节）
- [ ] 英文版插图引用的是 `fig*_en`，确认 docx 中显示正确
- [ ] AI 声明：两版正文末尾均已包含，措辞覆盖脚本编写与结果复核
- [ ] **单位用学校官方英文名**，不要自行翻译
- [ ] 两版互相交叉引用：先投的拿到编号后，补进另一版的 Comments
- [ ] 仓库地址：GitHub 推送后填入 Comments 字段与正文附录
- [ ] 确认目标期刊（若后续投稿）对预印本的政策——chinaXiv 由中科院建设，
      其定位即为保护首发权并与期刊衔接；国际期刊多数接受 arXiv 预印本

---

## 五、建议顺序

1. **先确认 endorsement**（10 分钟，决定 arXiv 走不走得通）
2. **填作者信息**
3. **投 chinaXiv**：中文稿即为可投状态，先占住首发权
4. **投 arXiv**：用 `paper_en.docx` 导出的 PDF（英文单语）
5. 两边都出来后，互相补上对方的编号
