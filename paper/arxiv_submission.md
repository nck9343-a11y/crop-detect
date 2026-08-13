# 投稿备忘：chinaXiv 与 arXiv

生成于 2026-08-13。正文见 `论文.md`（中文）与 `paper_en.md`（英文）。

---

## 一、文件对照

| 文件 | 用途 |
|---|---|
| `论文.md` / `论文.docx` | 中文正文，投 chinaXiv 用 |
| `paper_en.md` / `paper_en.docx` | 英文正文 |
| `arxiv_bilingual.docx` | **英文在前 + 中文在后**，合为单一文件，投 arXiv 用 |
| `figures/fig*_en.*` | 英文标注版插图（英文正文引用的是这一套） |

重新生成：

```bash
python scripts/Make_figures.py              # 中文插图
python scripts/Make_figures.py --lang en    # 英文插图
python scripts/Md_to_docx.py paper/paper_en.md paper/论文.md paper/arxiv_bilingual.docx
```

导出 PDF：用 Word 打开 docx → 另存为 PDF。本机没有 LaTeX 与 pandoc，
这是唯一可用的路径；arXiv 接受纯 PDF 投稿（"不接受由 TeX 源生成的 PDF"
那一条不适用，因为本文未使用 TeX）。

---

## 二、arXiv 的语言要求（2026-02-11 起生效，务必遵守）

官方帮助页 https://info.arxiv.org/help/faq/multilang.html 的操作性规定：

- **所有投稿必须有完整英文版**，可以是原文，也可以是随附的翻译；
- **两种语言必须合为单一文件，且英文在前**——原文："Please prepare your
  submission so that the English language version appears first, followed by
  the non-English language version."
- **元数据字段只能容纳有限的非 ASCII 字符**，因此标题与摘要字段填英文；
- **Comments 字段须注明正文语言**。

`arxiv_bilingual.docx` 已按此顺序装配。

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
Main text in Chinese with a full English version included; the English version
appears first. 26 pages, 3 figures, 16 tables. Code and evaluation artifacts:
[仓库地址，推送后填入]
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
- [ ] 仓库地址：GitHub 推送后填入 Comments 字段与正文附录
- [ ] 确认目标期刊（若后续投稿）对预印本的政策——chinaXiv 由中科院建设，
      其定位即为保护首发权并与期刊衔接；国际期刊多数接受 arXiv 预印本

---

## 五、建议顺序

1. **先确认 endorsement**（10 分钟，决定 arXiv 走不走得通）
2. **填作者信息**
3. **投 chinaXiv**：中文稿即为可投状态，先占住首发权
4. **投 arXiv**：用 `arxiv_bilingual.docx` 导出的 PDF
