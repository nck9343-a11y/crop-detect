# crop-detect

**公开葡萄病害数据集中的捷径学习：标注粒度不一致的因果作用及其限度**

*Shortcut Learning in a Public Grape Disease Dataset: Annotation Granularity as a
Modulator, Not a Cause*

一份论文的完整实验代码、日志与产物。中文正文见 [`paper/论文.md`](paper/论文.md)，
英文正文见 [`paper/paper_en.md`](paper/paper_en.md)。

**English readers: see [`README.en.md`](README.en.md) for the full English version.**

---

## English

**Full English version of this README: [`README.en.md`](README.en.md).**

In one public grape disease dataset, one class is annotated at whole-leaf level while
the other five are annotated at lesion level. No individual box is wrong; the
inconsistency is *between* classes. On 5156 cross-species images containing no grape,
65.7% of the model's false-positive boxes fall into that one class — an
over-representation of 13.41x against its share of the training annotations.
Counterfactual retraining shows granularity controls the *magnitude* of this shortcut;
a pre-registered manipulation in the opposite direction returns a negative result.
Granularity is a modulator, not a cause.

The pre-registered criteria of paper Section 4.4 are in the module docstring and
constants of [`scripts/Shortcut_experiment.py`](scripts/Shortcut_experiment.py)
(`BR_CUTS`, `MO_CUTS`, `E13_BASELINE`, `E13_TABLE`, each with an English gloss); the
verdict they yield is in the `prereg_verdict` key of
[`results/runs_shortcut/shortcut_E13.json`](results/runs_shortcut/shortcut_E13.json).
Readers checking that claim should start there.

The datasets and weights are not in this repository, but every number in the paper can
be re-derived from the artifacts that are, without retraining. See
[`README.en.md`](README.en.md) for setup, data and how to reproduce each table.

---

## 这个项目在做什么

起点是一件很具体的事：一处家庭种植的葡萄地块，往年凭经验在幼果期施一次药，
药照打，果实仍在成熟前一个月发病。原因在于主要真菌病害有潜伏期——
侵染在特定温湿度窗口完成，肉眼可见的症状要一至数周后才出现，
**等看见症状时，防治窗口已经过去了**。所以需要早期、客观、可量化的病征观测。

顺着这个需求做下去，本来只想训一个能用的病斑检测模型，
结果发现真正的障碍不在模型，而在数据。于是本项目变成了对一个问题的系统回答：

> **公开数据集的哪些固有属性，决定了在其上训练的检测系统是否真正可用？**

分三部分：

1. **性能上限在哪** —— 五组单变量对照，参数量从 2.58 M 到 32 M，输入分辨率从 384 到 960，
   还换了一次检测范式（YOLO 系 → DETR 系）。统一 pycocotools 口径下，
   测试集 mAP50 的极差只有 1.54 个百分点，与随机种子的波动同量级。
   **模型侧的优化空间基本耗尽了。**

2. **瓶颈在哪** —— 分尺度评估在五个架构上一致：大目标 AP 是小目标的 3.1–3.8 倍。
   提高输入分辨率确实改善了小目标（+1.35 pp，p = 0.018），
   但这个改善被聚合指标完全掩盖（mAP50-95 变化 −0.21 pp，p = 0.731）。
   **只看 mAP 会得出与事实相反的结论。**

3. **分布内的高分能不能信** —— 这是主要发现。

## 主要发现：标注粒度不一致会诱导捷径

数据集里有一类 `mosaic virus disease` 以**整叶级**标注为主（框中位面积占全图 43.16%），
其余五类一致采用**病斑级**标注（0.57%–8.12%）。
每个框单独看都是对的，没有一个标错——问题在类别之间的粒度不统一。

这类问题现有的标签纠错方法查不出来（因为没有错标），
分布内评估也看不见——它甚至表现为该类 AP 最高（样本最少却得分第一）。

用 5156 张**不含葡萄**的跨物种图像（木薯／玉米／番茄）作阴性对照，真值客观：

- 模型输出 12221 个假阳性框，其中 **65.7% 归入那一类**，相对训练占比过表达 **13.41 倍**；
- 方向与类别频率先验相反（该类恰是训练集中最稀有的，占 4.9%）；
- 1408 张图的置信度超过 0.5，**调阈值解决不了**。

**单变量反事实重训**确立了因果：图像、框数、类别身份全部不变，只把那一类的标注框
缩到中心 25% 面积——其分布内 AP 从 0.7609 掉到 0.4552（该类由第一名落到第四名），
跨物种假阳性同时减少 66%。**安慰剂对照**（改缩另一类）证实效应特异于被操纵的类别。

还有两个衍生发现：

- 若改为**整体剔除**那一类，问题不消失——假阳性转移到标注次粗的类别（+51%、+469%）。
  捷径依附的是"标注最粗的那个位置"，不是某个特定病害。
- 缩小任一粗标注类别，假阳性**总量**都会下降（9176 → 5971 → 3956）。
  所以干预不能只修最粗的一类，要对所有粗标注类别一并规范。

**如实说明限度**：标注粒度只解释了大约一半——缩框后仍余 6.16 倍的过表达，
其余成因未知；模型究竟依据什么图像特征，我们检验的假设已被否定（论文 4.3 节），
目前没有答案。

另有一项独立结论：依成像光学关系核算，**毫米级病斑的航拍检测在任何具备作业可行性的
高度上都不可达**（5 mm 病斑要达到 32 像素，飞行高度不得超过 0.88 m，覆盖一公顷约需 14 万张图）。
无人机的角色应是冠层级异常定位，病斑级确诊必须由地面环节承担。

---

## 环境配置

**硬件**：全部实验在 RTX 4060 Laptop 8 GB 上完成，峰值显存占用 2.8 GB。
不需要高性能计算设备。

**系统与 Python**：Windows，Python **3.11**（torch 轮子标签为 cp311，其他版本不兼容）。

### 1. 建虚拟环境

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
```

### 2. 装 torch（必须单独装，pip 默认源没有）

本项目用的是 CUDA 13.2 专用轮子：

```bash
pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu132
```

没有 NVIDIA 显卡也能跑评估脚本（会退到 CPU，慢很多），
此时改装 CPU 版即可：`pip install torch torchvision`。

### 3. 装其余依赖

```bash
pip install -r requirements.txt
```

要完全复现当时的环境（含全部传递依赖的精确版本），用 `requirements-lock.txt`。

### 4. 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 数据

本仓库**不包含数据集**（7.5 GB，且主数据集的再分发许可尚未确认）。
获取与重建方式见 [`datasets/README.md`](datasets/README.md)。

仓库里包含的是**实验产物**：模型在测试集与阴性对照集上的原始预测（COCO 格式 JSON）、
统一口径的评估结果、以及全部训练日志。这意味着**不需要 GPU、不需要重新训练，
也能直接复核论文里的每一个数字**。

自采的田间照片（10 张）**不在本仓库中**。样本量太小，论文 4.2 节已明确将其
排除在正式实验之外，不支撑任何已发表结果；仓库公开，故不随之公开。

---

## 复现论文中的表格

不需要数据集与权重，直接用仓库内已有的预测结果：

```bash
python scripts/Eval_perclass_seeds.py    # 表 10 末列：逐类别 AP 的种子间波动
```

需要数据集与权重（见 `datasets/README.md`）：

```bash
python scripts/Eval_unified.py           # 表 3、表 5、表 7
python scripts/Shortcut_experiment.py    # 表 9、11、12、13：跨物种阴性对照
python scripts/Counterfactual_prepare.py # 构造反事实数据集
python scripts/Counterfactual_train.py   # 反事实重训（约 12 小时）
python scripts/Eval_counterfactual.py    # 表 10
python scripts/Gsd_planner.py            # 表 14、表 15：航拍可行性核算
```

生成 Word 版论文：

```bash
python scripts/Md_to_docx.py             # paper/论文.md -> paper/论文.docx
```

脚本内部使用绝对路径（`D:\dev\crop-detect\...`）。
若项目换了位置，需同步修改各脚本顶部的路径常量。

---

## 目录结构

```
paper/            论文与投稿材料
  论文.md            正文（唯一正式版本，改这一份）
  ai_disclosure.md          生成式 AI 使用声明（四个版本备选）
  related_work_framework.md 相关工作的文献素材
  drafts/          历史草稿，仅本地保留，不入库

scripts/          全部实验脚本，均可独立运行
logs/             训练与实验的控制台日志，是论文 5.3 节几处说法的原始依据
results/          实验产物
  runs_unified/     统一 pycocotools 口径的评估结果与原始预测（表 3、7、10）
  runs_shortcut/    跨物种阴性对照（表 9、11、12、13）
  runs_scale/       分辨率实验
  runs_sahi/        切图推理（E7，设计无效，见论文 5.3 节）
  spatial_out/      空间格局分析的合成数据验证图

datasets/         数据集（不入库，见其 README）
my_grape/  runs/  weights/  installers/     不入库（依次为：未用于正式实验、体积超限）
```

本仓库为**公开**。转公开前清理过一次历史：`my_grape/` 已用 `git filter-branch`
从全部提交中移除，不只是从最新版本里删掉——已提交的内容即便日后删除仍留在历史记录里，
所以那一步必须在首次推送之前做完。

---

## 许可

代码以 **AGPL-3.0** 发布，见 [`LICENSE`](LICENSE)。

选择 AGPL 不是偏好，而是义务传递：本项目的 E1–E4 依赖 Ultralytics YOLO11，
其许可为 AGPL-3.0，对衍生作品的分发有开源要求。
E5 使用的 RF-DETR 为 Apache 2.0，约束较少。

数据集的许可另计，见 `datasets/README.md`。论文正文的著作权归作者所有。

---

## 状态

英文版准备投 arXiv（cs.CV，交叉 cs.LG），中文版投 ChinaXiv。拿到编号后回填此处。

投稿备忘见 [`paper/arxiv_submission.md`](paper/arxiv_submission.md)。
