# 数据集

本目录下的数据集**不纳入版本控制**（合计约 7.5 GB，且主数据集的再分发许可尚未确认）。
下面说明每一份数据是什么、从哪里来、如何重建。

## 需要准备的数据

| 目录 | 内容 | 用途 | 来源 |
|---|---|---|---|
| `grape_public/` | 3288 图，11995 框，6 类，YOLO 格式 | 主实验 E1–E5 | Roboflow，工作区 `wscs`，项目 `grape-uyimv`，版本 1（导出于 2025-11-03）。**许可待确认，见下** |
| `grape_coco/` | 同上，COCO 格式 | pycocotools 评估 | 同一份数据的 COCO 导出 |
| `fieldplant/` | 5156 图，27 类（木薯／玉米／番茄） | 跨物种阴性对照 | [Roboflow Universe: plant-disease-detection/fieldplant v11](https://universe.roboflow.com/plant-disease-detection/fieldplant/dataset/11)，CC BY 4.0 |
| `grape_cf_shrink/` | 主数据集的反事实版本：mosaic 类的框缩至中心 25% 面积 | 反事实重训（论文 4.4 节主组） | 由脚本生成，见下 |
| `grape_cf_drop/` | 主数据集剔除 mosaic 类 | 反事实重训（旁证组） | 由脚本生成 |
| `grape_cf_placebo/` | 改缩 botrytis 类的框 | 安慰剂对照 | 由脚本生成 |
| `tomato3/` | 番茄子集 | 早期探索，论文未使用 | 由 `scripts/prepare.py` 从 fieldplant 抽取 |

## 重建三组反事实数据集

准备好 `grape_public/` 之后，两条命令即可生成，无需人工干预：

```bash
python scripts/Counterfactual_prepare.py              # 生成 grape_cf_shrink 与 grape_cf_drop
python scripts/Counterfactual_prepare.py --only e12   # 生成 grape_cf_placebo
```

脚本只改动标注框，图像原样复制。缩框的操作是"中心不变、宽高各乘 0.5"，
因此面积变为原来的 25%。这一操作只施加于目标类别——
花叶病毒的症状是全叶弥漫性花斑，裁到中心后框内仍是真实的症状组织；
对黑腐病那种局部圆斑则不成立（论文 4.4 节）。

## 目录约定

各数据集均为 `train/valid/test` 三分，图像与标注分列 `images/` 与 `labels/`。
注意 Roboflow 导出的 `data.yaml` 中 `train: ../train/images` 这样的相对路径，
与 `path:` 拼接后会指向数据集目录之外，只靠框架的回退搜索才能工作；
`Counterfactual_prepare.py` 生成的 yaml 已改为正确的相对路径。

## 主数据集的许可

`grape_coco/README.dataset.txt`（Roboflow 在下载时生成的归属声明）记录：

```
# grape > 2025-11-03 9:39am
https://universe.roboflow.com/wscs/grape-uyimv
Provided by a Roboflow user
License: CC BY 4.0
```

即该数据集在 Roboflow **Universe** 上公开发布，许可为 **CC BY 4.0**，可再分发，
但须标注出处。论文 2.1 节已以上述地址作为出处标注。

需要留意一处记录不一致：`grape_public/data.yaml` 的 `roboflow:` 段写的是
`license: Private`、地址为 `app.roboflow.com/...`（工作区内部链接）。
两份文件来自不同的导出途径——`grape_coco` 自 Universe 下载，
`grape_public` 自工作区导出——后者的 license 字段反映导出时的工作区设置，
不是项目对外声明的许可。以 `README.dataset.txt` 为准。

FieldPlant 同为 CC BY 4.0，引用条目见论文参考文献 [10]。

**关于是否随仓库分发**：CC BY 4.0 允许再分发，因此不上传的原因只剩体积
（图像合计 7.5 GB）。若需要让他人复核论文表 2、表 16 与图 1 的粒度统计，
可只分发标注文件（YOLO 格式的 `labels/`，合计数 MB），并保留本节的出处标注。
