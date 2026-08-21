# crop-detect

**Shortcut Learning in a Public Grape Disease Dataset: Annotation Granularity as a
Modulator, Not a Cause**

Code, logs and experimental artifacts for the paper above.
Manuscript: [`paper/paper_en.md`](paper/paper_en.md) (English),
[`paper/论文.md`](paper/论文.md) (Chinese).
中文版说明见 [`README.md`](README.md).

---

## What this project is about

It started from something concrete: a household grape plot, sprayed once at the
young-fruit stage on the strength of experience. The spraying happened; the fruit
still fell ill a month before ripening. The reason is that the major fungal diseases
have a latent period — infection completes inside a particular window of temperature
and humidity, and visible symptoms appear one to several weeks later. **By the time
you can see the symptoms, the window for control has closed.** So what is needed is
early, objective, quantifiable observation of disease signs.

Following that need, the intent was only to train a workable lesion detector. The
real obstacle turned out not to be the model but the data. The project therefore
became a systematic answer to one question:

> **Which intrinsic properties of a public dataset determine whether a detection
> system built on it is actually usable?**

In three parts:

1. **Where the performance ceiling is.** Five single-variable controls: 2.58 M to
   32 M parameters, input resolution 384 to 960, and one change of detection paradigm
   (YOLO family → DETR family). Under a single pycocotools protocol the test-set
   mAP50 range is only 1.54 percentage points — the same order as seed-to-seed
   variation. **The room for optimisation on the model side is essentially exhausted.**

2. **Where the bottleneck is.** Scale-stratified evaluation agrees across all five
   architectures: AP on large objects is 3.1x to 3.8x that on small ones. Raising
   input resolution does improve small objects (+1.35 pp, p = 0.018), but that
   improvement is completely masked by the aggregate metric (mAP50-95 changes
   −0.21 pp, p = 0.731). **Reading mAP alone leads to the opposite conclusion.**

3. **Whether a high in-distribution score can be trusted.** This is the main finding.

## Main finding: inconsistent annotation granularity induces a shortcut

One class, `mosaic virus disease`, is annotated predominantly at **whole-leaf** level
(median box area 43.16% of the image); the other five are annotated consistently at
**lesion** level (0.57%–8.12%). Every individual box is correct — none is mislabelled.
The problem is the inconsistency *between* classes.

Existing label-error detection cannot find this (there is no label error), and
in-distribution evaluation cannot see it — it even shows up as the *highest* AP of any
class, the rarest class scoring first.

Using 5156 cross-species images containing **no grape** (cassava / maize / tomato) as
a negative control, where ground truth is objective:

- the model emits 12221 false-positive boxes, of which **65.7% fall into that one
  class** — an over-representation of **13.41x** against its share of the training
  annotations;
- the direction is opposite to a class-frequency prior (that class is the *rarest* in
  training, at 4.9%);
- 1408 images carry confidence above 0.5, so **thresholding does not fix it**.

**Single-variable counterfactual retraining** establishes the causal link: images, box
count and class identity all held fixed, only that class's boxes shrunk to the central
25% of their area — its in-distribution AP falls from 0.7609 to 0.4552 (from first to
fourth among the classes) while cross-species false positives drop by 66% at the same
time. A **placebo control** (shrinking a different class instead) confirms the effect
is specific to the manipulated class.

Two further findings:

- **Removing** that class outright does not make the problem go away — the false
  positives migrate to the next most coarsely annotated classes (+51%, +469%). The
  shortcut attaches to "whichever position is most coarsely annotated", not to a
  particular disease.
- Shrinking *any* coarsely annotated class lowers the **total** number of false
  positives (9176 → 5971 → 3956). Intervention must therefore regulate all coarsely
  annotated classes together, not just the most conspicuous one.

**Stating the limits honestly:** annotation granularity accounts for only about half.
An over-representation of 6.16x remains after shrinking, and the rest is unexplained.
What image features the model actually keys on is unknown — the hypothesis we tested
was refuted (paper Section 4.3).

One independent result: from the imaging-optics relation, **airborne detection of
millimetre-scale lesions is unattainable at any operationally feasible altitude** (for
a 5 mm lesion to span 32 pixels, flight altitude may not exceed 0.88 m, and covering
one hectare would take roughly 140 000 images). The role of a drone should be
canopy-level anomaly localisation; lesion-level diagnosis must be carried by a
ground-based step.

---

## The pre-registered criteria

Section 4.4 / Result 5 of the paper reports a **negative result** from a manipulation
whose criteria were fixed before the results were seen. Everything needed to check
that claim is in this repository:

| What | Where |
|---|---|
| The criteria, the reasoning behind the band thresholds, and the 3x3 decision table | module docstring and module-level constants of [`scripts/Shortcut_experiment.py`](scripts/Shortcut_experiment.py) — `BR_CUTS`, `MO_CUTS`, `E13_BASELINE`, `E13_TABLE`, each with an English gloss |
| The verdict those criteria yield, written to disk alongside the raw false-positive counts | `prereg_verdict` and `screening` keys of [`results/runs_shortcut/shortcut_E13.json`](results/runs_shortcut/shortcut_E13.json) |

The band thresholds are derived from the screening statistic of the manipulated
dataset, not from any observed false-positive count. All nine cells of the table carry
a pre-assigned reading — leaving a cell blank would leave room for post-hoc
interpretation. Substituting the E1 baseline values into the table yields "refuted",
so "nothing changed" necessarily falls on the refuting side and the table cannot
manufacture a confirmation.

The `reading` strings inside `E13_TABLE` are left in Chinese, with English glosses in
the comments, so that re-running the script reproduces the published artifact byte for
byte.

---

## Environment

**Hardware.** All experiments ran on an RTX 4060 Laptop 8 GB, peak GPU memory 2.8 GB.
No high-performance computing equipment is required.

**OS and Python.** Windows, Python **3.11** (the torch wheels are tagged cp311; other
versions are incompatible).

### 1. Virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
```

### 2. Install torch separately (not on the default pip index)

This project uses the CUDA 13.2 wheels:

```bash
pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu132
```

The evaluation scripts also run without an NVIDIA GPU (falling back to CPU, much
slower); in that case install the CPU build: `pip install torch torchvision`.

### 3. Remaining dependencies

```bash
pip install -r requirements.txt
```

To reproduce the exact environment including every transitive dependency, use
`requirements-lock.txt`.

### 4. Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Data

**The datasets are not in this repository** (7.5 GB, and the redistribution licence of
the main dataset is unconfirmed). See [`datasets/README.md`](datasets/README.md) for
what each one is, where it comes from, and how to rebuild it.

What *is* here are the **experimental artifacts**: the model's raw predictions on the
test set and on the negative control set (COCO JSON), all evaluations under a single
protocol, and the complete training logs. This means **every number in the paper can
be checked without a GPU and without retraining anything.**

A handful of self-collected field photos (10 images) are also not included. They were
excluded from the formal experiments (paper Section 4.2) and support none of the
reported results.

---

## Reproducing the tables

No dataset or weights needed — straight from the predictions already in the repo:

```bash
python scripts/Eval_perclass_seeds.py    # Table 10, last column: seed-to-seed variation of per-class AP
```

Dataset and weights needed (see `datasets/README.md`):

```bash
python scripts/Eval_unified.py           # Tables 3, 5, 7
python scripts/Shortcut_experiment.py    # Tables 9, 11, 12, 13: cross-species negative control
python scripts/Counterfactual_prepare.py # build the counterfactual datasets
python scripts/Counterfactual_train.py   # counterfactual retraining (~12 h)
python scripts/Eval_counterfactual.py    # Table 10
python scripts/Conf_sweep.py             # Result 5: low-threshold re-measurement
python scripts/Gsd_planner.py            # Tables 14, 15: airborne feasibility calculation
```

**Note on paths.** The scripts use absolute paths (`D:\dev\crop-detect\...`) pointing
at the author's machine. To run them elsewhere, edit the path constants at the top of
each script. The scripts are otherwise left exactly as they were when they produced
the published results.

---

## Layout

```
paper/            manuscript and submission material
  paper_en.md        English manuscript
  论文.md            Chinese manuscript
  arxiv_submission.md  submission memo (Chinese)
  ai_disclosure.md     declaration on the use of generative AI
  figures/             figures, Chinese and English versions

scripts/          all experiment scripts, each runnable on its own
logs/             console logs of training and experiments; the primary
                  evidence behind several statements in paper Section 5.3
results/          experimental artifacts
  runs_unified/     evaluations and raw predictions under a single
                    pycocotools protocol (Tables 3, 7, 10)
  runs_shortcut/    cross-species negative control (Tables 9, 11, 12, 13)
  runs_scale/       resolution experiment
  runs_sahi/        sliced inference (E7; the design was ineffective,
                    see paper Section 5.3)
  spatial_out/      synthetic-data validation plots for the spatial analysis

datasets/         datasets (not tracked; see its README)
my_grape/  runs/  weights/  installers/     not tracked
```

---

## Licence

Code is released under **AGPL-3.0**, see [`LICENSE`](LICENSE).

AGPL is not a preference but an obligation passed along: E1–E4 depend on Ultralytics
YOLO11, licensed AGPL-3.0, which imposes open-source requirements on the distribution
of derivative works. E5 uses RF-DETR, which is Apache 2.0 and less constraining.

Dataset licences are separate — see `datasets/README.md`. Copyright in the manuscript
remains with the author.

---

## Status

The English version is being submitted to arXiv (cs.CV, cross-listed cs.LG); the
Chinese version to ChinaXiv. Identifiers will be filled in here once assigned.
