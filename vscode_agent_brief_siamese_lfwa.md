# VS Code Agent Brief — Implementing Koch et al. (2015) Siamese CNN for LFW-a One‑Shot Face Verification

This document is written to be *consumed by an automated coding agent* (e.g., VS Code Agent).  
Goal: **duplicate the core method from the paper** *Siamese Neural Networks for One‑shot Image Recognition* (Koch et al., ICML 2015) and **solve BGU Deep Learning Assignment 2** (one-shot face verification on LFW-a), using **PyTorch** and **Weights & Biases (W&B)** logging.

---

## 0) Inputs (what the agent should assume exists)

### Assignment requirements (from the provided assignment spec)
- Use PyTorch.
- Dataset: **LFW-a** (as referenced in the assignment; same variant used in DeepFace).
- Use the official train/test subject split so **no identity overlaps** between train and test.
- Task: given **two facial images of previously unseen persons**, predict **same person vs different** (binary verification).
- Report must include dataset stats, experimental setup, and training curves; code must be documented and reproducible. fileciteturn2file1

### Paper reference (what to implement “loosely”)
- Train a siamese network on **same/different pairs** (verification).
- Use a learned embedding + **weighted L1 distance** + **sigmoid** to predict similarity.
- Use ReLU in early layers, **sigmoid in the last layers** (per paper description). fileciteturn2file0

---

## 1) Problem translation — from Omniglot to LFW-a

### Paper’s pipeline (verification → one-shot)
1. Train siamese network as a binary classifier over image pairs:  
   label = 1 if same class/person else 0. fileciteturn2file0
2. At inference, use network score `p(x1, x2)` to rank similarity; choose best match in a candidate set (one-shot / few-shot). fileciteturn2file0

### Assignment’s pipeline
- **Training**: sample pairs from the **training identities** only.
- **Testing**: evaluate on **unseen identities** (test identities), either as:
  - **Verification accuracy** on random test pairs, and/or
  - **N-way one-shot identification**: for each query face, pick the correct identity from a set of `N` candidates (recommended to include because the paper does N-way tasks).

---

## 2) Core model specification (must match paper’s key design choices)

### 2.1 Siamese weight sharing
Two identical CNN “towers” process `x1` and `x2`. All parameters are shared. fileciteturn2file0

### 2.2 Similarity head (paper’s metric layer)
Paper uses **weighted L1 distance** between embeddings followed by **sigmoid**:

\[
p=\sigma\left(\sum_j \alpha_j \left|h^{(j)}_1-h^{(j)}_2\right|\right)
\]

Where `α_j` are trainable weights (one per embedding dimension). fileciteturn2file0

**Implementation note (PyTorch):**
- Let `e1, e2` be embeddings of shape `(B, D)`.
- Compute `d = torch.abs(e1 - e2)`.
- Then `logit = linear(d)` where `linear` has weights `α` and optional bias.
- Then `p = torch.sigmoid(logit)`.
- Train with **binary cross entropy** (BCE / BCEWithLogitsLoss).

### 2.3 Activations
Paper: **ReLU** in first `L-2` layers, **sigmoid** in remaining layers. fileciteturn2file0

**Practical interpretation for a modern PyTorch replica:**
- Convolution blocks + ReLU.
- Embedding head: you may include a dense layer with **sigmoid** (or keep embeddings linear and apply sigmoid only in similarity head).  
- To adhere more literally: use a sigmoid activation in the last dense layer before the distance head, *and* sigmoid at the output.

### 2.4 Convolution specifics (paper)
- Convs with **stride=1** (paper explicitly states fixed stride 1). fileciteturn2file0
- Optional max-pool with filter size=2 and stride=2. fileciteturn2file0
- Convolution uses “valid” behavior in the paper; in PyTorch you can choose padding carefully (valid = padding=0). fileciteturn2file0

**For faces (LFW-a):**
- Use RGB (3 channels).  
- Resize to a consistent size (e.g., 105×105 to mimic Omniglot scale; or 128×128 / 160×160 if convenient).
- Use padding if you need stable spatial dims; document this deviation in the report.

---

## 3) Data — what must be built

### 3.1 Required directory convention (recommended)
```
project/
  data/
    lfwa/
      images/               # extracted LFW-a images
      splits/               # provided train/test subject lists from TA’s link
  src/
  notebooks/ (optional)
  outputs/
  README.md
```

### 3.2 Splits (critical)
Use the train/test identity lists referenced by the assignment (“no subject from test included in train”). fileciteturn2file1

### 3.3 Pair sampling strategy (core to training)
The agent must implement a **PairDataset**:
- For training:
  - Sample **positive pairs**: 2 images of same person.
  - Sample **negative pairs**: 1 image from person A, 1 from person B.
- Balance positives and negatives (50/50 per batch).
- Ensure randomization each epoch.

**Important edge case:**
Some identities have very few images. For a positive pair, identity must have ≥ 2 images. If not, skip or handle via replacement.

### 3.4 Train/val split
Assignment says a validation set is optional; choose one (e.g., 10%–20% of training identities) for early stopping and calibration. fileciteturn2file1

Do **identity-level split** for validation:
- Train identities and validation identities must not overlap.
- Otherwise the network can “memorize” identities and validation becomes misleading.

---

## 4) Training objective and metrics

### 4.1 Loss
Use BCE loss on pair labels `y ∈ {0,1}`. Paper uses cross entropy for binary classifier with L2 regularization. fileciteturn2file0

In PyTorch:
- Prefer `BCEWithLogitsLoss` and have the model output logits.
- Add weight decay (L2) in the optimizer for regularization.

### 4.2 Accuracy (verification)
Compute:
- `p = sigmoid(logit)`
- `pred = (p >= threshold)`  
Default threshold 0.5, but you can choose threshold based on validation ROC/PR for best F1/accuracy.

### 4.3 One-shot metric (recommended)
Implement N-way one-shot evaluation on **unseen identities**:
- For each trial:
  - Pick a query image from identity `c*`.
  - Build candidate set of size `N` with 1 reference image per identity (including the correct one).
  - Compute `p(query, ref_i)` for all i; predict `argmax p`.
- Report accuracy over many trials.

This matches paper’s evaluation idea (ranking similarity for one-shot). fileciteturn2file0

---

## 5) W&B logging (mandatory for reproducibility)

Agent must log:
- Hyperparameters: lr, batch_size, image_size, embedding_dim, optimizer, weight_decay, augmentations, seed.
- Training: loss per step/epoch, accuracy per epoch, learning rate schedule.
- Validation: loss, accuracy, ROC-AUC (optional), best threshold.
- Example pairs: log a panel of true positives/false positives/true negatives/false negatives as images with captions.

Recommended W&B structure:
- Project: `orisin-ben-gurion-university-of-the-negev/projects` (as user indicated earlier)
- Run name: include model + image_size + seed.

---

## 6) Architecture templates the agent may implement (choose ONE)

### Option A — “Paper-faithful” small Siamese CNN (recommended)
**Tower (shared):**
1. Conv(3→64, k=10, stride=1, pad=0) + ReLU + MaxPool(2)
2. Conv(64→128, k=7, stride=1, pad=0) + ReLU + MaxPool(2)
3. Conv(128→128, k=4, stride=1, pad=0) + ReLU + MaxPool(2)
4. Conv(128→256, k=4, stride=1, pad=0) + ReLU
5. Flatten
6. FC → 4096 + **Sigmoid** (paper mentions sigmoids in final layers) fileciteturn2file0

**Head:**
- Weighted L1 distance + sigmoid output.

This resembles the “best conv architecture” concept described (conv stack → 4096 FC → L1 distance). fileciteturn2file0

### Option B — Practical face baseline (allowed deviation)
Use a pretrained backbone (ResNet18) as tower, then the same weighted-L1 head.  
If you do this, clearly document it as a deviation from paper, but acceptable if the assignment allows “other possibilities”. fileciteturn2file1

---

## 7) Implementation plan (agent checklist)

### Step 1 — Data ingestion
- Implement `parse_splits.py` to read train/test identity lists.
- Build `index_identities()` mapping: identity → list of image paths.
- Verify split integrity (no overlap of identities).

### Step 2 — Datasets
- `PairsDataset(identity_to_images, pairs_per_epoch, transform, seed)`
- `OneShotEpisodeDataset` (optional) or a `make_episode()` function for evaluation.

### Step 3 — Model
- `SiameseTowerCNN`
- `SiameseNetwork` wrapper that returns logits for pair `(x1, x2)`.

### Step 4 — Training loop
- Deterministic seeding.
- Mixed precision optional.
- Log to W&B.
- Save best checkpoint by validation loss (or one-shot accuracy). Paper uses early stopping based on one-shot validation error. fileciteturn2file0

### Step 5 — Evaluation
- Verification metrics on test identities.
- N-way one-shot accuracy on test identities.
- Export confusion examples (TP/FP/TN/FN images).

### Step 6 — Reproducibility
- `config.yaml` or argparse flags.
- `requirements.txt` and clear README.
- Ensure code runs end-to-end from a single entry point: `python -m src.train ...`

---

## 8) “Gotchas” specific to this assignment

1. **Identity leakage**: do not split by images; split by identities. fileciteturn2file1  
2. **Imbalanced identities**: some people have many images; pair sampler should avoid overusing a few identities (cap samples per identity per epoch).
3. **Threshold selection**: for verification, threshold can matter; select on validation identities only.
4. **Augmentations**: paper uses affine distortions heavily for Omniglot. fileciteturn2file0  
   For faces, apply mild augmentations (small rotation, color jitter, random crop) — do not distort too aggressively.
5. **Image normalization**: document mean/std (ImageNet or computed from train set).
6. **Runtime**: LFW-a is bigger than Omniglot; keep batch sizes reasonable.

---

## 9) Expected deliverables from the agent

### Code deliverables
- `src/train.py` — trains model, logs to W&B, saves checkpoints
- `src/eval.py` — runs verification + one-shot evaluation on test split
- `src/data.py` — datasets + transforms + split parsing
- `src/model.py` — tower + siamese head
- `src/utils.py` — seeding, metrics, checkpointing
- `README.md` — how to run

### Report deliverables (you write, but agent must generate stats/plots)
- Dataset stats tables (train/test counts; per identity counts; min/median/max images per identity). fileciteturn2file1
- Training curves (loss/acc).
- Example correct/incorrect classifications. fileciteturn2file1
- Full hyperparameter listing. fileciteturn2file1

---

## 10) Minimal acceptance criteria (definition of “done”)

A run is acceptable if:
- Code executes end-to-end without manual edits.
- Uses correct train/test identity separation. fileciteturn2file1
- Produces W&B logs (loss, accuracy, config).
- Provides quantitative results on test set (verification acc; ideally N-way one-shot acc).
- Saves model checkpoint and prints path.

---

## Source files linked in this project
- Paper PDF: `oneshot1.pdf` fileciteturn2file0  
- Assignment spec (MD replica): `Assignment_2_Facial_Recognition_one_shot_learning.md` fileciteturn2file1
