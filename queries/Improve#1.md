# Siamese One‑Shot Project Plan (Keep Siamese; Borrow Improvements from BMVC’09)

This is tailored to your **Assignment 2 (Deep Learning, BGU)**: implement a **PyTorch Siamese one‑shot classifier** on **LFW‑a**, using the provided **Train/Test identity split** where *no subject from test appears in train*. fileciteturn4file0L18-L22  
The assignment explicitly asks for a Siamese network for **same / not‑same** decisions between two previously unseen people. fileciteturn4file0L12-L16

The goal here is **not** to replace your project with the BMVC’09 “Multiple OSS + SVM” system.  
Instead, we keep your Siamese one‑shot pipeline and **import only the parts that translate well**: stronger alignment discipline, explicit pose handling, and “multiple similarity signals” (implemented as *multiple heads / multiple score channels* inside the Siamese model), plus better stabilization (BatchNorm).

---

## 1) What stays the same (core Siamese one‑shot contract)

### Keep
- **Two-tower Siamese** (shared weights): `f(x1)`, `f(x2)`
- **Pair label**: `y ∈ {0,1}` (“same person” vs “different person”)
- **Training** on *train identities only*; **testing** on unseen identities (your provided split). fileciteturn4file0L18-L22
- A **single end-to-end PyTorch model** (assignment requirement). fileciteturn4file0L4-L9

### Avoid (for this assignment)
- External SVM stage, external ITML solver, handcrafted descriptor bank as the main representation.
Those are good academically, but they move you away from “Siamese network architecture” focus.

---

## 2) What to change (high impact, still Siamese)

Below are changes that preserve the Siamese setup but adopt ideas compatible with the BMVC’09 “why it worked” story:
- alignment quality matters
- pose can dominate similarity unless explicitly handled
- multiple similarity “views” are better than one

### Change A — Add **BatchNorm** throughout the backbone (stability + speed)
**Why:** your Siamese network is trained on many *pairs*, which makes optimization noisier. BN stabilizes feature scale and speeds convergence.

**Do**
- For each conv block: `Conv2d → BatchNorm2d → ReLU → MaxPool`
- For FC embedding: `Linear → BatchNorm1d → ReLU → Dropout`

**Do not**
- Put Sigmoid in intermediate layers (saturates). Keep Sigmoid only at the final output (or remove Sigmoid and use `BCEWithLogitsLoss`, recommended).

### Change B — Keep one embedding, but produce **multiple similarity channels**
BMVC’09 gets gains from **multiple similarity scores**. In Siamese terms: keep one embedding extractor, but compute *several* distances/similarities and let a small head learn how to combine them.

**Implement as:**
- Compute embeddings: `z1=f(x1)`, `z2=f(x2)`
- Compute a feature vector of similarities, e.g.:
  - `d_L1 = |z1 - z2|`
  - `d_L2 = (z1 - z2)^2`
  - `cos = cosine(z1, z2)`
  - `dot = z1 · z2`
- Concatenate: `h = [d_L1, d_L2, cos, dot]`
- Feed `h` to a small MLP head: `Linear → ReLU → Dropout → Linear → logits`

This preserves the Siamese structure but mimics “multiple OSS scores” with learnable fusion.

### Change C — Explicit **pose handling** (without abandoning Siamese)
BMVC’09’s key insight: in unconstrained faces, pose similarity can swamp identity. So we add a *pose-aware regularization* that still fits Siamese training.

Choose **one** of these (ordered from simplest to strongest):

1) **Pose-balanced sampling** (recommended baseline)
- When forming pairs, ensure you include:
  - same-ID / different-pose pairs
  - different-ID / same-pose pairs
This forces the model to learn identity cues beyond pose.

2) **Auxiliary head: pose bin prediction** (multi-task)
- Add a small classifier on top of the embedding to predict a coarse pose bin (e.g., left / frontal / right).
- Total loss: `L = L_pair + λ * L_pose`
This mirrors BMVC’09 “pose factors” but stays neural.

3) **Two-branch embedding (identity vs pose)**
- Shared backbone → two projection heads:
  - `z_id = head_id(backbone)`
  - `z_pose = head_pose(backbone)`
- Use `z_id` for pair decision, and regularize `z_pose` to be predictable from landmarks/pose bins.

### Change D — Use a more appropriate loss for Siamese
If you currently use a final Sigmoid with `BCELoss`, consider switching to:
- **`BCEWithLogitsLoss`** (remove final Sigmoid; feed logits directly)
This is numerically stable and standard for binary heads.

Optionally compare with:
- **Contrastive loss** (classic Siamese):
  - same pairs: minimize distance
  - different pairs: enforce margin

Report both if you have time; otherwise stick to logits BCE for simplicity.

### Change E — Better preprocessing and alignment discipline (must-have)
The assignment uses **LFW‑a**. Face alignment can make or break performance.
- Keep a consistent alignment/cropping strategy across train/test.
- Prefer a landmark-based alignment (eye centers + mouth) if you already have it.
- If not, at least enforce: tight face crop, resize to fixed size, consistent normalization.

---

## 3) Concrete architecture template (drop-in guidance)

### 3.1 Backbone (example)
- Input: `1×105×105` or `3×105×105` (depending on your pipeline)
- 4 conv blocks:

Block i:
- `Conv2d(in, out, kernel=3, stride=1, padding=1)`
- `BatchNorm2d(out)`
- `ReLU`
- `MaxPool2d(2)`

Typical channels:
- 64 → 128 → 256 → 256

### 3.2 Embedding head
- Flatten
- `Linear → BN1d → ReLU → Dropout(0.3) → Linear`
- Embedding dim: **512 or 1024** (usually better than 4096 for generalization on LFW scale)

### 3.3 Similarity fusion head (multi-channel)
Inputs:
- `abs(z1-z2)`  (vector)
- `(z1-z2)^2`   (vector)
- `cos(z1,z2)`  (scalar)
- `dot(z1,z2)`  (scalar)

Fusion:
- `Linear → ReLU → Dropout → Linear → logits`

---

## 4) Training workflow your agent should implement

### 4.1 Data pipeline
- Build **pair dataset** on-the-fly:
  - sample half “same”, half “not-same”
- Add **pose-balanced constraints** if you implement Change C(1)

### 4.2 Validation split (recommended)
The assignment says you may decide whether to use a validation set. fileciteturn4file0L18-L22  
Recommendation:
- Use 10–20% of *train identities* as validation identities (still disjoint from test).

### 4.3 Optimizer & schedule
- AdamW (good default), LR ~ 1e-3 to start
- Weight decay ~ 1e-4
- Early stopping on val loss (patience 5–10)

### 4.4 Logging (required by assignment spirit)
They expect “useful logging tools” and analysis. fileciteturn4file0L7-L10  
Minimum:
- train/val loss curves
- train/val accuracy curves
- example correct/incorrect pairs (with images) and explanations

---

## 5) Testing: how to run “one-shot” correctly

Because test identities are unseen, you should report:
1) **Pair verification accuracy** on test pairs (same/not-same)
2) A true one-shot identification task:
   - N-way, 1-shot: given 1 query, 1 support image per candidate identity; pick best match.
   - Report accuracy for N = 5, 10, 20 (if feasible)

Implementation:
- Build a “support set” from test identities (one image each).
- For each query image, compute similarity to each support image via your Siamese model.

---

## 6) What to write in the report (matches assignment rubric)
Your report must include:
- dataset analysis (size, per-class counts train/test) fileciteturn4file0L18-L22  
- full experimental setup (batch size, parameters, stopping criteria) fileciteturn4file0L20-L22  
- architecture description including **batchnorm usage** if any fileciteturn4file0L25-L27  
- performance analysis and error analysis with examples fileciteturn4file0L29-L36

---

## 7) “Diff” summary: BMVC’09 ideas translated into Siamese changes

| BMVC’09 Ingredient | Why it helped there | Siamese-equivalent change |
|---|---|---|
| Strong alignment | reduces nuisance variation | alignment + consistent crop/resize |
| Multiple OSS scores | multiple similarity views | multi-channel similarity head |
| Pose factoring | prevents pose dominating identity | pose-balanced sampling or pose aux-head |
| Metric learning (ITML) | better distance space | learn embedding + optionally contrastive loss |

---

## 8) Next step (actionable)
If you want, upload (or point me to) the **current Siamese model code** you’re running (or confirm it’s `notebook.ipynb`), and I will:
- produce a **patch-style change list** (exact code edits),
- refactor your model into: `backbone → embedding → multi-sim head`,
- add BN + BCEWithLogitsLoss correctly,
- and add a pose-balanced pair sampler (if you want that option).
