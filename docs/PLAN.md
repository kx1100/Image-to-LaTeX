# PLAN.md — Image-to-LaTeX

> **This document is the source of truth for the project.**
> Update it only when requirements, architecture, milestones, or implementation status
> materially change. Requirements and constraints must not be changed without explicit
> approval from the project owner. Decisions and their rationale live in
> [DECISIONS.md](./DECISIONS.md); the narrative spec lives in
> [project_spec.md](./project_spec.md).

Last materially updated: 2026-08-29

---

## 1. Goal

Build a system that takes a photograph or scan of handwritten or printed mathematical
notation and outputs correct, compilable LaTeX source representing that notation.

**Primary use case:** a user photographs handwritten math (homework, notes, a textbook
problem) on a phone; the system returns LaTeX that renders identically — or
near-identically — to the original.

**Hard constraint (do not relax without approval):** the product's runtime recognition
engine must not be a general-purpose multimodal LLM API (GPT-4V, Claude vision, etc.).
Image parsing and structure recognition are performed by a purpose-built model we train
and control. Reusing open pretrained *vision* weights as an initialization is permitted;
see [DECISIONS.md](./DECISIONS.md) D-002.

---

## 2. Requirements

### 2.1 Functional — in scope for v1

| ID | Requirement |
|---|---|
| R-1 | Accept a single mathematical expression per image (photo or scan). |
| R-2 | Support both handwritten and printed input. |
| R-3 | Recognize core LaTeX constructs: fractions, exponents/subscripts, roots, Greek letters, common operators, summations/integrals, parentheses/brackets, basic matrices. |
| R-4 | Emit a valid, compilable LaTeX string. |
| R-5 | Produce a per-output confidence score (from decoder token probabilities) and flag low-confidence recognitions for review. |
| R-6 | Validate output before returning it: brace/bracket balance plus a headless compile check, with repair heuristics or a flagged best-partial result on failure. |
| R-7 | Expose the trained model behind an inference API. |
| R-8 | Preprocess input images to a normalized form: grayscale, denoise, contrast normalization, binarize, deskew, crop-to-content, scale/aspect normalization. (Order is significant — see [DECISIONS.md](./DECISIONS.md) D-008.) |

### 2.2 Non-functional

| ID | Requirement |
|---|---|
| N-1 | No general-purpose multimodal LLM API in the runtime recognition path (see Goal). |
| N-2 | Training and inference environments are reproducible (pinned dependencies, containerized). |
| N-3 | Every training run is tracked: config, dataset version, metrics, checkpoint. |
| N-4 | Train/val/test splits are fixed and documented; no symbol-distribution leakage across splits. |
| N-5 | Evaluation is reproducible from a checkpoint + a split by a single command. |

### 2.3 Explicitly out of scope for v1

Deferred to v2+, and not to be built without approval:

- Full-page / document layout (multiple expressions, mixed text + math, paragraph structure)
- Diagrams, graphs, geometric figures
- Multi-line / aligned equation systems (`align`, `cases`) — stretch goal only
- Real-time video / live-camera recognition
- Non-Latin symbol sets beyond standard math notation
- Mobile/web frontend and the active-learning loop are **optional** v1 stretch items (M7, M8)

### 2.4 Evaluation metrics

The metric set is fixed; the numeric targets in §5 are proposals pending approval.

- **Exact match accuracy** — predicted string == ground truth after whitespace normalization
- **BLEU** — token-level sequence similarity (standard in the im2latex literature)
- **Token edit distance** — Levenshtein over the token sequence
- **Render match** — rendered prediction vs. rendered ground truth, image-level; tolerant of equivalent LaTeX (`x^2` vs `x^{2}`)
- **Compilation validity rate** — % of outputs that compile without error

---

## 3. Architecture and technologies

Pipeline (see [architecture.md](./architecture.md) for the full diagram):

```
Image → Preprocessing → Recognition Model → Postprocessing/Validation → LaTeX Output
```

| Part | Responsibility | Technology |
|---|---|---|
| **Preprocessing** | Grayscale, denoise (median/Gaussian), contrast normalization (CLAHE), binarize (Otsu/adaptive), deskew (Hough), crop to content bbox, scale/aspect normalization | Python, OpenCV, NumPy |
| **Tokenizer** | Custom LaTeX-token vocabulary (`\frac`, `{`, `}`, `^`, `_`, `\alpha`, digits, operators) — not character- or word-level | Hugging Face `tokenizers` |
| **Encoder** | Image → 2D grid of feature vectors, preserving spatial layout (math is 2D, not linear) | PyTorch; CNN (ResNet-style) or small ViT |
| **Attention** | Bridges encoder features to decoder steps; attention maps double as a debugging/explainability view | PyTorch |
| **Decoder** | Autoregressive generation of LaTeX tokens | PyTorch / Hugging Face `transformers` (Transformer decoder, preferred over LSTM) |
| **Decoding** | Beam search, width ≈ 5–10, at inference | PyTorch |
| **Postprocessing / validation** | Brace/bracket balance, headless compile check, token-repair heuristics, confidence scoring | KaTeX (headless, Node) or `matplotlib.mathtext` |
| **Data pipeline** | Dataset download/prep, synthetic LaTeX→image rendering, camera-artifact augmentation, split management | `pdflatex` / `matplotlib`, Albumentations, NumPy |
| **Experiment tracking** | Run config, metrics, checkpoints, dataset version | Weights & Biases *or* MLflow (choice open — see DECISIONS.md D-005) |
| **Serving** | Inference API wrapping the trained model | FastAPI; ONNX Runtime or TorchServe for the runtime |
| **Frontend (optional)** | Camera capture, crop/adjust, rendered preview, edit/correct | React Native or Flutter; KaTeX / MathJax |
| **Infra** | Reproducible environments, GPU training | Docker; cloud or local GPU |

---

## 4. Milestones

Build order. Each milestone's "done when" is the gate to starting the next.

### M1 — Environment + data pipeline
Repo scaffolding, pinned Python environment, preprocessing module, im2latex-100k
download/prep, synthetic LaTeX→image renderer, fixed train/val/test splits.
**Done when:** `preprocess(image) → normalized tensor` is unit-tested; im2latex-100k is
prepared and split reproducibly from a single command; the synthetic renderer produces
image/LaTeX pairs; the tokenizer round-trips a held-out set of LaTeX strings losslessly.

### M2 — Baseline model on printed text
CNN/ViT encoder + Transformer decoder trained on im2latex-100k. Greedy decoding.
**Done when:** a training run completes end-to-end, is tracked (N-3), and produces a
checkpoint plus a baseline BLEU / exact-match number on the printed test split, recorded
in this document under §5.

### M3 — Beam search + validation layer
Beam search decoding, brace/bracket checks, headless compile check, repair heuristics,
confidence scoring.
**Done when:** R-4, R-5, R-6 are satisfied; beam search measurably beats greedy on the
printed test split; compilation validity rate is reported.

### M4 — Handwriting fine-tuning
Incorporate CROHME (2014/2016/2019); InkML → image rendering; fine-tune and measure the
printed→handwritten domain gap.
**Done when:** R-2 is satisfied; handwritten test-split metrics are reported alongside
printed, and the gap is quantified.

### M5 — Camera-realistic augmentation + custom data
Photo-realistic augmentation (lighting, shadow, paper texture, blur, perspective skew);
collect and label a small photographed handwritten set.
**Done when:** the augmentation pipeline is applied in training, and a held-out
photographed-paper test set exists with reported metrics.

### M6 — Serving layer
Inference API around the trained model.
**Done when:** R-7 is satisfied — the API accepts an image and returns LaTeX +
confidence + validity, containerized, with the request path documented.

### M7 — (Optional) Mobile/web frontend
Capture → preview → edit loop.

### M8 — (Optional) Active learning loop
User corrections logged as retraining data.

---

## 5. Current progress

**Status: pre-M1.** Documentation only; no code has been written.

| Milestone | Status | Notes |
|---|---|---|
| M1 Environment + data pipeline | Not started | — |
| M2 Baseline (printed) | Not started | — |
| M3 Beam search + validation | Not started | — |
| M4 Handwriting fine-tuning | Not started | — |
| M5 Camera augmentation + custom data | Not started | — |
| M6 Serving layer | Not started | — |
| M7 Frontend (optional) | Not started | — |
| M8 Active learning (optional) | Not started | — |

**Artifacts that exist:** `README.md`, `docs/project_spec.md`, `docs/architecture.md`,
`docs/PLAN.md`, `docs/DECISIONS.md`.

**Measured results:** none yet. Baseline numbers land here as M2–M5 complete.

---

## 6. Definition of done

### 6.1 v1 is done when

1. All in-scope functional requirements R-1 … R-8 are implemented and demonstrated.
2. Milestones M1 through M6 meet their "done when" gates.
3. A single documented command reproduces evaluation of a released checkpoint on the
   printed, handwritten, and photographed test splits, reporting all five metrics in §2.4.
4. The system meets the accuracy bar in §6.2 on the photographed handwritten test set.
5. Non-functional requirements N-1 … N-5 hold.
6. The recognition path contains no general-purpose multimodal LLM API call.

### 6.2 Accuracy bar — **proposed, pending approval**

These numbers are not yet approved requirements. They are recorded here so M2–M5 have a
target to aim at; the owner should confirm, adjust, or replace them before M2 closes.

| Split | Exact match | Compilation validity | Render match |
|---|---|---|---|
| Printed (im2latex-100k test) | ≥ 75% | ≥ 98% | ≥ 85% |
| Handwritten (CROHME test) | ≥ 45% | ≥ 95% | ≥ 60% |
| Photographed handwritten (custom) | ≥ 35% | ≥ 95% | ≥ 50% |

### 6.3 Definition of done for any individual change

- Corresponds to a numbered plan item (a requirement or a milestone).
- Is the smallest change that satisfies that item.
- Has tests appropriate to its layer, and relevant existing tests still pass.
- Does not weaken a constraint in §2 to make itself easier.
- §5 is updated if — and only if — implementation status materially changed.

---

## 7. Key risks and open questions

Carried from [project_spec.md](./project_spec.md) §8; resolutions are recorded in
[DECISIONS.md](./DECISIONS.md) as they are made.

- **Domain gap.** CROHME is stylus-on-tablet ink; the product target is pen-on-paper
  photographs. Custom data collection (M5) may need more investment than planned.
- **Ambiguous handwriting.** `x`/`×`, `1`/`l`/`I`, `0`/`O` are ambiguous even to humans
  without context. Open: how the model should flag versus guess. (Open question Q-1.)
- **Complex layouts.** Nested fractions, multi-level scripts, matrices stress structural
  understanding — warrants a dedicated "structurally complex" eval subset. (Q-2.)
- **Compute budget.** Training a decoder from scratch needs meaningful GPU time; whether
  to initialize from open pretrained vision weights is D-002.
- **Future multi-expression support.** Deferred to v2, but whether the encoder should be
  designed for future region segmentation is a v1 architecture question. (Q-3.)
