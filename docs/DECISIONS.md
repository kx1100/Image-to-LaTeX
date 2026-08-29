# DECISIONS.md — Image-to-LaTeX

Architecture decisions, technology choices, tradeoffs, and things explicitly rejected.

[PLAN.md](./PLAN.md) is the source of truth for *what* is being built; this file records
*why* it is being built that way. Add a new entry rather than editing an old one when a
decision changes — supersede, don't rewrite. Decisions are not reversed without explicit
approval from the project owner.

Status values: **Accepted** · **Proposed** (awaiting approval) · **Open** (undecided) ·
**Superseded by D-nnn**

---

## Architecture decisions

### D-001 — Purpose-built encoder-decoder, not a multimodal LLM API
**Status:** Accepted (project-defining constraint)

The runtime recognition engine is an encoder-decoder model we train and control, not a
call to GPT-4V, Claude vision, or any equivalent general-purpose multimodal API.

**Why:** the point of the project is to own the image parsing and structure recognition.
A wrapper around a vision API would produce a product with no controllable model, no
meaningful evaluation story, per-call cost, and no path to improving on its own data.

**Tradeoff:** substantially more work and GPU spend, and near-term accuracy well below
what a frontier vision model would give out of the box.

**Scope of the constraint:** it binds the *product's runtime recognition path*. It does
not forbid using LLM assistance during development (this file was drafted with it), nor
reusing open pretrained vision weights — see D-002.

---

### D-002 — Open pretrained vision weights are allowed as initialization
**Status:** Proposed — needs owner approval before M2

Initializing the encoder (and possibly the decoder) from open pretrained checkpoints,
including math-OCR checkpoints such as pix2tex / LaTeX-OCR, is permitted. Training fully
from scratch remains an option.

**Why:** D-001 rules out an external recognition *service*, not reusing open weights we
hold and fine-tune. Initialization cuts the compute budget sharply.

**Tradeoff:** less "built from scratch" purity, and inherited architectural constraints
from the checkpoint, in exchange for a far cheaper path to a working baseline.

**Open sub-question:** whether the *first* baseline (M2) should be from scratch to
establish an honest floor, with pretrained initialization as a second run for comparison.

---

### D-003 — Transformer decoder, not LSTM
**Status:** Accepted

**Why:** better training stability and parallelism than an RNN decoder; it is the
standard choice in current im2latex work.

**Tradeoff:** more parameters and more memory per step at inference than a small LSTM.

---

### D-004 — 2D spatial encoder output; the encoder must not flatten early
**Status:** Accepted

The encoder emits a 2D grid of feature vectors and the decoder attends over it.

**Why:** mathematical notation is inherently two-dimensional — superscripts, fraction
numerators and denominators, matrix cells are positional. Collapsing to a 1D sequence
before attention discards the structure the model most needs.

**Secondary benefit:** attention maps over the 2D grid are directly viewable, which makes
recognition failures debuggable.

---

### D-005 — Beam search at inference, width ≈ 5–10
**Status:** Accepted

**Why:** greedy decoding commits early to locally-good tokens and mis-nests structure;
beam search is the standard quality lift in this literature.

**Tradeoff:** ~5–10× the inference compute of greedy. M3 must measure the gain and the
latency cost so the width can be tuned rather than assumed.

---

### D-006 — Validation layer between the model and the user
**Status:** Accepted

Model output is never returned raw. It passes brace/bracket balance checks and a headless
compile check; on failure, token-level repair heuristics run, and an unrepairable result
is returned as a best-partial flagged low-confidence.

**Why:** a sequence model can emit unbalanced or uncompilable LaTeX, and an output the
user cannot compile is worthless regardless of how close it looked.

**Tradeoff:** added latency, and heuristics that can mask genuine model weakness — so
compilation validity rate is tracked as a metric on the *raw* model output too.

---

### D-007 — Custom LaTeX-token vocabulary
**Status:** Accepted

The tokenizer's units are LaTeX tokens (`\frac`, `{`, `}`, `^`, `_`, `\alpha`, digits,
operators), not characters and not English subwords.

**Why:** `\frac` is one semantic unit; spelling it character-by-character wastes decoder
steps and invites malformed commands. A natural-language subword vocabulary is trained on
the wrong distribution entirely.

**Tradeoff:** unseen commands are out-of-vocabulary, so the vocabulary must be derived
from the corpora and its coverage checked (an M1 gate).

---

### D-008 — Contrast normalization runs before binarization
**Status:** Accepted (approved by the owner, 2026-08-29)

The preprocessing sequence is:

```
grayscale → denoise → contrast normalization → binarize → deskew → crop → resize/pad
```

This supersedes the ordering originally written in project_spec.md §3.1 and
architecture.md, which placed contrast normalization last. R-8, architecture.md, the
PLAN.md §3 table, and README.md were updated to match.

**Why:** two reasons, one expected and one found while implementing.

1. A photographed page is rarely lit evenly — a hand shadow or a window leaves a
   gradient across the paper that a threshold reads as ink. Equalizing locally (CLAHE)
   first flattens the gradient while leaving the ink/paper difference intact, so the
   threshold sees an evenly lit page. This is the whole reason R-8 lists contrast
   normalization at all.
2. Running it *after* binarization is not merely a no-op, as first assumed, but
   destructive. CLAHE redistributes the two-valued histogram onto a non-zero floor:
   measured on a test image, background 0 becomes 3, so every pixel is non-zero.
   `crop_to_content` and `deskew` locate ink with `findNonZero`, so both would treat
   the entire canvas as content and silently stop working.

**Tradeoff:** CLAHE can amplify paper texture and sensor noise into something the
threshold then treats as ink, which is why denoise precedes it. Its `clip_limit` is the
knob that trades stroke recovery against false ink, and it is exposed in
`configs/data.yaml` rather than hard-coded.

**How it is enforced:** the stage sequence is configuration, not code
(`configs/data.yaml`, validated at load time), so the ordering can be changed and
measured rather than argued about. Two tests hold the decision in place — one asserts
the repository config keeps contrast normalization ahead of binarization, and one
demonstrates the failure directly on an unevenly lit fixture.

---

## Technology choices

| Area | Chosen | Status | Why this, over what |
|---|---|---|---|
| Language | Python | Accepted | The CV/ML ecosystem is there; Node appears only for headless KaTeX. |
| Preprocessing | OpenCV + NumPy | Accepted | Standard, batteries-included for grayscale/denoise/threshold/Hough; PIL alone lacks deskew and adaptive thresholding. |
| DL framework | PyTorch | Accepted | Research ergonomics, custom attention is easy to write, dominant in this problem's literature. Over TensorFlow/JAX. |
| Decoder scaffolding | Hugging Face `transformers` | Accepted | Avoids rewriting generation, beam search, and caching. Kept swappable for a from-scratch decoder if control is needed. |
| Tokenizer | Hugging Face `tokenizers` | Accepted | Fast, serializable vocabulary with a custom token set (D-007). |
| Augmentation | Albumentations | Accepted | Camera-artifact transforms (perspective, shadow, blur, noise) available directly; `torchvision.transforms` is thinner here. |
| Synthetic rendering | `pdflatex` and/or `matplotlib` | Accepted | `matplotlib.mathtext` needs no TeX install and is fast; `pdflatex` is faithful to real LaTeX. Which is primary is **Open**. |
| Compile check | KaTeX headless *or* `matplotlib.mathtext` | **Open** | KaTeX is stricter and closer to real rendering but adds a Node dependency; mathtext keeps the stack pure-Python but accepts a smaller LaTeX subset. Decide at M3. |
| Experiment tracking | Weights & Biases *or* MLflow | **Open** | W&B is lower-friction and hosted; MLflow is self-hosted with no external data egress. Decide at M2 — N-3 requires one of them. |
| Serving | FastAPI | Accepted | Async, typed, OpenAPI for free. Over Flask. |
| Inference runtime | ONNX Runtime *or* TorchServe | **Open** | ONNX is faster and lighter but export of a beam-search loop is fiddly; TorchServe runs the PyTorch model as-is. Decide at M6. |
| Reproducibility | Docker | Accepted | Required by N-2; CUDA/OpenCV/TeX pinning is not otherwise reliable. |
| Frontend (optional) | React Native or Flutter + KaTeX/MathJax | **Open** | Not needed before M7; deciding now would be premature. |

---

## Tradeoffs accepted

- **Accuracy now vs. ownership.** D-001 costs real accuracy in the near term. Accepted:
  ownership of the engine is the project's purpose.
- **Scope narrowness vs. usefulness.** One expression per image (R-1) makes a large class
  of real photographs — a full page of homework — out of scope for v1. Accepted: single
  expressions are the tractable core, and page segmentation is a separable v2 problem.
- **Validation latency vs. trustworthy output.** D-006 adds a compile step to every
  request. Accepted: uncompilable output is not a product.
- **Beam width vs. inference cost.** D-005, measured at M3 rather than assumed.
- **Synthetic data volume vs. realism.** Synthetic renders are cheap and unlimited but
  are not photographs; the domain gap is closed with augmentation (M5) and real
  photographed data, and the split (§2.4, N-4) keeps synthetic data out of test.
- **Documented targets vs. premature commitment.** The accuracy bar in PLAN.md §6.2 is
  recorded as *proposed* rather than left blank, so milestones have a target — but it is
  explicitly not an approved requirement until the owner confirms it.

---

## Explicitly rejected

| Rejected | Why |
|---|---|
| **Multimodal LLM vision API as the recognition engine** (GPT-4V, Claude vision, Gemini) | The project-defining constraint, D-001. No controllable model, no evaluation story, per-call cost, no path to self-improvement. |
| **Classical symbol segmentation + grammar parsing** (connected components → per-symbol classifier → 2D grammar) | The historical approach; brittle on touching or overlapping handwritten symbols, and the grammar becomes a maintenance sink. End-to-end attention handles ambiguity better. |
| **LSTM decoder** | D-003 — worse training stability, no parallelism over the sequence. |
| **Character-level or English-subword tokenization** | D-007 — wastes decoder steps and produces malformed commands. |
| **Greedy-only decoding at inference** | D-005 — measurably worse structure; greedy is kept only as a training-time/eval baseline. |
| **Returning raw model output to the user** | D-006 — uncompilable LaTeX has no value. |
| **Full-page/document layout, figures, `align`/`cases`, live video, non-Latin symbol sets in v1** | PLAN.md §2.3. Each is a separate problem; bundling them into v1 would prevent v1 from finishing. |
| **Flattening encoder output to 1D before attention** | D-004 — discards the 2D structure that math recognition depends on. |
| **Skipping experiment tracking early on** | Violates N-3. Untracked early runs are unreproducible exactly when the baseline they set matters most. |

---

## Open questions

| ID | Question | Needed by |
|---|---|---|
| Q-1 | How should genuinely ambiguous glyphs (`x`/`×`, `1`/`l`/`I`, `0`/`O`) be handled — best guess, confidence flag, or an alternatives list? | M3 |
| Q-2 | Should there be a dedicated "structurally complex" eval subset (nested fractions, multi-level scripts, matrices)? | M2 |
| Q-3 | Should the v1 encoder be designed for future multi-expression region segmentation, or kept single-expression and revisited in v2? | M2 |
| Q-4 | From-scratch baseline first, or pretrained initialization immediately? (See D-002.) | M2 |
| Q-5 | Are the accuracy targets in PLAN.md §6.2 accepted as requirements? | M2 |
