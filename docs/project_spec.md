# Project Spec: Image-to-LaTeX Math Recognition System

## 1. Overview

**Goal:** Build a system that takes a photograph or scan of handwritten or printed mathematical notation and outputs correct, compilable LaTeX source representing that notation.

**Non-goal / constraint:** No reliance on general-purpose multimodal LLM APIs (e.g., calling GPT-4V or Claude's vision endpoint) as the recognition engine. The system should perform its own image parsing and structure recognition via a purpose-built model/pipeline that we train and control.

**Primary use case:** User photographs handwritten math (homework, notes, textbook problems) on a phone; app returns LaTeX that renders identically (or near-identically) to the original.

---

## 2. Scope

### In scope (v1)
- Single mathematical expression per image (not full documents/pages)
- Handwritten and printed input
- Core LaTeX constructs: fractions, exponents/subscripts, roots, Greek letters, common operators, summations/integrals, parentheses/brackets, basic matrices
- Output: valid, compilable LaTeX string

### Out of scope (v1, candidate for v2+)
- Full-page/document layout (multiple expressions, mixed text+math, paragraph structure)
- Diagrams, graphs, geometric figures
- Multi-line/aligned equation systems (`align`, `cases` environments) — stretch goal
- Real-time video/live camera recognition
- Non-Latin symbol sets beyond standard math notation

---

## 3. System Architecture

```
[Image Input]
     │
     ▼
[Preprocessing Module]
     │
     ▼
[Recognition Model] ──► [Token Sequence (LaTeX tokens)]
     │
     ▼
[Postprocessing / Validation]
     │
     ▼
[LaTeX Output] ──► [Render Preview (optional)]
```

### 3.1 Preprocessing Module
Responsibilities:
Stages run in this order; the sequence is significant (see DECISIONS.md D-008):

- Convert to grayscale
- Denoise (median/Gaussian filtering)
- Contrast normalization for uneven lighting (CLAHE)
- Binarize (Otsu or adaptive threshold)
- Deskew (Hough transform-based angle detection + rotation correction)
- Crop to content bounding box, normalize scale/aspect ratio

**Tech:** OpenCV (Python), NumPy

### 3.2 Recognition Model (core "real parsing" component)
Encoder-decoder architecture, trained from scratch (or fine-tuned from an open checkpoint) — this is the piece that replaces an LLM-vision-API wrapper.

- **Encoder:** CNN backbone (ResNet-style or a small Vision Transformer) that converts the preprocessed image into a 2D grid of feature vectors, preserving spatial layout information (critical since math is 2D, not linear).
- **Attention mechanism:** Bridges encoder features to decoder steps, allowing the decoder to attend to relevant image regions at each generation step (also enables optional attention-map visualization for debugging/explainability).
- **Decoder:** Transformer decoder (preferred over LSTM for training stability and parallelism) that autoregressively generates a sequence of LaTeX tokens.
- **Tokenizer:** Custom vocabulary of LaTeX tokens (commands, symbols, braces) rather than character-level or English-word tokenization — e.g., `\frac`, `{`, `}`, `^`, `_`, `\alpha`, digits, operators.
- **Decoding strategy:** Beam search (beam width ~5–10) at inference for higher-quality sequences vs. greedy decoding.

**Tech:** PyTorch, Hugging Face `transformers` (for transformer decoder scaffolding), or a from-scratch implementation for full control.

### 3.3 Postprocessing / Validation Module
- Brace/bracket balance checking
- Attempt LaTeX compilation (headless, via a minimal LaTeX engine or `matplotlib`'s mathtext / `KaTeX` in a sandboxed renderer) to catch invalid output
- Fallback/error handling: if compilation fails, attempt token-level repair heuristics or return best partial result with confidence flag
- Confidence scoring per output (e.g., decoder token probabilities) to flag low-confidence recognitions for user review

### 3.4 (Optional) Frontend / App Layer
- Mobile capture UI (camera, crop/adjust before submission)
- Rendered LaTeX preview (KaTeX or MathJax) alongside raw source
- Edit/correct interface — corrections can be logged for future retraining data (active learning loop)

**Tech:** React Native or Flutter (mobile), KaTeX (rendering), backend API (FastAPI/Flask) serving the model

---

## 4. Data

| Dataset | Content | Use |
|---|---|---|
| **im2latex-100k** | ~100k printed LaTeX formulas rendered as images | Pretraining / printed-text baseline |
| **CROHME** (2014/2016/2019) | Handwritten math expressions with structural ground truth (InkML) | Handwriting fine-tuning |
| **Synthetic data** | LaTeX source → rendered image (via `pdflatex`/`matplotlib`), with augmentation (rotation, noise, blur, warping) to simulate photographed handwriting | Data augmentation, filling gaps |
| **Self-collected** | Photographed handwritten math samples (own handwriting, volunteers) with manually labeled LaTeX | Domain-specific fine-tuning, closes the "photo of paper" gap that CROHME (stylus/tablet ink) doesn't fully cover |

**Data pipeline needs:**
- Image augmentation library (Albumentations) to simulate camera artifacts: lighting variance, paper texture, shadows, slight blur, perspective skew
- Train/val/test split with care to avoid symbol-distribution leakage
- Annotation tooling if collecting custom handwritten data (or use existing InkML → image renderers for CROHME)

---

## 5. Technology Stack Summary

| Layer | Technology |
|---|---|
| Preprocessing | Python, OpenCV, NumPy |
| Model training | PyTorch, Hugging Face `transformers`/`tokenizers` |
| Experiment tracking | Weights & Biases or MLflow |
| Data augmentation | Albumentations |
| Validation/render check | KaTeX (Node-based headless render) or `matplotlib.mathtext` |
| Serving/backend | FastAPI, ONNX Runtime or TorchServe for inference |
| Frontend (optional) | React Native / Flutter + KaTeX/MathJax for preview |
| Infra | Docker for reproducibility; GPU instance (cloud or local) for training |

---

## 6. Evaluation Metrics

- **Exact match accuracy:** predicted LaTeX string == ground truth (post-normalization, e.g., whitespace-insensitive)
- **BLEU score:** token-level sequence similarity (standard in im2latex literature)
- **Edit distance (Levenshtein) on token sequence**
- **Render match:** does the rendered image of predicted LaTeX visually match rendered ground truth (image-level comparison) — more forgiving of equivalent-but-differently-written LaTeX (e.g., `x^2` vs `x^{2}`)
- **Compilation validity rate:** % of outputs that compile without error

---

## 7. Milestones / Build Order

1. **Environment + data pipeline**: set up preprocessing, download/prepare im2latex-100k, build synthetic renderer
2. **Baseline model on printed text**: CNN encoder + Transformer decoder trained on im2latex-100k; establish baseline BLEU/exact-match
3. **Beam search + validation layer**: improve inference quality, add compile-check postprocessing
4. **Handwriting fine-tuning**: incorporate CROHME, evaluate domain gap between printed and handwritten performance
5. **Camera-realistic augmentation**: add photo-realistic augmentations (lighting, skew, texture); collect small custom photographed dataset for fine-tuning/testing
6. **Serving layer**: wrap trained model in an inference API
7. **(Optional) Mobile/web frontend**: capture → preview → edit loop
8. **(Optional) Active learning loop**: capture user corrections to continuously improve the model

---

## 8. Key Risks / Open Questions

- **Domain gap**: CROHME (stylus-on-tablet ink data) may not generalize well to photographed pen-on-paper images; may need to invest more in custom data collection than expected.
- **Ambiguous handwriting**: symbols like `x`/`×`, `1`/`l`/`I`, `0`/`O` are genuinely ambiguous even to humans without context — decide how the model should handle/flag ambiguity.
- **Complex layouts**: nested fractions, multi-level sub/superscripts, and matrices stress-test the model's structural understanding — worth having a specific eval subset for "structurally complex" expressions.
- **Compute budget**: training a transformer decoder from scratch requires meaningful GPU time; decide early whether to fine-tune an existing open checkpoint (e.g., pix2tex/LaTeX-OCR weights) vs. fully from-scratch, given the "no LLM wrapper" constraint applies to the *product's runtime recognition engine*, not necessarily to ruling out reusing open pretrained CV/vision-encoder weights as a starting point.
- **Multi-expression/page support**: explicitly deferred to v2, but worth deciding now whether the model architecture should be designed with future extensibility toward segmentation of multiple expression regions per image.
