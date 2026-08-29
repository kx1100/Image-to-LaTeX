# Image-to-LaTeX

Convert photos or scans of handwritten and printed math into compilable LaTeX. Powered by a purpose-built encoder-decoder recognition model.

## Overview

Point your phone at handwritten homework, notes, or a textbook problem, and get back LaTeX source that renders to the original. 
This project trains and controls the recognition engine (image parsing and structure recognition) used to do so. 

## Features (v1 scope)

- Recognizes a single mathematical expression per image
- Supports both handwritten and printed input
- Handles core LaTeX constructs: fractions, exponents/subscripts, roots, Greek letters, common operators, summations/integrals, parentheses/brackets, basic matrices
- Outputs valid, compilable LaTeX
- Optional confidence scoring and rendered preview for low-confidence flags

**Not yet supported** (in the future!): full-page/document layout, diagrams and figures, multi-line/aligned equation systems, real-time video recognition, non-Latin symbol sets.

## Architecture

```
Image → Preprocessing → Recognition Model → Postprocessing/Validation → LaTeX Output
```

| Stage | What it does | Tech |
|---|---|---|
| **Preprocessing** | Grayscale, denoise, contrast correction, binarize, deskew, crop/normalize | OpenCV, NumPy |
| **Recognition Model** | CNN/ViT encoder → attention → Transformer decoder generates LaTeX tokens via beam search | PyTorch, Hugging Face `transformers`/`tokenizers` |
| **Postprocessing/Validation** | Brace/bracket checks, headless compile check, repair heuristics, confidence scoring | KaTeX (headless) / `matplotlib.mathtext` |
| **Serving** | Wraps the trained model behind an inference API | FastAPI, ONNX Runtime / TorchServe |
| **Frontend (optional)** | Capture, crop, preview, and correct recognized output | React Native / Flutter, KaTeX / MathJax |

See [`architecture.mermaid`](./image-to-latex-architecture.mermaid) for the full system diagram, and [`spec.md`](./image-to-latex-spec.md) for the detailed project spec.

## Data

| Dataset | Content | Use |
|---|---|---|
| im2latex-100k | ~100k printed LaTeX formulas rendered as images | Pretraining / printed baseline |
| CROHME (2014/2016/2019) | Handwritten expressions with structural ground truth (InkML) | Handwriting fine-tuning |
| Synthetic renders | LaTeX → rendered image via `pdflatex`/`matplotlib`, augmented | Filling data gaps |
| Self-collected | Photographed handwritten samples, manually labeled | Closing the photo-of-paper domain gap |

## Evaluation

Models are evaluated on exact-match accuracy, BLEU score, token-level edit distance, render match (image-level comparison, tolerant of equivalent LaTeX like `x^2` vs `x^{2}`), and compilation validity rate.

## Roadmap

1. Environment + data pipeline setup
2. Baseline model on printed text (im2latex-100k)
3. Beam search + validation layer
4. Handwriting fine-tuning (CROHME)
5. Camera-realistic augmentation + custom photo dataset
6. Serving layer
7. Mobile/web frontend
8. Active learning loop (user corrections → retraining data)

## Status

Currently still in development!

## Getting Started

> _Setup instructions coming soon

## Contributing

me
