# AI-ML

A cleaned-up collection of AI/ML notebooks and references, organized by topic.

## Repository layout

```text
.
├── Autoencoders/
├── Documentation/
├── GAN/
├── Reinforcement Learning/
├── Unified mentor/
├── diffusion-models/
├── keras-tuner/
├── model_optimization(TF)/
├── regression Models/
├── tensorflow_usl/
├── .vscode/
├── ML_workflow.excalidraw
├── confusion_matrix_rf.png
├── training_history.png
└── README.md
```

## Notebook and project organization

- Topic folders group related experiments.
- Notebook filenames should be descriptive and stable, avoiding trailing underscores or duplicate suffixes.
- Keep supporting markdown notes in `Documentation/`.
- Keep images and plots at the top level only if they are project-wide assets; otherwise place them next to the notebook they support.

## Notebooks from `Transformers_ML`

Recommended moves into this repo:

- `DP_.ipynb` → `Documentation/` or a dedicated preprocessing folder
- `RLT_.ipynb` → `Reinforcement Learning/`
- `SRT_.ipynb` → `tensorflow_usl/` or a speech/sequence topic folder
- `TTSD_.ipynb` → `tensorflow_usl/` or a text-to-speech topic folder
- `VIT_.ipynb` → `Documentation/` or a computer vision topic folder
- `VIT_.py` → remove if it is empty, or place beside `VIT_.ipynb` if it contains code

## Cleanup suggestions

- Standardize names to lowercase snake_case or clear topic names.
- Add a top-level `notebooks/` index file if you want a single landing page.
- Add `requirements.txt` or `environment.yml` if dependencies are missing.
- Add brief README files inside major folders to explain their contents.
