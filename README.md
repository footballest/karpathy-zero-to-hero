# karpathy-zero-to-hero

My implementations of Andrej Karpathy's **Neural Networks: Zero to Hero** series.

## Layout

- `src/` – Python packages and modules
  - `micrograd/` – autodiff engine + tiny neural nets
  - `makemore/` – character-level language models
  - `gpt/` – transformer / GPT-style models
  - `utils/` – shared helpers
- `notebooks/` – experiments and visualizations
- `data/` – datasets (not tracked in git by default)
- `tests/` – unit tests and sanity checks
- `docs/` – extra notes

## Setup (short version)

1. Create a virtual environment (PyCharm can do this automatically).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run code from inside the `src/` directory (marked as *Sources Root* in PyCharm).
