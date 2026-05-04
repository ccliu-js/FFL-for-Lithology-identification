# FFL-for-Lithology-identification

This repository contains the implementation for **FFL-for-Lithology-identification**.
A study on few-shot lithology identification using drilling vibration signals,
based on metric-based meta-learning, enabling effective classification under
limited labeled data conditions.

It includes data loading, signal slicing, episodic few-shot sampling, model
training, repeated evaluation, and cross-speed testing.

The project is organized as a uv-managed Python project. The recommended entry
point is `main.py`, with all experiment settings defined in `config.yaml`.

## Project Structure

```text
.
├── config.yaml             # Shared experiment, data, and model configuration
├── main.py                 # Unified command-line entry point
├── OpenSrc/
│   ├── NN/                 # Model, loss, backbone, and network modules
│   ├── utils/              # Data loading, slicing, and episodic dataset tools
│   ├── infer.py            # Evaluation and confusion-matrix utilities
│   └── train.py            # Trainer class
├── dataset/
│   └── dataset.pkl         # Processed dataset file
├── weights/
│   └── best_model.pth      # Model checkpoint
├── pyproject.toml          # uv project configuration
└── README.md
```

## Requirements

- Python 3.10 or later
- uv
- CUDA-compatible PyTorch environment, if GPU acceleration is needed

Install uv if it is not available on your system:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the project dependencies:

```bash
uv sync
```

## Dataset and Checkpoints

Dataset:

```text
https://zenodo.org/records/19702433
```

CheckPoints:

```text
https://huggingface.co/chengduliu/few-shot-lithology/tree/main
```

After downloading, place the processed dataset and checkpoint according to
`config.yaml`, for example:

```text
dataset/dataset.pkl
weights/best_model.pth
```

## Quick Start

Run training:

```bash
uv run python main.py train
```

Run repeated evaluation and draw confusion matrices:

```bash
uv run python main.py infer-multi
```

Evaluate the trained checkpoint under different speeds:

```bash
uv run python main.py test-diff-speed
```

Use a custom configuration file:

```bash
uv run python main.py --config config.yaml train
```

## Configuration

All main experiment settings are defined in `config.yaml`.

The `common` section stores shared paths, runtime settings, slicing parameters,
and split ratio:

```yaml
common:
  paths:
    data: ./dataset/dataset.pkl
    weights: ./weights/best_model.pth
  runtime:
    device: auto
    seed: 42
  slicing:
    slice_len: 512
    overlap: 0.0
    drop_last: true
  split:
    ratio: 0.8
```

The `model` section controls model construction, including the prototypical
network scale, STFT settings, time-domain encoder, frequency encoder, fusion
module, and Bayesian embedding size.

The `train`, `infer_multi`, and `test_diff_speed` sections control the three
main experiment modes.

## Data Preparation

The default configuration expects a processed pickle file at:

```text
dataset/dataset.pkl
```

The expected processed data structure is:

```python
{
    "r300": {
        "Acc_Z": {
            "Arkose": [...],
            "Sandstone": [...],
            "Marble": [...],
            "Granite": [...],
            "Marlstone": [...],
            "Limestone": [...],
            "Shale": [...]
        }
    }
}
```

Each rock class should contain a list of one-dimensional signal samples. During
training and evaluation, each signal is sliced into fixed-length segments and
sampled as few-shot episodes.

If you start from raw CSV files, see:

```text
OpenSrc/utils/load_data.py
OpenSrc/utils/slice_data.py
```

## Experiment Modes

### Training

```bash
uv run python main.py train
```

This command trains the Bayesian prototypical network using the `train` section
of `config.yaml`. The best checkpoint is saved to the path configured under:

```yaml
common:
  paths:
    weights: ./weights/best_model.pth
```

### Repeated Inference

```bash
uv run python main.py infer-multi
```

This command repeatedly samples support/query sets from a selected speed and
axis, computes accuracy statistics, and optionally saves confusion-matrix
figures.

### Cross-Speed Testing

```bash
uv run python main.py test-diff-speed
```

This command loads the full dataset, keeps the configured sensor axis, and
evaluates the checkpoint across the available speeds in the dataset.
