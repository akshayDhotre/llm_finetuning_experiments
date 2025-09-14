# LLM Finetuning Experiments

This repository contains experiments for finetuning Large Language Models (LLMs) using [MLX](https://github.com/ml-explore/mlx) and [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685).

## Features

- Finetuning LLMs with MLX for efficient training on Apple Silicon.
- Integration of LoRA for parameter-efficient adaptation.
- Example scripts and utilities for dataset preparation, training, and evaluation.

## Getting Started

### Prerequisites

- Python 3.8+
- MLX installed (`pip install mlx`)
- Other dependencies in `requirements.txt`

### Installation

```bash
git clone https://github.com/yourusername/llm_finetuning_experiments.git
cd llm_finetuning_experiments
uv pip install -r requirements.txt
```

### Usage

1. Prepare your dataset in the required format.
2. Run finetuning with MLX and LoRA:

    ```bash
    python finetune.py --model <base-model> --data <dataset-path> --lora
    ```

3. Evaluate the finetuned model:

    ```bash
    python evaluate.py --model <finetuned-model> --data <eval-dataset>
    ```

## Directory Structure

```
llm_finetuning_experiments/
├── data/
├── finetune.py
├── evaluate.py
├── lora_utils.py
├── requirements.txt
└── README.md
```

## References

- [MLX Documentation](https://github.com/ml-explore/mlx)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## License

MIT License
