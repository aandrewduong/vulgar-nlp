# vulgar-nlp

A BERT (Bidirectional Encoder Representations from Transformers) based text classification model for detecting vulgar language in English text. This project uses the BERT transformer model to classify English text as either vulgar or non-vulgar.

More on BERT: https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/

## Features

- **Text classification using BERT** (`bert-base-uncased`)
- **Robust text normalization**: Unicode NFKC, zero-width removal, expanded leetspeak/obfuscation normalization, repetition collapsing.
- **Improved Predictor**: Model and tokenizer are cached in memory for high-performance inference.
- **Flexible Training**: Configurable via CLI arguments (epochs, batch size, learning rate, base model).
- **Efficient CLI**: Supports single text or file input with memory-efficient batch processing and JSON output.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aandrewduong/vulgar-nlp.git
cd vulgar-nlp
```

2. Set up a virtual environment and install dependencies:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train the model on your own dataset (example `data.csv` provided):

```bash
# Run with default settings
python3 train.py

# Run with custom parameters
python3 train.py --epochs 5 --batch-size 16 --lr 3e-5 --model-name distilbert-base-uncased
```

The model will be saved in the `model` directory.

### Prediction

#### Python API
The `predictor` module caches the model automatically after the first call.

```python
from predictor import predict

# Example usage
text = "Your text here"
is_vulgar, confidence = predict(text)
print(f"Is vulgar: {is_vulgar}, Confidence: {confidence:.2f}")
```

#### CLI

```bash
# Single text
python3 cli.py --text "Shut up"

# From a file (processes in memory-efficient batches)
python3 cli.py --file path/to/texts.txt --batch-size 64

# JSON output for integration
python3 cli.py --file path/to/texts.txt --json
```

## Model Details

- **Base Model**: BERT (`bert-base-uncased`) or DistilBERT (`distilbert-base-uncased`)
- **Max sequence length**: 128 tokens
- **Binary classification**: `clean` (0) or `vulgar` (1)
- **Early Stopping**: Automatically stops training when performance (F1) plateaus.

## Text Cleaning

The pipeline handles various obfuscation techniques:
- **Unicode normalization** (NFKC) and zero-width character removal.
- **Expanded Leetspeak**: Normalizes `@`->`a`, `0`->`o`, `!`->`i`, `1`/`l`/`|`->`i`, `5`/$->`s`, `6`/`9`->`g`, etc.
- **Censorship removal**: Handles `*` used to censor letters (e.g., `f*ck` -> `fck`).
- **Vulgar Word Canonicalization**: Uses regex to find and normalize common vulgar words across various spacings and punctuation.
- **Repetition Collapsing**: Collapses `fuuuuuck` to `fuuck` to reduce vocabulary sparsity while maintaining emphasis.
