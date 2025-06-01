# vulgar-nlp

A BERT (Bidirectional Encoder Representations from Transformers) based text classification model for detecting vulgar language in English text. This project uses the BERT transformer model to classify English text as either vulgar or non-vulgar

More on BERT: https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/
## Features

- Text classification using BERT (bert-base-uncased)
- Handles common vulgar word variations and substitutions
- Text cleaning and normalization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aandrewduong/vulgar-nlp.git
cd vulgar-nlp
```

2. Install the required dependencies:
```bash
pip install torch transformers datasets
```

## Usage

### Training

To train the model on your own dataset (example dataset has been given):
1. Prepare your data in CSV (Spreadsheet) format with columns:
   - `text`: The input text
   - `label`: 0 for non-vulgar, 1 for vulgar

2. Run the training script:
```bash
python train.py
```

The model will be saved in the `model` directory.

### Prediction

To use the trained model for predictions:

```python
from predictor import predict

# Example usage
text = "Your text here"
is_vulgar, confidence = predict(text)
print(f"Is vulgar: {is_vulgar}, Confidence: {confidence:.2f}")
```

```bash
python cli.py --text "Shut up"
```

## Model Details

- Base Model: BERT (bert-base-uncased https://huggingface.co/google-bert/bert-base-uncased)
- Max sequence length: 64 tokens
- Binary classification (vulgar/non-vulgar)
- Training epochs: 4
- Batch size: 8

## Text Cleaning

The model includes a text cleaning pipeline that handles:
- Case normalization
- Common vulgar word variations
- Special character substitutions
