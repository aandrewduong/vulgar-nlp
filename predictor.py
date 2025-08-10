import torch
from torch.nn.functional import softmax
from typing import List, Optional, Union
from transformers import AutoModelForSequenceClassification
from utils import clean_text, get_tokenizer


def _select_device(preferred_device: Optional[str] = None) -> torch.device:
    if preferred_device:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Prefer Apple Silicon GPU if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir: str = "model", device: Optional[str] = None):
    """Load tokenizer and model, moved to the requested device."""
    tokenizer = get_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(_select_device(device))
    return tokenizer, model


def predict(
    text: Union[str, List[str]],
    model_dir: str = "model",
    threshold: float = 0.5,
    device: Optional[str] = None,
    max_length: int = 128,
):
    """Predict whether input text is vulgar.

    - Accepts a single string or a list of strings
    - Returns (is_vulgar, confidence) for a single string
    - Returns (is_vulgar_list, confidence_list) for a list
    """
    is_single = isinstance(text, str)
    texts = [text] if is_single else list(text)

    tokenizer, model = load_model(model_dir, device=device)
    inputs = tokenizer(
        [clean_text(t) for t in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1).detach().cpu()

    vulgar_probs = probs[:, 1].tolist()
    predictions = [p >= threshold for p in vulgar_probs]

    if is_single:
        return predictions[0], vulgar_probs[0]
    return predictions, vulgar_probs