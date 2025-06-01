import torch
from transformers import BertForSequenceClassification
from utils import clean_text, get_tokenizer

def predict(text, model_dir="model", threshold=0.5):
    tokenizer = get_tokenizer().from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(clean_text(text), return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    return pred == 1, confidence