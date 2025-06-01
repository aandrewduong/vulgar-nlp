import re
from transformers import BertTokenizerFast

def clean_text(text):
    substitutions = {
        r"f[\W_]*u[\W_]*c[\W_]*k": "fuck",
        r"s[\W_]*h[\W_]*i[\W_]*t": "shit",
        r"b[\W_]*i[\W_]*t[\W_]*c[\W_]*h": "bitch",
        r"d[\W_]*a[\W_]*m[\W_]*n": "damn",
    }
    text = text.lower()
    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)
    return text

def get_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")