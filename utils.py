import re
import unicodedata
from typing import Dict
from transformers import AutoTokenizer

# Precompiled patterns and mappings used during normalization
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Comprehensive leetspeak/obfuscation normalization
OBFUSCATION_MAP: Dict[str, str] = {
    "0": "o",
    "1": "i",
    "l": "i",
    "3": "e",
    "4": "a",
    "6": "g",
    "9": "g",
    "@": "a",
    "$": "s",
    "5": "s",
    "7": "t",
    "+": "t",
    "8": "b",
    "!": "i",
    "|": "i",
    "*": "",  # Often used to censor: f*ck -> fck
}

# Canonicalize common vulgar word variants to reduce sparse forms
VULGAR_PATTERNS = {
    r"f[\W_]*u[\W_]*c[\W_]*k": "fuck",
    r"s[\W_]*h[\W_]*i[\W_]*t": "shit",
    r"b[\W_]*i[\W_]*t[\W_]*c[\W_]*h": "bitch",
    r"d[\W_]*a[\W_]*m[\W_]*n": "damn",
    r"a[\W_]*s[\W_]*s": "ass",
    r"c[\W_]*u[\W_]*n[\W_]*t": "cunt",
    r"p[\W_]*u[\W_]*s[\W_]*s[\W_]*y": "pussy",
    r"d[\W_]*i[\W_]*c[\W_]*k": "dick",
}

def normalize_text(text: str) -> str:
    """Normalize unicode, remove zero-width chars, basic leetspeak, and whitespace.

    This function is intentionally conservative to avoid changing semantics while
    still improving model robustness against common obfuscations.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = text.lower()
    text = "".join(OBFUSCATION_MAP.get(ch, ch) for ch in text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text

def clean_text(text: str) -> str:
    """Apply normalization and collapse common vulgar variants.

    Also reduces long character repetitions (e.g., "fuuuuuck" -> "fuuck").
    """
    text = normalize_text(text)
    for pattern, replacement in VULGAR_PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    # Collapse 3+ repeats down to 2 (keeps some emphasis without exploding vocab)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text

def get_tokenizer(model_name_or_dir: str = "bert-base-uncased"):
    """Create a fast tokenizer for the given model name or local directory."""
    return AutoTokenizer.from_pretrained(model_name_or_dir, use_fast=True)