import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from utils import clean_text, get_tokenizer

# Load CSV
raw = load_dataset("csv", data_files="data.csv")["train"]
raw = raw.train_test_split(test_size=0.2)
tokenizer = get_tokenizer()

def tokenize(batch):
    cleaned = [clean_text(x) for x in batch["text"]]
    return tokenizer(cleaned, truncation=True, padding="max_length", max_length=64)

tokenized = raw.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="model",
    save_strategy="epoch",
    eval_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

trainer.train()
model.save_pretrained("model")
tokenizer.save_pretrained("model")