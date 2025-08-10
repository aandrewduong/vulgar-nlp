import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from utils import clean_text, get_tokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Load CSV and split
raw = load_dataset("csv", data_files="data.csv")["train"]
raw = raw.train_test_split(test_size=0.2, seed=42)

tokenizer = get_tokenizer("bert-base-uncased")
max_length = 128


def tokenize(batch):
    cleaned = [clean_text(x) for x in batch["text"]]
    return tokenizer(cleaned, truncation=True, padding=False, max_length=max_length)


tokenized = raw.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Model with explicit label mapping
id2label = {0: "clean", 1: "vulgar"}
label2id = {"clean": 0, "vulgar": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# Evaluation metrics
import evaluate

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="binary")[
            "precision"
        ],
        "recall": recall.compute(predictions=preds, references=labels, average="binary")[
            "recall"
        ],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }


# Class weights to mitigate imbalance
train_labels = raw["train"]["label"]
num_total = len(train_labels)
num_pos = int(sum(train_labels))
num_neg = num_total - num_pos
if num_pos > 0 and num_neg > 0:
    w_neg = num_total / (2.0 * num_neg)
    w_pos = num_total / (2.0 * num_pos)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
else:
    class_weights = None


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.get("label")
        outputs = model(**{k: v for k, v in inputs.items() if k not in ("labels", "label")})
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="model",
    save_strategy="epoch",
    eval_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    dataloader_pin_memory=False,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_strategy="steps",
    logging_steps=50,
    report_to=[],  # disable integrations by default
    seed=42,
)


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

set_seed(42)
trainer.train()
trainer.evaluate()
model.save_pretrained("model")
tokenizer.save_pretrained("model")