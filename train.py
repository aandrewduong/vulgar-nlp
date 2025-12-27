import argparse
import random
import numpy as np
import torch
import evaluate
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


def main():
    parser = argparse.ArgumentParser(description="Train a vulgar language classifier.")
    parser.add_argument("--data", type=str, default="data.csv", help="Path to CSV data")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Base model name")
    parser.add_argument("--output-dir", type=str, default="model", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load CSV and split
    raw = load_dataset("csv", data_files=args.data)["train"]
    raw = raw.train_test_split(test_size=0.2, seed=args.seed)

    tokenizer = get_tokenizer(args.model_name)
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
        args.model_name, num_labels=2, id2label=id2label, label2id=label2id
    )

    # Evaluation metrics
    metrics = {
        "accuracy": evaluate.load("accuracy"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
        "f1": evaluate.load("f1"),
    }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        res = {}
        for name, metric in metrics.items():
            if name == "accuracy":
                res[name] = metric.compute(predictions=preds, references=labels)["accuracy"]
            else:
                res[name] = metric.compute(predictions=preds, references=labels, average="binary")[name]
        return res

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
        output_dir=args.output_dir,
        save_strategy="epoch",
        eval_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=10,  # More frequent logging for small datasets
        report_to=[],
        seed=args.seed,
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

    trainer.train()
    trainer.evaluate()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
