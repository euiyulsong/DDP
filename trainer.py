# 05_hf_trainer_classification.py
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"


def build_dataset():
    texts = [
        "this movie is great",
        "this movie is terrible",
        "I love this product",
        "I hate this product",
    ] * 100

    labels = [1, 0, 1, 0] * 100

    return Dataset.from_dict({"text": texts, "label": labels}).train_test_split(test_size=0.2)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = build_dataset()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenized = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    args = TrainingArguments(
        output_dir="./outputs/classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model("./outputs/classifier/best")


if __name__ == "__main__":
    main()
