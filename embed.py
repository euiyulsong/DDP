# 06_hf_trainer_embedding_infonce.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_pair_dataset():
    queries = [
        "how to reset password",
        "refund policy",
        "change delivery address",
        "cancel my order",
    ] * 200

    positives = [
        "You can reset your password from account settings.",
        "Refunds are available within 30 days.",
        "Delivery address can be changed before shipment.",
        "You can cancel an order before it is shipped.",
    ] * 200

    return Dataset.from_dict(
        {
            "query": queries,
            "positive": positives,
        }
    ).train_test_split(test_size=0.1)


class EmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        emb = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        positive_input_ids,
        positive_attention_mask,
    ):
        q_emb = self.encode(query_input_ids, query_attention_mask)
        p_emb = self.encode(positive_input_ids, positive_attention_mask)

        logits = q_emb @ p_emb.T
        logits = logits / 0.05

        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = build_pair_dataset()

    def tokenize(batch):
        q = tokenizer(
            batch["query"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )
        p = tokenizer(
            batch["positive"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        return {
            "query_input_ids": q["input_ids"],
            "query_attention_mask": q["attention_mask"],
            "positive_input_ids": p["input_ids"],
            "positive_attention_mask": p["attention_mask"],
        }

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["query", "positive"],
    )

    model = EmbeddingModel(MODEL_NAME)

    args = TrainingArguments(
        output_dir="./outputs/embedding-infonce",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    trainer.train()
    trainer.save_model("./outputs/embedding-infonce/final")


if __name__ == "__main__":
    main()
