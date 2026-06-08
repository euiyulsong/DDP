# 07_hf_trainer_embedding_triplet.py
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


def build_triplet_dataset():
    queries = ["refund policy", "reset password", "cancel order"] * 200
    positives = [
        "Refunds are available within 30 days.",
        "Password can be reset in account settings.",
        "Orders can be cancelled before shipment.",
    ] * 200
    negatives = [
        "Change your profile image in settings.",
        "Shipping usually takes three business days.",
        "You can update your payment method.",
    ] * 200

    return Dataset.from_dict(
        {
            "query": queries,
            "positive": positives,
            "negative": negatives,
        }
    )


class TripletEmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.mean_pooling(out.last_hidden_state, attention_mask)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        positive_input_ids,
        positive_attention_mask,
        negative_input_ids,
        negative_attention_mask,
    ):
        q = self.encode(query_input_ids, query_attention_mask)
        p = self.encode(positive_input_ids, positive_attention_mask)
        n = self.encode(negative_input_ids, negative_attention_mask)

        pos_sim = F.cosine_similarity(q, p)
        neg_sim = F.cosine_similarity(q, n)

        margin = 0.2
        loss = F.relu(margin - pos_sim + neg_sim).mean()

        return {
            "loss": loss,
            "pos_sim": pos_sim.mean(),
            "neg_sim": neg_sim.mean(),
        }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = build_triplet_dataset().train_test_split(test_size=0.1)

    def tokenize(batch):
        q = tokenizer(batch["query"], padding="max_length", truncation=True, max_length=64)
        p = tokenizer(batch["positive"], padding="max_length", truncation=True, max_length=128)
        n = tokenizer(batch["negative"], padding="max_length", truncation=True, max_length=128)

        return {
            "query_input_ids": q["input_ids"],
            "query_attention_mask": q["attention_mask"],
            "positive_input_ids": p["input_ids"],
            "positive_attention_mask": p["attention_mask"],
            "negative_input_ids": n["input_ids"],
            "negative_attention_mask": n["attention_mask"],
        }

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["query", "positive", "negative"],
    )

    model = TripletEmbeddingModel(MODEL_NAME)

    args = TrainingArguments(
        output_dir="./outputs/embedding-triplet",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        logging_steps=20,
        remove_unused_columns=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    trainer.train()


if __name__ == "__main__":
    main()
