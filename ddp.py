# 04_ddp_training.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyDataset(Dataset):
    def __init__(self, n=10000, d=20, c=3):
        self.x = torch.randn(n, d)
        W = torch.randn(d, c)
        self.y = (self.x @ W).argmax(dim=-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, d=20, h=128, c=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, c),
        )

    def forward(self, x):
        return self.net(x)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def train():
    local_rank = setup()
    device = torch.device("cuda", local_rank)

    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    model = MLP().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)
        model.train()

        total_loss = 0.0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if dist.get_rank() == 0:
            print(f"epoch={epoch}, loss={total_loss / len(loader):.4f}")

    cleanup()


if __name__ == "__main__":
    train()
