import torch

class RandomDataset:

    def __init__(self, size=100):
        self.x = torch.randn(size, 10)
        self.y = torch.sum(self.x, dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
