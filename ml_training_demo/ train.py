import torch
from torch.utils.data import DataLoader
from dataset import RandomDataset
from model import SimpleModel

dataset = RandomDataset()
loader = DataLoader(dataset, batch_size=16)

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters())

for x, y in loader:

    pred = model(x)

    loss = (pred - y) ** 2
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

print("training finished")
