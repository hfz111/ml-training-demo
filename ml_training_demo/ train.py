import torch
from torch.utils.data import DataLoader
from ml_training_demo.dataset import RandomDataset
from ml_training_demo.model import SimpleModel

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
