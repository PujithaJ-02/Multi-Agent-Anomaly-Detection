"""
In this file I am training the Transformer.

I load the windowed data I made earlier, then I run the training loop: the model
guesses the next reading, I measure how wrong it is, and it slowly corrects itself.
I only train on the training set. The test set stays hidden until I measure later.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformer import AnomalyTransformer

# On Apple Silicon I can use the Mac GPU (mps) for a speedup. If it is not there
# I just fall back to the normal cpu, which is still fast on this small data.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("using device:", device)

# Loading the windows I saved in the preprocess step.
X_train = np.load("../../data/processed/X_train.npy")
y_train = np.load("../../data/processed/y_train.npy")

X = torch.tensor(X_train)
y = torch.tensor(y_train)

# DataLoader feeds the data to the model in small batches instead of all at once.
loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

model = AnomalyTransformer().to(device)

# The optimizer is what actually nudges the model to improve each step.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# The loss measures how far my prediction is from the real next reading.
loss_fn = nn.MSELoss()

EPOCHS = 15
for epoch in range(EPOCHS):
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()          # clear last step's leftovers
        pred = model(xb)               # model guesses the next readings
        loss = loss_fn(pred, yb)       # how wrong was it
        loss.backward()                # figure out how to improve
        optimizer.step()               # take the improvement step
        total += loss.item()
    print(f"epoch {epoch+1:2d}/{EPOCHS}  loss {total/len(loader):.6f}")

# Saving the trained weights so the next steps can load and use the model.
torch.save(model.state_dict(), "../../models/transformer.pt")
print("saved model to models/transformer.pt")
