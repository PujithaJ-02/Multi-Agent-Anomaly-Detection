"""
In this file I am just checking that the model runs and gives the right output shape.
I am feeding it 8 fake windows of 64 readings and expecting 8 predictions back.
"""
import torch
from transformer import AnomalyTransformer

model = AnomalyTransformer()
fake = torch.randn(8, 64)          # 8 fake windows
out = model(fake)
print("input shape:", fake.shape)
print("output shape:", out.shape)  # should be (8,) = one prediction per window
