"""
In this file I am fixing the randomness so my training runs are reproducible.

Training a neural net uses randomness in two places: the initial weights and the order
batches are shuffled. By setting one seed for all the random sources, every run starts
from the same state and gives essentially the same result. This does NOT make the model
better, it just makes my numbers repeatable so I (and anyone cloning the repo) get the
same output. On the Mac GPU (mps) results are near-identical but may not be bit-for-bit
perfect, because some GPU operations run in a nondeterministic order.
"""
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
