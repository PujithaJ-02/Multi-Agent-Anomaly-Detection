"""
In this file I am building the Transformer model.

The job of this model is simple: I give it a window of 64 readings and it tries to
predict the next reading. Later, when its prediction is far off from what actually
happened, that gap is what tells me something anomalous is going on.
"""
import torch
import torch.nn as nn

class AnomalyTransformer(nn.Module):
    def __init__(self, window=64, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        # Each reading is just one number. The Transformer works with vectors, not
        # single numbers, so here I am turning each reading into a vector of size
        # d_model so the model has room to represent it richly.
        self.input_proj = nn.Linear(1, d_model)

        # A Transformer has no built-in sense of order, so I add a learned position
        # signal that tells it which reading is 1st, 2nd, 3rd and so on.
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))

        # This is the actual Transformer brain: it looks at all 64 readings at once
        # and learns how they relate to each other.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Finally I squeeze everything back down to a single number: my prediction
        # for the next reading.
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x comes in as (batch, 64). I add a last dimension so each reading is
        # treated as its own little vector: (batch, 64, 1).
        x = x.unsqueeze(-1)
        x = self.input_proj(x)      # (batch, 64, d_model)
        x = x + self.pos            # add the position signal
        x = self.encoder(x)         # let the Transformer look across the window
        x = x.mean(dim=1)           # blend the 64 positions into one summary
        return self.head(x).squeeze(-1)   # one predicted number per window
