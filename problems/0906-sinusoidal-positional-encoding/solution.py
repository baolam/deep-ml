import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)

    pos = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=float) * (-math.log(10000.0 / d_model))
    )

    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)

    return pe
