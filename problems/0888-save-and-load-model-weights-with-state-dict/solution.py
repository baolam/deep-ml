import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    buff = io.BytesIO()

    torch.save(src.state_dict(), buff)
    buff.seek(0)
    state = torch.load(buff)
    dst.load_state_dict(state)
    return dst
