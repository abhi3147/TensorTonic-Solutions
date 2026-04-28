import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    S = Q @ K.transpose(-2, -1)
    d_k = K.shape[-1]
    S_scaled = S/math.sqrt(d_k)
    W = torch.softmax(S_scaled,dim = -1)
    O = W @ V

    return O