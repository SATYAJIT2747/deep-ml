import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return (Q , K , V)

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / d_k ** 0.5 # For each query, how much attention should it give to each key?
    return torch.softmax(scores, dim= -1) @ V # dim = -1 i.e Apply softmax across each row.

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len , d_mod = Q.shape
    d_head = d_mod//n_heads # use // to get 
    assert d_mod % n_heads == 0
    #reshape and divide q k v among all heads(seq_len , n_heads , n_features per head)
    Q_head = Q.reshape(seq_len , n_heads , d_head)
    K_head = K.reshape(seq_len , n_heads , d_head)
    V_head = V.reshape(seq_len , n_heads , d_head)
    # per head self self_attention
    outputs = []
    for h in range(n_heads):
        head_output = self_attention(
            Q_head[:, h, :],
            K_head[:, h, :],
            V_head[:, h, :]
        )
        outputs.append(head_output)

    output = torch.cat(outputs , dim = -1)
    return output

    