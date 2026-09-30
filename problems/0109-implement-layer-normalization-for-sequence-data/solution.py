import torch
import torch.nn as nn
def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    # Your code here
    # mean = X.mean(dim = -1 , keepdim = True)
    # var = ((X-mean)**2).mean(dim = -1 , keepdim = True)
    # X_hat = (X- mean)/torch.sqrt(var + epsilon)
    # y_hat = gamma * X_hat + beta
    # return  y_hat
    _, _, d_model = X.shape

    layer_norm = nn.LayerNorm(
        d_model,
        eps=epsilon,
        elementwise_affine=False
    )

    return layer_norm(X) * gamma + beta