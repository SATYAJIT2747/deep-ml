import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    # Your code here
    if position == 0 or d_model <= 0:
        return -1

    pe = torch.zeros(position, d_model, dtype=torch.float16)

    for pos in range(position):
        for i in range(0, d_model, 2):

            angle = pos / (10000 ** (i / d_model))

            pe[pos, i] = torch.sin(torch.tensor(angle))

            if i + 1 < d_model:
                pe[pos, i + 1] = torch.cos(torch.tensor(angle))

    return pe