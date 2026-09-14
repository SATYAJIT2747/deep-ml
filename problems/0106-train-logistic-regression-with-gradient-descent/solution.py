import numpy as np

def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    iterations: int
) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression,
    optimizing parameters with Binary Cross Entropy loss.
    """

    X = np.column_stack((np.ones(X.shape[0]), X))

    w = np.zeros(X.shape[1])

    losses = []

    for _ in range(iterations):
        z = X @ w

        p = 1 / (1 + np.exp(-z))

        p = np.clip(p, 1e-15, 1 - 1e-15)

        loss = -np.sum(
            y * np.log(p) +
            (1 - y) * np.log(1 - p)
        )

        losses.append(round(loss, 4))

        dw = X.T @ (p - y)

        w -= learning_rate * dw

    return w, losses