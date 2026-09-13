import math
import numpy as np
def sigmoid(z: float) -> float:
	result = 1 / (1 + np.exp(-z))
	return result