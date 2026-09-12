import numpy as np
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    a = np.array((x))
    a_min = a.min()
    a_max = a.max()
    if(a_max == a_min):
         return -1
    return (a-a_min)/(a_max-a_min)