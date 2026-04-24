import numpy as np


def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    PE = np.zeros((seq_length,d_model))
    for i in range(seq_length):
        for j in range(d_model):
            angle = i / np.power(10000, (2 * (j // 2)) / d_model)
            if(j % 2 == 0):
                PE[i,j] = np.sin(angle)
            else:
                PE[i,j] = np.cos(angle)

    return PE