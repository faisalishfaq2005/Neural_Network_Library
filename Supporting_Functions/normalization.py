import numpy as np
def normalize(data, min_val=None, max_val=None):
    """
    Normalizes the data to a range of [0, 1].
    If min_val and max_val are not provided, they are calculated from the data.
    """
    if min_val is None:
        min_val = np.min(data, axis=0)
    if max_val is None:
        max_val = np.max(data, axis=0)
    
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val

def denormalize(normalized_data, min_val, max_val):
    """
    Rescales the normalized data back to its original range.
    """
    return normalized_data * (max_val - min_val) + min_val