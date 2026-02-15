import numpy as np
from scipy.spatial.distance import cdist

def compute_cosine_distance(A, B):
    """Cosine distance between two sets of vectors."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    # 1 - cosine_similarity = cosine_distance
    return cdist(A, B, metric="cosine")