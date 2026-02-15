import numpy as np

def nms(spots, window=30):
    spots = sorted(spots, key=lambda x: x[1], reverse=True)
    keep = []

    for t, score, cls in spots:
        if all(abs(t - kt) > window for kt, _, _ in keep):
            keep.append((t, score, cls))
    return keep
