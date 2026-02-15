# tracking/matching.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm

def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

def cosine_distance(a, b):
    # a: (D,), b: (D,)
    if a is None or b is None:
        return 1.0
    num = np.dot(a, b)
    den = (norm(a) * norm(b) + 1e-8)
    return 1.0 - (num / den)

def compute_cost_matrix(tracks, detections, embeds=None, iou_weight=0.5, appearance_weight=0.5):
    """
    tracks: list of TrackState
    detections: list of bbox [x1,y1,x2,y2]
    embeds: list of embedding vectors for detections (or None)
    """
    N = len(tracks)
    M = len(detections)
    if N == 0 or M == 0:
        return np.zeros((N, M))
    iou_mat = np.zeros((N, M))
    app_mat = np.zeros((N, M))
    for i, tr in enumerate(tracks):
        for j, det in enumerate(detections):
            iou_mat[i,j] = iou_xyxy(tr.bbox, det)
            if tr.embed is not None and embeds is not None and j < len(embeds):
                app_mat[i,j] = cosine_distance(tr.embed, embeds[j])
            else:
                app_mat[i,j] = 1.0
    # Convert IoU to cost
    iou_cost = 1.0 - iou_mat
    app_cost = app_mat
    cost = iou_weight * iou_cost + appearance_weight * app_cost
    return cost

def linear_assignment(cost_matrix, thresh=0.7):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[1])), list(range(cost_matrix.shape[0]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_dets, unmatched_tracks = [], [], []
    matched_rows = set()
    matched_cols = set()
    for r,c in zip(row_ind, col_ind):
        if cost_matrix[r,c] > thresh:
            unmatched_tracks.append(r)
            unmatched_dets.append(c)
        else:
            matches.append((r,c))
            matched_rows.add(r)
            matched_cols.add(c)
    for r in range(cost_matrix.shape[0]):
        if r not in matched_rows:
            unmatched_tracks.append(r)
    for c in range(cost_matrix.shape[1]):
        if c not in matched_cols:
            unmatched_dets.append(c)
    return matches, unmatched_dets, unmatched_tracks
