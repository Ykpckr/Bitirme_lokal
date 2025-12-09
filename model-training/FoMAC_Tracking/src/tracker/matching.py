import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def iou_batch(bboxes1, bboxes2):
    """
    Vektörel IoU (Intersection over Union) hesaplaması.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return o

def embedding_distance(tracks, detections, metric='cosine'):
    """
    Track features ile detection features arasındaki mesafeyi ölçer.
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([d[4] for d in detections], dtype=np.float32) # d[4] feature

    for i, track in enumerate(tracks):
        # Track'in son özelliklerini al (Moving average veya gallery)
        # Buradaki basit yaklaşım: Son eklenen özelliği kullan
        # Gelişmiş yaklaşım: Track içindeki tüm özelliklerin ortalaması veya en yakını
        track_feat = track.features[-1].reshape(1, -1)
        
        # Cdist cosine distance döndürür
        dist = cdist(track_feat, det_features, metric)
        cost_matrix[i, :] = dist.min(axis=0) # En yakın mesafeyi al
        
    return cost_matrix

def linear_assignment(cost_matrix, threshold):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    x, y = linear_sum_assignment(cost_matrix)
    matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= threshold])

    if len(matches) == 0:
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_detections = list(range(cost_matrix.shape[1]))
    else:
        unmatched_tracks = list(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
        unmatched_detections = list(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_tracks, unmatched_detections