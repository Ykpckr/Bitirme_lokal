# tracking/kalman.py
import numpy as np

class KalmanFilter:
    """
    Simplified Kalman filter for bounding boxes (cx, cy, w, h) with simple constant velocity model.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    """
    def __init__(self):
        dt = 1.0
        # State transition
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = dt
        # Process noise
        q = 1e-2
        self.Q = np.eye(8) * q
        # Measurement matrix: we measure cx,cy,w,h
        self.H = np.zeros((4,8))
        self.H[0,0] = 1
        self.H[1,1] = 1
        self.H[2,2] = 1
        self.H[3,3] = 1
        # Measurement noise
        r = 1e-1
        self.R = np.eye(4) * r

    def initiate(self, bbox):
        # bbox: [x1,y1,x2,y2]
        x1,y1,x2,y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        state = np.array([cx, cy, w, h, 0,0,0,0], dtype=float)
        P = np.eye(8) * 1.0
        return state, P

    def predict(self, mean, cov):
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + self.Q
        return mean, cov

    def update(self, mean, cov, measurement):
        # measurement: [cx,cy,w,h]
        z = np.array(measurement)
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ mean)
        mean = mean + (K @ y)
        cov = (np.eye(len(mean)) - K @ self.H) @ cov
        return mean, cov
