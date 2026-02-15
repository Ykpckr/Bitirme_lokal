import numpy as np

class KalmanFilter:
    """
    Minimal Kalman Filter used in DeepSORT.
    State vector (x, y, w, h, vx, vy, vw, vh)
    """
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1 * self._std_weight_position * measurement[3],
            1 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            5 * self._std_weight_velocity * measurement[3],
            5 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        mean = np.dot(self._motion_mat, mean)
        covariance = np.dot(np.dot(self._motion_mat, covariance), self._motion_mat.T)
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        covariance += motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.dot(
            np.dot(self._update_mat, covariance), self._update_mat.T)

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3]
        ]
        innovation_cov = projected_cov + np.diag(np.square(std))

        kalman_gain = np.dot(
            np.dot(covariance, self._update_mat.T),
            np.linalg.inv(innovation_cov)
        )

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.dot(
            np.dot(kalman_gain, innovation_cov), kalman_gain.T
        )
        return new_mean, new_covariance
