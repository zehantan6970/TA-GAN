import numpy as np


class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 2, 1 #ndim--考虑的状态数，dt时间间隔
        self.motion_mat = np.eye(2*ndim, 2*ndim) #运动方程F
        for i in range(ndim):
            self.motion_mat[i, ndim+i] = dt
        self.update_mat = np.eye(2 * ndim, 2 * ndim) #观测矩阵H
        self.Q = np.diag(np.array([0.05, 0.05, 0.01, 0.01])) #协方差噪声矩阵
        self.R = np.diag(np.array([0.05, 0.05, 0.01, 0.01])) #观测噪声矩阵

    def predict(self, mean, covariance):
        """
        :param mean: 上一时刻均值
        :param covariance: 上一时刻协方差
        :return:
        """
        mean = np.dot(self.motion_mat, mean)
        covariance = np.linalg.multi_dot((self.motion_mat, covariance, self.motion_mat.T)) + self.Q
        return mean, covariance

    def project(self, mean, covariance):
        """
        :param mean: 当前时刻预测值
        :param covariance: 当前时刻预测的协方差
        :return:
        """
        mean = np.dot(self.update_mat, mean)
        covariance = np.linalg.multi_dot((self.update_mat, covariance, self.update_mat.T)) + self.R
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        :param mean: 当前时刻预测值
        :param covariance: 当前时刻预测的协方差
        :param measurement: 当前时刻观测值
        :return:
        """
        measurement = np.pad(measurement.reshape(-1, 1), ((0, 2), (0, 0)))
        project_mean, project_covariance = self.project(mean, covariance)
        y = measurement - project_mean
        K = np.linalg.multi_dot((covariance, self.update_mat.T, np.linalg.inv(project_covariance)))#卡尔曼增益
        mean = mean + np.dot(K, y)
        covariance = np.dot(np.eye(4) - np.dot(K, self.update_mat), covariance)

        return mean, covariance

