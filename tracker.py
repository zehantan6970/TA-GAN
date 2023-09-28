import kalman_filter
import track
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import torch


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


class Tracker(object):
    def __init__(self, max_margin):
        self.tracks = []
        self.next_id = 1
        self.kf = kalman_filter.KalmanFilter()
        self.max_margin = max_margin #跟踪允许的最大间隔
        self.match_threshold_1 = 0.7 #匹配的距离阈值
        self.match_threshold_2 = 1.2

    def predict(self):
        for t in self.tracks:
            t.predict(self.kf)

    #根据聚类中心的马氏距离和点的数量特征进行匈牙利算法匹配
    def match(self, clusters, cluster_centers):
        """
        :param clusters: 聚类的点
        :param cluster_centers: 每个聚类的中心， 即measurements
        :return:
        """
        def maha_distance(mean, covariance, measurements, only_position=True):
            """
            计算马氏距离
            :param mean: 卡尔曼滤波器预测的均值
            :param covariance: 卡尔曼滤波器预测的协方差矩阵
            :param measurements: 当前的所有检测结果
            :param only_position: 是否只计算点坐标间的马氏距离
            :return: track和detections间的马氏距离
            """
            mean, covariance = self.kf.project(mean, covariance) #从预测分布向观测分布转换
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
            cholesky_factor = np.linalg.cholesky(covariance)
            d = measurements - mean.reshape(1, -1)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z*z, axis=0)
            return squared_maha

        '''def cal_cost_matrix(track_inds, measurements):
            cost_matrix = []
            for ind in track_inds:
                cost_matrix_row = maha_distance(self.tracks[ind].mean, self.tracks[ind].covariance, measurements)
                cost_matrix.append(cost_matrix_row)
            cost_matrix = np.array(cost_matrix).reshape(len(track_inds), -1)
            return cost_matrix'''

        def hungarian_match(track_inds, measurements):
            if not track_inds or not measurements.any():
                return [], [], []
            else:
                cost_matrix = []
                for ind in track_inds:
                    cost_matrix_row = maha_distance(self.tracks[ind].mean, self.tracks[ind].covariance, measurements)
                    cost_matrix.append(cost_matrix_row)
                cost_matrix = np.array(cost_matrix).reshape(len(track_inds), -1)
                track_row, measurement_col = linear_sum_assignment(cost_matrix)
                return cost_matrix, track_row, measurement_col


        matched_tracks, matched_measurement, unmatched_tracks, unmatched_measurements = [], [], [], []
        if not self.tracks:
            unmatched_measurements = [i for i in range(len(cluster_centers))]
            return matched_tracks, matched_measurement, unmatched_tracks, unmatched_measurements
        else:
            # 被认证的轨迹，即此前被成功跟踪过的轨迹id
            confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirm]
            # 未认证的轨迹id
            # unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirm]

            # 先匹配认证的轨迹，再匹配未认证的轨迹
            if confirmed_tracks:
                # 匹配认证的轨迹
                '''cost_matrix_confirmed = cal_cost_matrix(confirmed_tracks, cluster_centers)
                if cost_matrix_confirmed.any():
                    track_row, measurement_col = linear_sum_assignment(cost_matrix_confirmed)
                else:
                    track_row, measurement_col = [], []'''
                cost_matrix_confirmed, track_row, measurement_col = hungarian_match(confirmed_tracks, cluster_centers)

                # 删除超过代价阈值的匹配
                for row, col in zip(track_row, measurement_col):
                    if cost_matrix_confirmed[row, col] < self.match_threshold_1:
                        matched_tracks.append(confirmed_tracks[row])
                        matched_measurement.append(col)

            unmatched_tracks = [i for i, _ in enumerate(self.tracks) if i not in matched_tracks] #未认证的轨迹和匹配不成功的轨迹
            unmatched_measurements = [i for i in range(len(cluster_centers)) if i not in matched_measurement]

            # 匹配 未认证的轨迹和匹配不成功的轨迹
            '''cost_matrix_unconfirmed = cal_cost_matrix(unmatched_tracks,
                                                      cluster_centers[unmatched_measurements])
            if cost_matrix_unconfirmed.any():
                track_row, measurement_col = linear_sum_assignment(cost_matrix_unconfirmed)
            else:
                track_row, measurement_col = [], []'''
            cost_matrix_unconfirmed, track_row, measurement_col = hungarian_match(unmatched_tracks,
                                                                                cluster_centers[unmatched_measurements])
            # 删除超过代价阈值的匹配
            for row, col in zip(track_row, measurement_col):
                if cost_matrix_unconfirmed[row, col] < self.match_threshold_2:
                    matched_tracks.append(unmatched_tracks[row])
                    matched_measurement.append(unmatched_measurements[col])

            # 没有匹配上的轨迹和检测
            unmatched_tracks = [i for i, _ in enumerate(self.tracks) if i not in matched_tracks]
            unmatched_measurements = [i for i in range(len(cluster_centers)) if i not in matched_measurement]

            return matched_tracks, matched_measurement, unmatched_tracks, unmatched_measurements

    def update(self, clusters, cluster_centers):
        matched_tracks, matched_measurement, unmatched_tracks, unmatched_measurements = self.match(clusters,
                                                                                                   cluster_centers)
        # 对跟踪成功的轨迹进行更新
        for track_id, measure_id in zip(matched_tracks, matched_measurement):
            self.tracks[track_id].update(self.kf, cluster_centers[measure_id])

        # 处理未匹配上的轨迹，修改其状态
        for track_id in unmatched_tracks:
            self.tracks[track_id].edit_state()

        # 根据状态删除轨迹
        self.tracks = [t for t in self.tracks if not t.is_delete]

        # 将未匹配的检测添加为新的track
        for cluster_id in unmatched_measurements:
            self.initate_track(clusters[cluster_id], cluster_centers[cluster_id])



    def initate_track(self, cluster_points, cluster_center):
        '''
        :param cluster_points: 某一聚类的所有点的坐标
        :param cluster_center: 某一聚类的中心坐标
        :return:
        '''
        # 初始化均值
        mean_pos = np.array(cluster_center)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel].reshape(-1, 1)
        # 初始化协方差
        std = [0.4, 0.4, 0.1, 0.1]
        covariance = np.diag(std)

        self.tracks.append(track.Track(mean, covariance, self.next_id,
                                       cluster_points, self.max_margin))
        self.next_id += 1

    def trajectory_predict(self, model):
        history_trajectory = []
        for t in self.tracks:
            if len(t.history_points) == 20:
                history_trajectory.append(np.array(t.history_points).reshape(20, 2))
        if history_trajectory:
            obs_traj = np.array(history_trajectory).transpose((1, 0, 2))
            obs_traj_rel = np.zeros_like(obs_traj)
            obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]
            #print(obs_traj_rel)
            obs_traj = torch.from_numpy(obs_traj).type(torch.float).cuda()
            obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float).cuda()
            seq_start_end = torch.tensor([[0, obs_traj.size(1)]]).cuda()
            #print(seq_start_end)
            pred_traj = model(obs_traj, obs_traj_rel, seq_start_end)
            pred_traj = relative_to_abs(pred_traj, obs_traj[-1])
            pred_traj = pred_traj.cpu().detach().numpy()
            return pred_traj
        else:
            return None
