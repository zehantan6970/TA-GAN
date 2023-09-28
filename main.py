import time

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tracker import Tracker
import torch
from ta_gan.sgan.models_transformer_ori import Trajectory_Generator


def deal_laser(msg):
    #处理雷达数据
    distances = np.array(msg.ranges)
    angles = []
    angle_min = msg.angle_min
    angle_increment = msg.angle_increment
    #根据增量计算雷达角度数据
    for i in range(len(distances)):
        angles.append(angle_min)
        angle_min += angle_increment

    #滤除inf数据,距离小于阈值的数据
    #ind_to_remove = np.where(distances==np.inf)[0]
    ind_to_remove = np.where(distances > 2.5)[0]
    angles = [value for ind, value in enumerate(angles) if ind not in ind_to_remove]
    angles = np.array(angles)
    #distances = list(filter(lambda x: x != np.inf, distances))
    distances = list(filter(lambda x: x <= 2.5, distances))

    #将极坐标数据转为直角坐标数据
    x = (distances * np.cos(angles)).reshape(-1, 1)
    y = (distances * np.sin(angles)).reshape(-1, 1)

    points = np.concatenate([x, y], axis=1)

    return points, x, y



if __name__ == '__main__':
    rospy.init_node("get_laser")
    fig = plt.figure(figsize=(9, 9))
    plt.ion()
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 定义跟踪器，允许跳跃的最大间隔为5
    laser_object_tracker = Tracker(max_margin=10)
    # 定义并加载轨迹预测模型
    model = Trajectory_Generator(obs_len=20,
                                 embedding_dim=16,
                                 encoder_input_dim=16,
                                 encoder_output_dim=16,
                                 encoder_mlp_dim=16,
                                 encoder_num_head=2,
                                 drop_rate=0,
                                 rel_traj_dim=16,
                                 noise_dim=4,
                                 merge_mlp_dim=16)
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load("ta_gan/scripts/best_model_indoor.pt"))
    count = 0
    while not rospy.is_shutdown():
        # now = time.time()
        msg = rospy.wait_for_message("/scan", LaserScan, timeout=None)
        points, x, y = deal_laser(msg)
        plt.clf()
        if points.any():
            # dbscan聚类
            clustering = DBSCAN(eps=0.5, min_samples=4).fit(points)#0.3
            # 静态点滤除--后续可将激光数据转化至全局静态地图滤除静态点
            # labels = clustering.labels_
            # unique_labels, counts = np.unique(labels, return_counts=True)
            # unique_labels = unique_labels[counts <= 200]
            # x = x[np.isin(labels, unique_labels)]
            # y = y[np.isin(labels, unique_labels)]
            # color = labels[np.isin(labels, unique_labels)] + 1

            # 计算每个类的中心点
            labels = clustering.labels_
            unique_labels, counts = np.unique(labels, return_counts=True)
            if -1 in unique_labels:
                unique_labels = unique_labels[1:]
            cluster_centers = [] #聚类中心
            clusters = [] #聚类的点
            for label in unique_labels:
                cluster_points = points[labels == label]
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
                clusters.append(cluster_points)
            cluster_centers = np.array(cluster_centers)

            # 预测
            laser_object_tracker.predict()

            # 匹配+更新
            laser_object_tracker.update(clusters, cluster_centers)

            #轨迹预测
            pred_traj = laser_object_tracker.trajectory_predict(model)

            # then = time.time()
            # print(then - now)
            # 绘制散点图

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.scatter(x, y, c='r', marker='+')

            for t in laser_object_tracker.tracks:
                c_x, c_y = t.mean[0], t.mean[1]
                plt.scatter(c_x, c_y, s=60, c='k', marker='.')
                plt.annotate(str(t.id), xy=(c_x[0], c_y[0]), xytext=(c_x[0] + 0.1, c_y[0] + 0.1))

            if pred_traj is not None:
                pred_traj = pred_traj.reshape(-1, 2)
                plt.scatter(pred_traj[:, 0], pred_traj[:, 1], s=20, c='g', marker='.')


            #plt.savefig('./images/'+str(count)+'.jpg')
            count += 1

            plt.pause(0.001)
    plt.ioff()
    plt.show()