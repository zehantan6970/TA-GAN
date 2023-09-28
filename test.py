import time

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN



def deal_laser(msg):
    #处理雷达数据
    distances = np.array(msg.ranges)
    angles = []
    angle_min = msg.angle_min #+ 3.14
    angle_increment = msg.angle_increment
    #根据增量计算雷达角度数据
    for i in range(len(distances)):
        angles.append(angle_min)
        angle_min += angle_increment

    #滤除inf数据,距离小于阈值的数据
    #ind_to_remove = np.where(distances==np.inf)[0]
    ind_to_remove = np.where(distances > 10)[0]
    angles = [value for ind, value in enumerate(angles) if ind not in ind_to_remove]
    angles = np.array(angles)
    #distances = list(filter(lambda x: x != np.inf, distances))
    distances = list(filter(lambda x: x <= 10, distances))

    #将极坐标数据转为直角坐标数据
    #global points, x, y
    x = (distances * np.cos(angles)).reshape(-1, 1)
    y = (distances * np.sin(angles)).reshape(-1, 1)

    points = np.concatenate([x, y], axis=1)
    print(points.shape[0])

    return points, x, y





if __name__ == '__main__':
    rospy.init_node("get_laser")
    fig = plt.figure(figsize=(9, 9))
    plt.ion()
    # count = 0
    while not rospy.is_shutdown():
        then = time.time()
        msg = rospy.wait_for_message("/scan", LaserScan, timeout=None)
        now = time.time()
        print(now - then)
        points, x, y = deal_laser(msg)

        # dbscan聚类
        clustering = DBSCAN(eps=0.4, min_samples=5).fit(points)
        #静态点滤除--后续可将激光数据转化至全局静态地图滤除静态点
        # labels = clustering.labels_
        # unique_labels, counts = np.unique(labels, return_counts=True)
        # unique_labels = unique_labels[counts <= 200]
        # x = x[np.isin(labels, unique_labels)]
        # y = y[np.isin(labels, unique_labels)]
        # color = labels[np.isin(labels, unique_labels)] + 1

        #计算每个类的中心点
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        if -1 in unique_labels:
            unique_labels = unique_labels[1:]
        cluster_centers = []
        for label in unique_labels:
            cluster_points = points[labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        cluster_centers = np.array(cluster_centers)

        # 绘制散点图
        color = clustering.labels_ + 1
        plt.clf()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        plt.scatter(x, y, s=color * 2, c=color, marker='.', cmap=plt.cm.Spectral)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=20, c='k')

        # plt.scatter(x, y, c='r', marker='.')
        # if count == 20:
        #     plt.savefig('result.jpg')
        # count += 1
        plt.pause(0.01)
    plt.ioff()
    plt.show()

