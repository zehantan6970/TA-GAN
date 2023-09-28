

class Track(object):
    def __init__(self, mean, covariance, id, points, max_margin):
        self.mean = mean
        self.covariance = covariance
        self.id = id
        self.max_margin = max_margin
        self.track_num = 0 #该跟踪器一共跟踪了几帧
        self.time_since_update = 0 #距离上次更新经过的帧数
        self.is_confirm = False  #该轨迹是否被成功跟踪过
        self.is_delete = False #该轨迹是否要被删除
        # self.predictable = False #获取连续20个历史轨迹点时为True
        self.history_points = []


    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    def update(self, kf, measurement):
        # 状态参数更新
        self.time_since_update = 0 #更新 距离上次更新经过的帧数
        self.track_num += 1 #更新 跟踪的总帧数
        if self.track_num > 2: #连续跟踪3次以上被认证
            self.is_confirm = True

        # 轨迹更新
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)

        #添加历史轨迹点
        self.history_points.append(self.mean[:2].reshape(-1, 2))
        if len(self.history_points) > 20:
            del self.history_points[0]
            #print(self.history_points)

    def edit_state(self):
        # 对于未匹配上的轨迹修改其状态
        if not self.is_confirm:
            self.is_delete = True
        else:
            self.time_since_update += 1
            if self.time_since_update > self.max_margin:
                self.is_delete = True
            else:
                #添加历史轨迹点
                self.history_points.append(self.mean[:2].reshape(-1, 2))
                if len(self.history_points) > 20:
                    del self.history_points[0]