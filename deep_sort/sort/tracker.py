# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init    # 只有连续3帧都能够匹配上才会变为Confirmed,才会有输出结果，默认值 3 

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []        # 跟踪列表
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade. 
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)          # 级联特征匹配, 得到匹配对、未匹配的tracks、未匹配的dectections; 里面包含 iou match

        # Update track set.
        for track_idx, detection_idx in matches:         # ====卡尔曼滤波====， 对于matches每个匹配成功的track，用其对应的detection进行更新
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])     # 匹配上的，只保留一个最新的reid 特征？
        for track_idx in unmatched_tracks:               # 对于unmatched_tracks未匹配的成功的track，将其标记为丢失
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:       # 对于unmatched_detections未匹配成功的 detection，初始化为新的 track
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]        # 筛选出没有被删除的目标

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]      # 筛选 is_confirmed 目标
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():        # 如果track 没有确认，跳过
                continue
            features += track.features      # 列表拼接，将tracks列表拼接到features列表
            targets += [track.track_id for _ in track.features] # ？
            track.features = []              # #清空confirmed track的特征值？
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)  # 距离度量中的特征集更新， 更新确认态轨迹(同一ID)以往所有时刻的外观语义特征字典

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):   # 基于外观信息和马氏距离，计算卡尔曼滤波预测的tracks和当前时刻检测到的detections的代价矩阵；功能： 用于计算track和detection之间的距离，代价函数需要使用在 KM 算法之前
            features = np.array([dets[i].feature for i in detection_indices])       # 外观信息，reid 模型提取的
            targets = np.array([tracks[i].track_id for i in track_indices])         # 跟踪id
            # 基于外观信息，计算tracks和detections的余弦距离代价矩阵
            cost_matrix = self.metric.distance(features, targets)                   # 这里得到的代价矩阵就是每个track对象和现在det对象的代价值；    # 1. 通过最近邻计算出代价矩阵 cosine distance
            # 基于马氏距离，过滤掉代价矩阵中一些不合适的项 (将其设置为一个较大的值)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)      # 判断距离关系，使用马氏距离，大于阈值的都把代价变成无穷。                                                  # 2. 计算马氏距离,得到新的状态矩阵

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.        # 获取已跟踪与未跟踪的内容
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # 1
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)      # 对于 confirmed_tracks 确定性的跟踪器结果，使用外观特征进行和检测器结果匹配
        # 2
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]      # 所有未匹配上的目标 + 丢失一次没匹配上的目标
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]     # 已经很久没有匹配上
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
