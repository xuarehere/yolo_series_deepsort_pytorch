# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter


INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)       # 计算代价矩阵；所有的轨迹，这一帧的检测所有检测box，抽出来的某一批轨迹序号（越小则丢失的次数越少），所有的检测 box 的序号
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5   # 限定最大值

    row_indices, col_indices = linear_assignment(cost_matrix)       # 匈牙利代价矩阵结果，最优解

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):     # 在当前检测帧结果中，找出未匹配的detections
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):             # 在已有的轨迹列表中，找出未匹配的轨迹 tracks
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]                          # 取出匹配上的历史轨迹
        detection_idx = detection_indices[col]                  # 取出匹配上的detection
        if cost_matrix[row, col] > max_distance:                # 如果相应的cost大于阈值max_distance，也视为未匹配成功
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

# 级联匹配源码
def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.
    
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:           # 大多数时候，默认是 detection_indices 是 None
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    test_unmatched_detections = []
    for level in range(cascade_depth):  # 由小到大依次对每个level的tracks做匹配; 指的是目标丢失的次数的从小到大进行匹配。越小，说明目标一直都在，丢失的次数比较少；越大，说明目标丢失的次数越多
        if len(unmatched_detections) == 0:  # No detections left    # 如果没有detections，退出循环
            break

        track_indices_l = [     
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]                              # 当前level的所有tracks索引     #连续匹配失败次数与级联匹配深度对应的轨迹集合。逐"批"取出来轨迹信息，从丢失次数少的开始取出来，取到最大
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue
        elif len(track_indices_l) >1:
            # print("len(track_indices_l)-->".format(len(track_indices_l)))
            pass
        # 基于外观信息的余弦距离代价矩阵 ==> cost_matrix  ==> 基于马氏距离代价矩阵  ==>  cost_matrix
        # 匈牙利匹配。代价矩阵：    行：轨迹id信息，列是box 信息。 cos_matrix[i, j] 某个轨迹id与某个 box 之间的cosine距离的最小值
        # 每一次，把上次没有匹配上的检测目标 unmatched_detections 与历史轨迹信息组别 i ，进行匹配
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)  # 匈牙利匹配; 把每一组轨迹数据取出来（轨迹组别顺序是，一直在，丢失一次，丢失n次），跟所有的检测框 box 进行轨迹匹配
        matches += matches_l
        test_unmatched_detections += unmatched_detections
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    # 根据通过卡尔曼滤波获得的状态分布，使成本矩阵中的不可行条目无效。
    gating_dim = 2 if only_position else 4      # 根据是否只考虑位置分量选择维度
    gating_threshold = kalman_filter.chi2inv95[gating_dim]  # 9.4877  #根据维度(自由度)选择门控距离阈值
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])   #测量值矩阵
    #计算轨迹状态分布和测量分布之间的门控距离对外观语义特征相似度矩阵进行限制 
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)      #计算门控距离       mean: 是预测的还是观测的？预测，卡尔曼滤波估计值        *********计算马氏距离的函数*********
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost   #门控距离大于阈值则被设置为无效值
    return cost_matrix
