# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]    # # 取四个坐标数组
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]  # w + x1
    y2 = boxes[:, 3] + boxes[:, 1]  # h + y1

    area = (x2 - x1 + 1) * (y2 - y1 + 1)    # 计算检测框的像素面积
    if scores is not None:          # 按得分排序（如没有置信度得分，可按坐标从小到大排序，如右下角坐标）
        idxs = np.argsort(scores)   # 得分从小到大，返回对应的索引
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:            # 开始遍历，并删除重复的框
        last = len(idxs) - 1        # 将得分最高的放入
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])    # 找剩下的其余框中最大坐标和最小坐标，即得到相交区域，左上及右下 
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]       # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）; 这里计算采用的是：重叠面积 / 面积1

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick
