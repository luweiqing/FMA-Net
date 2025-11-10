# Copyright (c) OpenMMLab. All rights reserved.
import lap
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps
from .base_tracker import BaseTracker
from lib.utils.kalman_filter import KalmanFilter

import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.transform import resize

def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det


def bbox_xyxy_to_cxcyah(bboxes):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, ratio, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx = (bboxes[:, 2] + bboxes[:, 0]) / 2
    cy = (bboxes[:, 3] + bboxes[:, 1]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    xyah = torch.stack([cx, cy, w / h, h], -1)
    return xyah

def bbox_cxcyah_to_xyxy(bboxes):
    """Convert bbox coordinates from (cx, cy, ratio, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, ratio, h = bboxes.split((1, 1, 1, 1), dim=-1)
    w = ratio * h
    x1y1x2y2 = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    return torch.cat(x1y1x2y2, dim=-1)

class ByteTracker(BaseTracker):
    """Tracker for ByteTrack.
    Args:
        obj_score_thrs (dict): Detection score threshold for matching objects.
            - high (float): Threshold of the first matching. Defaults to 0.6.
            - low (float): Threshold of the second matching. Defaults to 0.1.
        init_track_thr (float): Detection score threshold for initializing a
            new tracklet. Defaults to 0.7.
        weight_iou_with_det_scores (bool): Whether using detection scores to
            weight IOU which is used for matching. Defaults to True.
        match_iou_thrs (dict): IOU distance threshold for matching between two
            frames.
            - high (float): Threshold of the first matching. Defaults to 0.1.
            - low (float): Threshold of the second matching. Defaults to 0.5.
            - tentative (float): Threshold of the matching for tentative
                tracklets. Defaults to 0.3.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 obj_score_thrs=dict(high=0.6, low=0.1),
                 init_track_thr=0.7,
                 weight_iou_with_det_scores=False,
                 match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
                 num_tentatives=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thrs = obj_score_thrs
        self.init_track_thr = init_track_thr

        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thrs = match_iou_thrs

        self.num_tentatives = num_tentatives

    @property
    def confirmed_ids(self):
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    @property
    def unconfirmed_ids(self):
        """Unconfirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if track.tentative]
        return ids

    def init_track(self, id, obj):
        """Initialize a track."""
        super().init_track(id, obj)
        self.tracks[id].last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.tracks[id].observations = dict()
        self.tracks[id].history_observations = []
        self.tracks[id].velocity = None
        self.tracks[id].delta_t = 3
        self.tracks[id].age = 0

        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True

        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id, obj):
        """Update a track."""
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        track_label = self.tracks[id]['labels'][-1]
        label_idx = self.memo_items.index('labels')
        obj_label = obj[label_idx]
        assert obj_label == track_label
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

        # update velocity
        if bbox is not None:
            if self.tracks[id].last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.tracks[id].delta_t):
                    dt = self.tracks[id].delta_t - i
                    if self.tracks[id].age - dt in self.tracks[id].observations:
                        previous_box = self.tracks[id].observations[self.tracks[id].age - dt]
                        break
                if previous_box is None:
                    previous_box = self.tracks[id].last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.tracks[id].velocity = speed_direction(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.tracks[id].last_observation = bbox
            self.tracks[id].observations[self.tracks[id].age] = bbox
            self.tracks[id].history_observations.append(bbox)



    def pop_invalid_tracks(self, frame_id):
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            # case1: disappeared frames >= self.num_frames_retrain
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            # case2: tentative tracks but not matched in this frame
            case2 = v.tentative and v['frame_ids'][-1] != frame_id

            # case3 = v["hist_sim"] > 0.3 and frame_id > 5

            if case1 or case2:  # or case3:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def adjust_dists(self, dists):
        rows, cols = dists.shape
        for i in range(rows):
            overlaps_idxs = []
            corresponding_thermal_signature = []
            for j in range(cols):
                if dists[i, j] < 1 and dists[i, j] >= 0:
                    overlaps_idxs.append(j)
                    corresponding_thermal_signature.append(dists[i, j])

            if len(overlaps_idxs) > 1:
                # for p in range(len(overlaps_idxs)):
                #    print("Row", i, "column ", overlaps_idxs[p], "value ", dists_iou[i,overlaps_idxs[p]],"thermal",corresponding_thermal_signature[p])

                min_thermal_sig_idx = corresponding_thermal_signature.index(min(corresponding_thermal_signature))

                for p in range(len(overlaps_idxs)):
                    if p != min_thermal_sig_idx:
                        dists[i, overlaps_idxs[p]] = 1

                v = 1
        return dists

    def assign_ids(self,
                   ids,
                   det_bboxes,
                   det_labels,
                   weight_iou_with_det_scores=False,
                   match_iou_thr=0.5,
                   apply_thermal_iou_adjustment=False,
                   velocities=None,
                   previous_obs=None,
                   vdc_weight=1.0):

        # 1. 获取当前 tracker 的边界框（转换格式）
        track_bboxes = np.zeros((0, 4))
        for id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        # 2. 计算 IOU 成本
        ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4], mode='iou')
        if apply_thermal_iou_adjustment:
            ious_np = ious.cpu().numpy()
            ious_np = self.adjust_dists(ious_np)
            ious = torch.from_numpy(ious_np).to(det_bboxes)
        if weight_iou_with_det_scores:
            ious *= det_bboxes[:, 4][None]
        dists_iou = 1 - ious.cpu().numpy()

        # 3. 如果提供了速度信息和上一帧观测，则计算速度方向一致性成本
        if velocities.shape[0] != 0  and previous_obs.shape[0] != 0:
            # 将检测边界框转换为 numpy 数组，形状 (num_dets, 4)

            dets = det_bboxes[:, :4]
            previous = previous_obs[:, :4]

            # 这里 previous_obs 应为 numpy 数组，形状 (num_tracks, 4)
            # 利用 speed_direction_batch 计算每个 tracker 与每个检测的归一化方向向量，返回两个矩阵 (num_tracks, num_dets)
            speed_dy, speed_dx = speed_direction_batch(dets, previous)

            num_tracks = len(ids)
            num_dets = dets.shape[0]
            # 将每个 tracker 的速度扩展成 (num_tracks, num_dets)
            inertia_dy = np.repeat(velocities[:, 0][:, np.newaxis], num_dets, axis=1)
            inertia_dx = np.repeat(velocities[:, 1][:, np.newaxis], num_dets, axis=1)

            # 计算 tracker 运动方向与检测方向之间的余弦相似度
            diff_angle_cos = inertia_dx * speed_dx.cpu().numpy() + inertia_dy * speed_dy.cpu().numpy()
            diff_angle_cos = np.clip(diff_angle_cos, -1, 1)
            diff_angle = np.arccos(diff_angle_cos)
            # 将角度差转换为成本得分：角度差越小（一致性越好），成本越低
            angle_cost = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

            # # 将检测得分纳入考虑：检测得分 shape 为 (num_dets,)
            # scores = det_bboxes[:, 4].cpu().numpy()
            # scores = np.repeat(scores[np.newaxis, :], num_tracks, axis=0)
            angle_cost = angle_cost * vdc_weight

            # 4. 融合成本：从 IOU 成本中减去角度一致性得分（使得运动方向一致的匹配具有更低的总成本）
            dists = dists_iou - angle_cost
        else:
            dists = dists_iou

        # 5. 二分图匹配：使用 LAPJV 算法求解匹配对
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col

    @force_fp32(apply_to=('bboxes'))
    def track(self,
              bboxes,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.

        Args:
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to False.
        Returns:
            tuple: Tracking results.
        """

        self.kf = KalmanFilter()

        if self.empty or bboxes.size(0) == 0:
            valid_inds = bboxes[:, -1] > self.init_track_thr
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks

        else:
            # 0. init
            ids = torch.full((bboxes.size(0),),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)

            # get the detection bboxes for the first association
            first_det_inds = bboxes[:, -1] > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_ids = ids[first_det_inds]

            # get the detection bboxes for the second association
            second_det_inds = (~first_det_inds) & (
                    bboxes[:, -1] > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_ids = ids[second_det_inds]

            # 1. use Kalman Filter to predict current location
            for id, track in self.tracks.items():
                # 如果上一帧未匹配到目标，可选择对速度归零或衰减
                if track.frame_ids[-1] != frame_id - 1:
                    track.mean[7] = 0
                track.age += 1
                (track.mean, track.covariance) = self.kf.predict(
                    track.mean, track.covariance)

            # -------------------------------
            # 2. 第一轮匹配：对 confirmed 轨迹进行匹配
            confirmed_ids = self.confirmed_ids
            # 提取 confirmed 轨迹的速度，若无则默认为 [0,0]
            vel_confirmed = np.array([
                self.tracks[id].velocity if hasattr(self.tracks[id], 'velocity') and self.tracks[
                    id].velocity is not None
                else np.array((0, 0)) for id in confirmed_ids
            ])
            # 提取 confirmed 轨迹的上一帧观测，取轨迹历史中最后一个检测框；若为空则用 [-1,-1,-1,-1]
            prev_obs_confirmed = np.array([
                self.tracks[id].bboxes[-1].cpu().numpy().squeeze(0) if len(self.tracks[id].bboxes) > 0
                else np.array([-1, -1, -1, -1, -1]) for id in confirmed_ids
            ])

            first_match_track_inds, first_match_det_inds = self.assign_ids(
                confirmed_ids, first_det_bboxes, first_det_labels,
                self.weight_iou_with_det_scores, self.match_iou_thrs['high'],
                apply_thermal_iou_adjustment=False,
                velocities=vel_confirmed, previous_obs=prev_obs_confirmed,
                vdc_weight=0.2
            )

            # 将匹配到的 confirmed 轨迹更新检测 id
            valid = first_match_det_inds > -1
            first_det_ids[valid] = torch.tensor(confirmed_ids)[first_match_det_inds[valid]].to(labels)

            first_match_det_bboxes = first_det_bboxes[valid]
            first_match_det_labels = first_det_labels[valid]
            first_match_det_ids = first_det_ids[valid]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = first_det_bboxes[~valid]
            first_unmatch_det_labels = first_det_labels[~valid]
            first_unmatch_det_ids = first_det_ids[~valid]
            assert (first_unmatch_det_ids == -1).all()

            # -------------------------------
            # 3. 对未匹配到的检测与 tentative（未确认）轨迹进行匹配
            unconfirmed_ids = self.unconfirmed_ids
            vel_unconfirmed = np.array([
                self.tracks[id].velocity if hasattr(self.tracks[id], 'velocity') and self.tracks[
                    id].velocity is not None
                else np.array((0, 0)) for id in unconfirmed_ids
            ])
            prev_obs_unconfirmed = np.array([
                self.tracks[id].bboxes[-1].cpu().numpy().squeeze(0) if len(self.tracks[id].bboxes) > 0
                else np.array([-1, -1, -1, -1, -1]) for id in unconfirmed_ids
            ])

            tentative_match_track_inds, tentative_match_det_inds = self.assign_ids(
                unconfirmed_ids, first_unmatch_det_bboxes, first_unmatch_det_labels,
                self.weight_iou_with_det_scores, self.match_iou_thrs['tentative'],
                apply_thermal_iou_adjustment=False,
                velocities=vel_unconfirmed, previous_obs=prev_obs_unconfirmed,
                vdc_weight=0.2
            )
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            # -------------------------------
            # 4. 第二轮匹配：对 confirmed 轨迹中未匹配的部分再次匹配
            first_unmatch_track_ids = []
            for i, id in enumerate(confirmed_ids):
                # tracklet 未在第一轮匹配中匹配上
                case_1 = first_match_track_inds[i] == -1
                # 且上一帧该 track 有更新
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1
                if case_1 and case_2:
                    first_unmatch_track_ids.append(id)
            if len(first_unmatch_track_ids) > 0:
                vel_subset = np.array([
                    self.tracks[id].velocity if hasattr(self.tracks[id], 'velocity') and self.tracks[
                        id].velocity is not None
                    else np.array((0, 0)) for id in first_unmatch_track_ids
                ])
                prev_obs_subset = np.array([
                    self.tracks[id].bboxes[-1].cpu().numpy().squeeze(0) if len(self.tracks[id].bboxes) > 0
                    else np.array([-1, -1, -1, -1, -1]) for id in first_unmatch_track_ids
                ])
                second_match_track_inds, second_match_det_inds = self.assign_ids(
                    first_unmatch_track_ids, second_det_bboxes, second_det_labels,
                    False, self.match_iou_thrs['low'], apply_thermal_iou_adjustment=False,
                    velocities=vel_subset, previous_obs=prev_obs_subset, vdc_weight=0.2
                )
                valid = second_match_det_inds > -1
                second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                    second_match_det_inds[valid]].to(ids)

            # -------------------------------
            # 5. 汇总所有匹配结果
            bboxes = torch.cat(
                (first_match_det_bboxes, first_unmatch_det_bboxes), dim=0)
            bboxes = torch.cat((bboxes, second_det_bboxes[second_det_ids > -1]), dim=0)

            labels = torch.cat(
                (first_match_det_labels, first_unmatch_det_labels), dim=0)
            labels = torch.cat((labels, second_det_labels[second_det_ids > -1]), dim=0)

            ids = torch.cat((first_match_det_ids, first_unmatch_det_ids), dim=0)
            ids = torch.cat((ids, second_det_ids[second_det_ids > -1]), dim=0)

            # 6. 对未匹配到的检测，分配新 id
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()

        self.update(ids=ids, bboxes=bboxes, labels=labels, frame_ids=frame_id)
        if -1 in self.ids:
            v = 1
        return bboxes, labels, ids
