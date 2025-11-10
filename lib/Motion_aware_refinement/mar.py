import torch
import numpy as np
import lap
from collections import defaultdict
import cv2
from lib.utils.utils import bbox_overlaps

def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)

def convert_detections_to_tensor(det_list):
    """
    从检测列表中提取出 [x1, y1, x2, y2] 并转换为 tensor，
    增加 batch 维度，最终形状为 (1, N, 4)
    """
    # 提取每个检测的前四个坐标
    boxes = [det[:4] for det in det_list]
    # 转换为 tensor，并确保数据类型为 float32
    return torch.tensor(boxes, dtype=torch.float32)

def motion_aware_refinement(opt, file_folder_buffer, image_buffer, dets_buffer,  inds_buffer):

    if file_folder_buffer[opt.seqLen - 1] != file_folder_buffer[0]:
        return dets_buffer, inds_buffer
    # 第0个是准的
    cur_img = image_buffer[0]  # dis: B, N-1, 2, H, W
    _, _, _, H, W = cur_img.shape
    cur_dets = defaultdict(list)
    whs = defaultdict(list)
    hits = defaultdict(list)
    match_inds = defaultdict(list)
    unmatch_preds = defaultdict(list)
    unmatch_curs = defaultdict(list)
    inds_temp_buffer = inds_buffer.copy()

    img_seq = image_buffer[opt.seqLen-1]
    for i in range(opt.seqLen-1):  # 0, 1, 2
        # cur dets
        curr_frame = img_seq[:, i, :, :, :]
        next_frame = img_seq[:, i+1, :, :, :]
        # device = curr_frame.device
        # 如果图像维度为 [B, 3, H, W]，则取第一个 batch，并转置为 HxWx3

        curr_img = cv2.cvtColor(curr_frame[0].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY).astype(np.uint8)
        next_img = cv2.cvtColor(next_frame[0].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY).astype(np.uint8)

        cur_dets[i] = cur_dets[i] + dets_buffer[i + 1]
        M = len(cur_dets[i])

        cur_cts = np.array(
            [((item[0] + item[2]) / 2, (item[1] + item[3]) / 2) for item in cur_dets[i]])  # M * 2, cur_cts
        cur_whs = np.array(
            [[(item[2] - item[0]), (item[3] - item[1]), item[4], item[5], item[6]] for item in cur_dets[i]])  # M * 4
        cur_sizes = np.array([((item[2] - item[0]) * (item[3] - item[1])) for item in cur_dets[i]])  # M, cur_det的尺寸集合


        # 第0帧用原始inds_buffer[0], 后续帧应该用更新的inds, 不过还是inds_buffer[i]
        if i == 0:
            pre_whs = np.array([[(item[2] - item[0]), (item[3] - item[1]), item[4], item[5], item[6]] for item in
                                dets_buffer[0]])  # K * 4
        else:
            pre_whs = np.array(whs[i - 1])  # M+K-T * 4

        K = len(inds_temp_buffer[i])
        orig_indices = np.array(inds_temp_buffer[i])
        if len(orig_indices) != pre_whs.shape[0]:
            print(f"警告: orig_indices 长度 ({len(orig_indices)}) 与 pre_whs 行数 ({pre_whs.shape[0]}) 不相等！")
        num_points = len(orig_indices)
        pre_det_list = []
        for p in range(num_points):
            idx = orig_indices[p]
            # 直接取 pre_whs 中第 i 行的信息
            w, h, score, cls, _ = pre_whs[p]
            # 通过 idx 计算中心点（假设索引按照图像从左到右、从上到下排列）
            cx = idx % W
            cy = idx / W
            # 构造检测框，利用中心点和宽高计算左上角与右下角
            new_x1 = np.clip(cx - w / 2., 0, W - 1)
            new_y1 = np.clip(cy - h / 2., 0, H - 1)
            new_x2 = np.clip(cx + w / 2., 0, W - 1)
            new_y2 = np.clip(cy + h / 2., 0, H - 1)
            pre_det_list.append([new_x1, new_y1, new_x2, new_y2, score, cls, idx])
        if len(pre_det_list) == 0:
            # 若没有检测点，则构造空的输入
            pre_cts_input = np.empty((0, 2), dtype=np.float32)
        else:
            # 从 pre_det_list 中提取中心点
            centers = np.array([[(det[0] + det[2]) / 2., (det[1] + det[3]) / 2.] for det in pre_det_list],
                               dtype=np.float32)
            pre_cts_input = centers  # 形状: (num_points, 2)

        # 利用光流计算匹配后的中心点（注意要求输入为 np.float32）
        if pre_cts_input.shape[0] == 0:
            next_pts = np.empty((0, 2), dtype=np.float32)
            status = np.empty((0, 1), dtype=np.uint8)
        else:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(curr_img, next_img, pre_cts_input, None)

        if status.size > 0:
            valid_mask = status.reshape(-1) == 1  # 布尔数组，长度与检测点数一致
        else:
            valid_mask = np.empty((0,), dtype=bool)


        pre_det_array = np.array(pre_det_list)

        # ===== 3. 利用匹配后的中心点和原检测信息更新检测框 =====
        new_det_list = []
        H_img, W_img = curr_img.shape  # 当前图像高度和宽度

        for p, orig_det in enumerate(pre_det_array):
            # 计算原始中心点（从构造的检测框得到）
            orig_center = np.array([(orig_det[0] + orig_det[2]) / 2.0,
                                    (orig_det[1] + orig_det[3]) / 2.0], dtype=np.float32)
            # 如果该点 optical flow 匹配成功，则采用更新后的中心点；否则保持原中心点
            if valid_mask[p]:
                new_center = next_pts[p]
            else:
                new_center = orig_center

            cx_new, cy_new = new_center
            # 保留原始的宽、高、score、cls
            w = orig_det[2] - orig_det[0]
            h = orig_det[3] - orig_det[1]
            score = orig_det[4]
            cls = orig_det[5]
            # 根据新的中心点重构检测框

            new_x1 = np.clip(cx_new - w / 2., 0, W - 1)
            new_y1 = np.clip(cy_new - h / 2., 0, H - 1)
            new_x2 = np.clip(cx_new + w / 2., 0, W - 1)
            new_y2 = np.clip(cy_new + h / 2., 0, H - 1)

            # 根据新的中心点重新计算索引，公式例如：new_index = round(cx_new) + round(cy_new) * W_img
            new_index = int(round(cx_new) + round(cy_new) * W_img)
            new_det = [new_x1, new_y1, new_x2, new_y2, score, cls, new_index]
            new_det_list.append(new_det)

        if len(new_det_list) == 0:
            pred_cts = np.empty((0, 2), dtype=np.float32)
        else:
            pred_cts = np.array([[(det[0] + det[2]) / 2., (det[1] + det[3]) / 2.] for det in new_det_list],
                                dtype=np.float32)

        if len(new_det_list) == 0 or len(cur_dets[i]) == 0:
            if len(new_det_list) == 0 and len(cur_dets[i]) == 0:
                ious = torch.empty((0, 0))
            elif len(new_det_list) == 0:
                ious = torch.empty((0, len(cur_dets[i])))
            else:
                ious = torch.empty((len(new_det_list), 0))
        else:
            new_det_tensor = convert_detections_to_tensor(new_det_list)
            cur_dets_tensor = convert_detections_to_tensor(cur_dets[i])
            ious = bbox_overlaps(new_det_tensor, cur_dets_tensor, mode='iou')

        dists = 1 - ious.cpu().numpy()
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=0.9)
        else:
            row = np.zeros(K).astype(np.int32) - 1
            col = np.zeros(M).astype(np.int32) - 1
        matched_indices = []
        for k, j in enumerate(row):
            if j != -1:
                matched_indices.append([k, j])
        if i == 0:
            # 第0帧似乎只需要unmatch_preds
            match_inds[i + 1] = np.array(matched_indices, np.int32).reshape(-1, 2)
            unmatch_preds[i + 1] = [d for d in range(pred_cts.shape[0]) if not (d in match_inds[i + 1][:, 0])]  # 对应K
            unmatch_curs[i + 1] = [d for d in range(cur_cts.shape[0]) if not (d in match_inds[i + 1][:, 1])]  # 对应M
        else:
            match_inds[i + 1] = np.array(matched_indices, np.int32).reshape(-1, 2)

        # 核心 更新第1个的inds和dets
        if i == 0:
            # 第1个需要更新unmatch_pred中的ind到cur_ind中去
            inds_temp = inds_temp_buffer[i + 1].tolist()  # 检测直出, 应该是M个, 并假设M与Kmatch到T个
            whs_temp = cur_whs.tolist()  # M * 5

            cts_update = np.array(pred_cts[unmatch_preds[i + 1], :], np.int32)  # K-T个
            inds_update = cts_update[:, 0] + cts_update[:, 1] * W
            whs_update = pre_whs[unmatch_preds[i + 1]].tolist()  # K-T * 5
            unmatch_pred0_update = (M + np.array(range(len(unmatch_preds[i + 1])))).tolist()

            inds_temp = inds_temp + inds_update.tolist()  # M+K-T个
            inds_temp_buffer[i + 1] = np.array(inds_temp, np.int64)
            whs[i] = whs_temp + whs_update  # M+K-T * 5

            hits[i] = np.zeros_like(inds_temp_buffer[i + 1])  # M+K-T
            hits[i][match_inds[i + 1][:, 1]] += 1
        else:
            hits[i] = np.zeros_like(inds_temp_buffer[i])
            hits[i][match_inds[i + 1][:, 0]] += 1
            cts_update = np.array(pred_cts, np.int32)
            inds_update = cts_update[:, 0] + cts_update[:, 1] * W

            for j in range(match_inds[i + 1].shape[0]):
                pre_whs[match_inds[i + 1][j, 0]][0:5] = cur_whs[match_inds[i + 1][j, 1]][0:5]
                inds_update[match_inds[i + 1][j, 0]] = cur_whs[match_inds[i + 1][j, 1]][4]

            inds_temp_buffer[i + 1] = np.array(inds_update, np.int64)
            whs[i] = pre_whs.tolist()

    inds_temp1 = inds_buffer[1].tolist()
    FN_ind, FP_ind = [], []

    # 对应处理FP, 没问题
    # hits_sum_1 = hits[0] + hits[1] + hits[2] + hits[3]  # seqLen == 5
    # remain_inds_1 = np.where(hits_sum_1 > 0)[0]
    # # remain_inds_1 = np.where((hits[0] > 0) & ((hits[1] > 0) | (hits[2] > 0)| (hits[3] > 0)))[0]
    # for i, ind in enumerate(inds_temp1):
    #     if ind not in inds_temp_buffer[1][remain_inds_1].tolist() and i in unmatch_curs[1]:
    #         FP_ind.append(i)
    # for i in sorted(FP_ind, reverse=True):
    #     del dets_buffer[1][i]
    #     del inds_temp1[i]

    # 对应处理FN
    hits_sum_2 = hits[0] + hits[1] + hits[2] + hits[3]  # seqLen == 5
    remain_inds_2 = np.where(hits_sum_2 > 0)[0]
    for i, num in enumerate(unmatch_pred0_update):  # 序号重排后
        temp_ind = inds_temp_buffer[1][num]
        # if ind not in inds_temp1 and i in remain_inds:
        #     FN_ind.append(i)
        if temp_ind > 0 and temp_ind < H * W:
            ct_y = temp_ind / W
            ct_x = temp_ind % W
            w, h, score, cls, _ = whs[0][num]
            if num in remain_inds_2:
                hits[0][num] += 1
                inds_temp1.append(temp_ind)
                dets_buffer[1].append(
                    [ct_x - w / 2., ct_y - h / 2., ct_x + w / 2., ct_y + h / 2., score, cls, temp_ind])

    inds_buffer[1] = np.array(inds_temp1)

    return dets_buffer, inds_buffer