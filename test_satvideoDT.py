from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##using both kalman filter and inter-frame distance check
import os
from lib.utils.opts import opts

opt = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
import cv2
import torch
import numpy as np
from collections import defaultdict

from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco_icpr import COCO
from lib.utils.byte_tracker import ByteTracker
from lib.Motion_aware_refinement.mar import motion_aware_refinement

from lib.utils.decode import ctdet_decode
from lib.utils.post_process import generic_post_process


from progress.bar import Bar

def process(model, image,vid=None):
    with torch.no_grad():
        output_all = model(image, training=False, vid=vid)[-1]
        output = output_all[1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        dets = ctdet_decode(hm, wh, reg=reg, num_classes=opt.num_classes, K=opt.K)
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()
    return dets


def post_process(dets_all, meta, scale=1):
    dets = generic_post_process(
        dets_all, [meta['c']], [meta['s']], meta['out_height'] // opt.down_ratio, meta['out_width'] // opt.down_ratio)
    if scale != 1:
        for i in range(len(dets[0])):
            for k in ['bbox']:
                if k in dets[0][i]:
                    dets[0][i][k] = (np.array(dets[0][i][k], np.float32) / scale).tolist()
    return dets[0]  # [item1, item2 ...]


def pre_process(image, scale=1):
    height, width = image.shape[3:5]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height,
            'out_width': inp_width}
    return meta


def merge_outputs(dets_all_post, num_class):
    results = []
    inds = []
    detections = dets_all_post
    for i in range(len(detections)):
        item = detections[i]
        temp_box = [item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3], item['score'][0], item['class'],
                    item['ind']]
        if item['score'][0] > opt.conf_thres:
            results.append(temp_box)
            inds.append(item['ind'])
    return results, np.array(inds)


def test(opt, split, modelPath, show_flag, results_name):
    # Logger(opt)
    print(opt.model_name)

    # ------------------load data and model------------------
    dataset = COCO(opt, split)
    num_classes = opt.num_classes
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model = get_det_net({'hm': num_classes, 'wh': 2, 'reg': 2}, opt.model_name, opt.radius_mapping)
    model = load_model(model, modelPath)
    model = model.cuda()
    model.eval()
    model_num = modelPath.split('/')[-1].split('.')[0]

    # some useful initialization
    file_folder_pre = 'INIT'
    trackingflag = 'off'
    bufferflag = 'off'
    im_count = 0
    buffer_size = opt.seqLen
    dets_track = defaultdict(list)
    dets_buffer = defaultdict(list)
    inds_buffer = defaultdict(list)
    dets_temp = defaultdict(list)
    image_buffer = defaultdict(list)
    image_temp = defaultdict(list)
    inds_temp = defaultdict(list)
    file_folder_buffer = defaultdict(list)

    saveTxt = opt.save_track_results
    if saveTxt:
        track_results_save_dir = os.path.join(opt.save_results_dir,
                                              'Results' + opt.model_name + '_' + model_num + '_conf_thres_' + str(
                                                  opt.conf_thres) + '_seq_' + str(opt.seqLen))
        if not os.path.exists(track_results_save_dir):
            os.mkdir(track_results_save_dir)
    # ------------------initialization ends------------------

    num_iters = len(data_loader)
    bar = Bar('processing', max=num_iters)
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):


        if (ind > len(data_loader) - 1):
            break

        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        # read images
        folder = pre_processed_images['file_name'][0].split('_')[0]
        file_folder_cur = folder

        meta = pre_process(pre_processed_images['input'], scale=1)
        image = pre_processed_images['input'].cuda()
        seq_num = opt.seqLen
        file_path = pre_processed_images['file_name'][0]
        parts = file_path.split('_')
        seq_folder = parts[0]  # e.g. '000001'
        img_name = parts[1]  # e.g. '000123.png'
        temp_dir = split + '_data'
        im_id_str, im_ext = os.path.splitext(img_name)
        img_path = opt.data_dir + temp_dir + '/' + seq_folder + '/img1/' + img_name

        # 先读一张样例图，获取 H,W,C
        sample_im = cv2.imread(img_path)
        H, W, C = sample_im.shape
        # 创建 (H,W,C,seq_num) 数组，依次填入前 seq_num 帧
        seq_imgs = np.zeros((H, W, C, seq_num), dtype=np.uint8)

        for ii in range(seq_num):
            idx = max(int(im_id_str) - seq_num + ii + 1, 1)
            new_name = f"{idx:06d}{im_ext}"
            read_path = opt.data_dir + temp_dir + '/' + seq_folder + '/img1/' + new_name
            seq_imgs[:, :, :, ii] = cv2.imread(read_path)

            # 转成 (1, seq_num, C, H, W) 并裁切到 32 的倍数
            inp = seq_imgs.transpose(3, 2, 0, 1).astype(np.float32)  # (seq_num, C, H, W)
            h_crop, w_crop = H - H % 32, W - W % 32
            inp = inp[:, :, :h_crop, :w_crop]
            inp = np.expand_dims(inp, axis=0)



        # use buffer or not
        if file_folder_cur != file_folder_pre:
            if bufferflag == 'on':
                # load model
                model = get_det_net({'hm': num_classes, 'wh': 2, 'reg': 2}, opt.model_name, opt.radius_mapping)
                model = load_model(model, modelPath)
                model = model.cuda()
                model.eval()
            bufferflag = 'off'

        # cur image detection
        dets_all = process(model, image, vid=file_folder_cur + '_' + str(im_count))
        dets_all_post = post_process(dets_all, meta)
        ret, cur_inds = merge_outputs(dets_all_post, num_classes)


        # detection buffer
        if file_folder_cur != file_folder_pre:
            for j in range(buffer_size):
                if j != buffer_size - 1:
                    file_folder_buffer[j] = file_folder_buffer[j + 1]
                    dets_temp[j] = dets_temp[j + 1]
                    inds_temp[j] = inds_temp[j + 1]
                    image_temp[j] = image_temp[j + 1]

                    dets_buffer[j] = dets_buffer[j + 1]
                    inds_buffer[j] = inds_buffer[j + 1]
                    image_buffer[j] = image_buffer[j + 1]
                else:
                    file_folder_buffer[j] = file_folder_cur
                    dets_temp[j] = ret
                    inds_temp[j] = cur_inds
                    image_temp[j] = inp
        else:
            for j in range(buffer_size):
                if j != buffer_size - 1:
                    file_folder_buffer[j] = file_folder_buffer[j + 1]
                    dets_buffer[j] = dets_buffer[j + 1]
                    inds_buffer[j] = inds_buffer[j + 1]
                    image_buffer[j] = image_buffer[j + 1]
                else:
                    file_folder_buffer[j] = file_folder_cur
                    dets_buffer[j] = ret
                    image_buffer[j] = inp
                    inds_buffer[j] = cur_inds

        if file_folder_buffer[0] != file_folder_pre and len(file_folder_buffer[0]) != 0:
            trackingflag = 'on'
            bufferflag = 'on'
            if saveTxt and file_folder_pre != 'INIT':
                fid.close()
            file_folder_pre = file_folder_buffer[0]

            car_tracker = ByteTracker(
                obj_score_thrs=dict(high=0.35, low=0.2),
                init_track_thr=0.2,
                weight_iou_with_det_scores=False,
                match_iou_thrs=dict(high=0.1, low=0.1, tentative=0.1),
                num_tentatives=2)
            dets_buffer = dets_temp
            inds_buffer = inds_temp
            image_buffer = image_temp
            if saveTxt:
                im_count = 0
                im_n = 0
                txt_path = os.path.join(track_results_save_dir, file_folder_buffer[0] + '.txt')
                fid = open(txt_path, 'w+')


        if trackingflag is not 'off':
            '''
            # input: dets_buffer, file_folder_buffer, inds_buffer
            '''
            # use buffer or not
            if bufferflag == 'on':
                dets_buffer, inds_buffer = motion_aware_refinement(opt, file_folder_buffer, image_buffer, dets_buffer, inds_buffer)
            for i in range(1, num_classes + 1):
                # 更新跟踪器
                track_temp = []
                label_temp = []
                for item in dets_buffer[0]:
                    if int(item[5]) == i:
                        track_temp.append(item[:5])
                        label_temp.append(int(item[5]))
                if track_temp:
                    dets_np = np.array(track_temp, dtype=np.float32)
                    valid_mask = (dets_np[:, 0] >= 0) & (dets_np[:, 1] >= 0) & \
                                 ((dets_np[:, 2] - dets_np[:, 0]) > 0) & \
                                 ((dets_np[:, 3] - dets_np[:, 1]) > 0)
                    dets_np = dets_np[valid_mask]
                    labels_np = np.array(label_temp, dtype=np.int64)[valid_mask]

                    if dets_np.shape[0] > 0:
                        dets_track[i] = dets_np
                        labels = labels_np
                    else:
                        dets_track[i] = np.empty((0, 5), dtype=np.float32)
                        labels = np.empty((0,), dtype=np.int64)

                else:
                    dets_track[i] = np.empty((0, 5))
                    labels = np.empty((0, 1), dtype=np.int64)

                dets_track_tensor = torch.tensor(dets_track[i], dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int64)

                # 根据类别选择对应的跟踪器
                if i == 1:  # 汽车
                    bboxes, labels, ids = car_tracker.track(dets_track_tensor, labels_tensor, im_n)
                    bboxes_np = bboxes.cpu().numpy()  # shape: (N, 5)
                    ids = ids + 1
                    ids_np = ids.cpu().numpy().reshape(-1, 1)  # shape: (N, 1)
                    car_track_bbs_ids = np.hstack([bboxes_np[:, :4], ids_np])  # shape: (N, 5)
                    car_track_bbs_ids = car_track_bbs_ids[::-1, :]
                    car_track_bbs_ids[:, 2:4] = car_track_bbs_ids[:, 2:4] - car_track_bbs_ids[:, :2]
            im_n += 1

            if saveTxt:
                im_count += 1
                for it in range(car_track_bbs_ids.shape[0]):
                    fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,1,1\n' % (im_count,
                                                                         car_track_bbs_ids[it, 4],
                                                                         car_track_bbs_ids[it, 0],
                                                                         car_track_bbs_ids[it, 1],
                                                                         car_track_bbs_ids[it, 2],
                                                                         car_track_bbs_ids[it, 3]))

        bar.next()
    bar.finish()


if __name__ == '__main__':
    split = 'val'
    show_flag = False
    if (not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    if opt.load_model != '':
        modelPath = opt.load_model
    else:
        modelPath = './checkpoints/SatvideoDT.pth'
    print(modelPath)

    results_name = opt.model_name + '_' + modelPath.split('/')[-1].split('.')[0]
    test(opt, split, modelPath, show_flag, results_name)
