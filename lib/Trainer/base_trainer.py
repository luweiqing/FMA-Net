from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from lib.utils.data_parallel import DataParallel
from lib.utils.utils import AverageMeter
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
import numpy as np
import os
import matplotlib.pyplot as plt

def post_process(output, meta, num_classes, scale=1):
    # decode
    hm = output[1]['hm'].sigmoid_()
    wh = output[1]['wh']
    reg = output[1]['reg']

    torch.cuda.synchronize()
    dets = ctdet_decode(hm, wh, reg=reg,num_classes= num_classes)
    for k in dets:
        dets[k] = dets[k].detach().cpu().numpy()
    # dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    # for j in range(1, num_classes + 1):
    #     dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    #     dets[0][j][:, :4] /= scale
    return dets[0]

def merge_outputs(detections, num_classes ,max_per_image):
    results = []
    inds = []
    detections = detections
    for i in range(len(detections)):
        item = detections[i]
        temp_box = [item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3], item['score'][0], item['class'],
                    item['ind']]
        if item['score'][0] > 0.2:
            results.append(temp_box)
            inds.append(item['ind'])
    return results, np.array(inds)


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # print(batch['input'].shape)
        outputs = self.model(batch['input'])#
        loss, loss_stats = self.loss(outputs, batch)  #19892
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None, scheduler=None):
        self.opt = opt
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        self.lr_history = []

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            gpus = list(range(len(gpus)))
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
            torch.cuda.empty_cache()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        # bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
              break
            data_time.update(time.time() - end)
            global_step = (epoch - 1) * num_iters + iter_id + 1

            for k in batch:
                if k != 'meta' and k != 'file_name' and k not in [1, 2, 4]:
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
                elif k in [1, 2, 4]:
                    for item in batch[k]:
                        batch[k][item] = batch[k][item].to(device=opt.device, non_blocking=True)  ##把图片和heatmap送到GPU
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean() #mean就是batch中多个样本的多个点求完每个点分别计算loss，然后求平均，还有sum，就是全部点的loss和，或者不选，就是全部点的loss都输出
            if phase == 'train':
                self.optimizer.zero_grad() #19892
                loss.backward()
                self.scheduler.step(global_step)
                self.optimizer.step()  #4434
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_history.append((global_step, current_lr))
            batch_time.update(time.time() - end)

            print('phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f, seq_loss=%0.4f' \
                  % (phase, epoch,iter_id+1,num_iters, time.time() - end,
                     loss.mean().cpu().detach().numpy(),
                     loss_stats['hm_loss'].mean().cpu().detach().numpy(),
                     loss_stats['wh_loss'].mean().cpu().detach().numpy(),
                     loss_stats['off_loss'].mean().cpu().detach().numpy(),
                     loss_stats['seq_loss'].mean().cpu().detach().numpy()))

            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats


        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = 1 / 60.

        return ret, results

    def plot_lr_curve(self):
        """绘制学习率变化曲线"""
        steps, lrs = zip(*self.lr_history)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lrs, label='Learning Rate')
        plt.xlabel('Global Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_eval_epoch(self, phase, epoch, data_loader, base_s, dataset):
        model_with_loss = self.model_with_loss

        if len(self.opt.gpus) > 1:
            model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        end = time.time()

        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
              break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta' and k != 'file_name'and k not in [1, 2, 4]:
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
                elif k in [1, 2, 4]:
                    for item in batch[k]:
                        batch[k][item] = batch[k][item].to(device=opt.device, non_blocking=True)  ##把图片和heatmap送到GPU
            output, loss, loss_stats = model_with_loss(batch)

            inp_height, inp_width = batch['input'].shape[3],batch['input'].shape[4]
            c = np.array([inp_width / 2., inp_height / 2.], dtype=np.float32)
            s = max(inp_height, inp_width) * 1.0

            meta = {'c': c, 's': s,
                    'out_height': inp_height,
                    'out_width': inp_width}

            dets = post_process(output, meta, opt.num_classes)
            ret,_ = merge_outputs(dets, opt.num_classes, max_per_image=opt.K)
            results[im_id.numpy().astype(np.int32)[0]] = ret

            loss = loss.mean()
            batch_time.update(time.time() - end)

            print(
                'phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f, seq_loss=%0.4f' \
                % (phase, epoch, iter_id + 1, num_iters, time.time() - end,
                   loss.mean().cpu().detach().numpy(),
                   loss_stats['hm_loss'].mean().cpu().detach().numpy(),
                   loss_stats['wh_loss'].mean().cpu().detach().numpy(),
                   loss_stats['off_loss'].mean().cpu().detach().numpy(),
                   loss_stats['seq_loss'].mean().cpu().detach().numpy()))
            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats
            # torch.cuda.empty_cache()

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        stats1, _ = dataset.run_eval(results, opt.save_results_dir, 'latest')
        ret['time'] = 1 / 60.
        ret['ap50'] = stats1[1]

        return ret, results, stats1

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader, base_s, dataset):

        return self.run_eval_epoch('val', epoch, data_loader, base_s, dataset)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)