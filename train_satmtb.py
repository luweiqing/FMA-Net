from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.utils.opts import opts
import math
opt = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.utils.data
from lib.utils.logger import Logger


from lib.models.stNet import get_det_net,load_model, save_model
from lib.dataset.coco_mtb import COCO
from lib.Trainer.ctdet import CtdetTrainer


class MultiStepLR:
    def __init__(self, optimizer, steps, gamma=0.1, iters_per_epoch=None, warmup=None, warmup_iters=None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.steps = steps
        self.steps.sort()
        self.gamma = gamma
        self.iters_per_epoch = iters_per_epoch
        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter=None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return

        # multi policy
        if self.iters % self.iters_per_epoch == 0:
            epoch = int(self.iters / self.iters_per_epoch)
            power = -1
            for i, st in enumerate(self.steps):
                if epoch < st:
                    power = i
                    break
            if power == -1:
                power = len(self.steps)

            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * (self.gamma ** power)


class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0, warmup=None, warmup_iters=None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter=None):
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return

        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2

def get_scheduler(optimizer, cfg, iters_per_epoch):
    if cfg.scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, cfg.steps, cfg.gamma, iters_per_epoch, cfg.warmup, iters_per_epoch if cfg.warmup_iters is None else cfg.warmup_iters)
    elif cfg.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, cfg.num_epochs * iters_per_epoch, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)
    else:
        raise NotImplementedError
    return scheduler

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    DataTrain = COCO(opt, 'train')


    train_loader = torch.utils.data.DataLoader(
        DataTrain,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Creating model...')
    head = {'hm': DataTrain.num_classes, 'wh': 2, 'reg': 2, 'dis': 2}
    model = get_det_net(head, opt.model_name, opt.radius_mapping)  # 建立模型
    print(opt.model_name)
    print(opt.radius_mapping)
    num_iters = len(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)  #设置优化器
    scheduler = get_scheduler(optimizer, opt, num_iters)


    start_epoch = 0

    if(not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    if(not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    logger = Logger(opt)

    if opt.load_model != '':
        model, optimizer, start_epoch= load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)  # 导入训练好的模型


    trainer = CtdetTrainer(opt, model, optimizer, scheduler)
    trainer.set_device(opt.gpus, opt.device)

    print('Starting training...')

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('epoch: {} |'.format(epoch))

        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                   epoch, model, optimizer)

        for k, v in log_dict_train.items():
            logger.write('{} {:8f} | '.format(k, v))
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                   epoch, model, optimizer)
        logger.write('\n')

    trainer.plot_lr_curve()
    logger.close()

if __name__ == '__main__':
    main(opt)