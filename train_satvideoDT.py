from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.utils.opts import opts
opt = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

import torch
import torch.utils.data
from lib.utils.logger import Logger

from lib.models.stNet import get_det_net,load_model, save_model
from lib.dataset.coco_icpr import COCO
from lib.Trainer.ctdet_1 import CtdetTrainer

def main(opt):
    torch.manual_seed(opt.seed)
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
    head = {'hm': DataTrain.num_classes, 'wh': 2, 'reg': 2}
    model = get_det_net(head, opt.model_name, opt.radius_mapping)  # 建立模型

    print(opt.model_name)
    print(opt.radius_mapping)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)  #设置优化器

    start_epoch = 0

    if(not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    if(not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    logger = Logger(opt)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)  # 导入训练好的模型


    trainer = CtdetTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    print('Starting training...')


    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('epoch: {} |'.format(epoch))

        save_model(os.path.join(opt.save_dir, 'model_icpr.pth'),
                   epoch, model, optimizer)

        for k, v in log_dict_train.items():
            logger.write('{} {:8f} | '.format(k, v))
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                   epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()

if __name__ == '__main__':
    main(opt)