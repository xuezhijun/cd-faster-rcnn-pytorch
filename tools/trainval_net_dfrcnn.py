from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from model.utils.parser_func import parse_args, set_dataset_args
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import sampler, save_checkpoint, adjust_learning_rate, clip_gradient
from model.utils.net_utils import FocalLoss, EFocalLoss, CrossEntropy


def consistency_reg(N, d_image_y, d_inst_y, domain):
    y = d_image_y.sum(dim=0)
    L_cst = 0
    if domain == 'src':
        r = 0
        f = 1
    else:
        r = 1
        f = 1
        #f = 15
    y = y[r]
    #size = list(d_inst_y.size())[0]
    size = min(list(d_inst_y.size())[0], 128)
    for i in range(size):
        #for i in range(size/f)
        #L_cst += torch.norm((y/N - d_inst_y[f*i][r]), p=2)
        L_cst += torch.norm((y / N - d_inst_y[i][r]), p=2)
    #print(i)
    return L_cst


if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)
    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda

    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)
    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size, sampler=sampler_batch, num_workers=args.num_workers)

    sampler_batch_t = sampler(train_size_t, args.batch_size)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size, sampler=sampler_batch_t, num_workers=args.num_workers)

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initialize the network here.
    from model.faster_rcnn.vgg16_dfrcnn import vgg16
    from model.faster_rcnn.resnet_dfrcnn import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    #tr_momentum = cfg.TRAIN.MOMENTUM
    #tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        print("loading checkpoint %s" % args.load_name)
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % args.load_name)

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    #iters_per_epoch = int(train_size / args.batch_size)
    iters_per_epoch = int(10000 / args.batch_size)

    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")

    #### epoch begin ####
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        #### step begin ####
        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)

            # put source data into variable
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            fasterRCNN.zero_grad()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, \
            out_d_img, out_d_inst, dim = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            # domain label
            domain_s = Variable(torch.zeros(out_d_img.size(0)).long().cuda())
            inst_s = Variable(torch.zeros(out_d_inst.size(0)).long().cuda())
            # main loss
            dloss_s = 0.1 * CrossEntropy(out_d_img, domain_s)
            # instance loss
            dloss_s_p = 0.1 * CrossEntropy(out_d_inst, inst_s)
            # consistency regularisation
            cst_loss_s = 0.1 * consistency_reg(dim, out_d_img, out_d_inst, domain='src')

            # put target data into variable
            im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
            # gt is empty
            gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()

            out_d_img, out_d_inst, dim = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)

            # domain label
            domain_t = Variable(torch.ones(out_d_img.size(0)).long().cuda())
            inst_t = Variable(torch.ones(out_d_inst.size(0)).long().cuda())
            # main loss
            dloss_t = 0.1 * CrossEntropy(out_d_img, domain_t)
            # instance loss
            dloss_t_p = 0.1 * CrossEntropy(out_d_inst, inst_t)
            # consistency regularisation
            cst_loss_t = 0.1 * consistency_reg(dim, out_d_img, out_d_inst, domain='tar')

            if args.dataset == 'sim10k':
                loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + cst_loss_s + cst_loss_t) * args.eta
            else:
                loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + cst_loss_s + cst_loss_t)

            if args.mGPUs:
                loss_temp += loss.mean().item()
            else:
                loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            if args.mGPUs:
                loss = loss.mean()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            #### step-display begin ####
            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()

                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()

                    dloss_s = dloss_s.item()
                    dloss_t = dloss_t.item()
                    dloss_s_p = dloss_s_p.item()
                    dloss_t_p = dloss_t_p.item()
                    cst_loss_s = cst_loss_s.item()
                    cst_loss_t = cst_loss_t.item()

                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\t fg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\t rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                print("\t\t\t dloss_s: %.4f, dloss_t: %.4f, dloss_s_pixel: %.4f, dloss_t_pixel: %.4f" % (dloss_s, dloss_t, dloss_s_p, dloss_t_p))
                print("\t\t\t consistency_s: %.4f, consistency_t: %.4f" % (cst_loss_s, cst_loss_t))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()
            #### step-display end ####
        #### step end ####

        save_name = os.path.join(output_dir,
                                 'dfrcnn_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                     args.dataset_t, args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))
    #### epoch end ####

    if args.use_tfboard:
        logger.close()
