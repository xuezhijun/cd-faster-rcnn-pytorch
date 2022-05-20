from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pprint
import pdb
import time
import cv2

import torch
from torch.autograd import Variable

from model.utils.parser_func import parse_args, set_dataset_args
from model.utils.config import cfg, cfg_from_file, cfg_from_list

from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from model.roi_layers import nms
from model.utils.net_utils import vis_detections
from model.utils.blob import im_list_to_blob


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


'''
def _get_image_blob(im):
    """
    Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)
'''


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims = []
    im_scale_factors = []

    im_scale = 1.0
    im_scale_factors.append(im_scale)
    processed_ims.append(im_orig)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    args = parse_args()
    #print('Called with args:')
    #print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    #print('Using config:')
    #pprint.pprint(cfg)

    cfg.USE_GPU_NMS = args.cuda

    np.random.seed(cfg.RNG_SEED)

    if args.dataset_t == "cityscape_car":
        pascal_classes = np.asarray(['__background__',
                                     'car'])
    elif args.dataset_t == "cityscape_synthia":
        pascal_classes = np.asarray(['__background__',
                                     'car', 'bus', 'motorcycle',
                                     'bicycle', 'person', 'rider'])
    elif args.dataset_t == "water":
        pascal_classes = np.asarray(['__background__',
                                     'bicycle', 'bird', 'car',
                                     'cat', 'dog', 'person'])
    elif args.dataset_t == "clipart":
        pascal_classes = np.asarray(['__background__',
                                     'aeroplane', 'bicycle', 'bird', 'boat',
                                     'bottle', 'bus', 'car', 'cat', 'chair',
                                     'cow', 'diningtable', 'dog', 'horse',
                                     'motorbike', 'person', 'pottedplant',
                                     'sheep', 'sofa', 'train', 'tvmonitor'])
    elif args.dataset_t == "foggy_cityscape":
        pascal_classes = np.asarray(['__background__',
                                     'bus', 'bicycle', 'car',
                                     'motorcycle', 'person', 'rider',
                                     'train', 'truck'])
    else:
        print("No such dataset_t yet!")
        pdb.set_trace()

    # initialize the network here.
    if args.cd_method == 'baseline':
        from model.faster_rcnn.vgg16 import vgg16
        from model.faster_rcnn.resnet import resnet

        if args.net == 'vgg16':
            fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

    elif args.cd_method == 'coral':
        from model.faster_rcnn.vgg16_d import vgg16
        #from model.faster_rcnn.resnet_d import resnet

        if args.net == 'vgg16':
            fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic, fc=args.fc, cv=args.cv)
        #elif args.net == 'res50':
            #fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic, fc=args.fc, cv=args.cv)
        #elif args.net == 'res101':
            #fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic, fc=args.fc, cv=args.cv)
        else:
            print("network is not defined")
            pdb.set_trace()

    elif args.cd_method == 'DA':
        from model.faster_rcnn.vgg16_dfrcnn import vgg16
        from model.faster_rcnn.resnet_dfrcnn import resnet

        if args.net == 'vgg16':
            fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

    elif args.cd_method == 'SW':
        from model.faster_rcnn.vgg16_global_local import vgg16
        from model.faster_rcnn.resnet_global_local import resnet

        if args.net == 'vgg16':
            fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)
        elif args.net == 'res50':
            fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)
        elif args.net == 'res101':
            fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)
        else:
            print("network is not defined")
            pdb.set_trace()

    elif args.cd_method == 'SCL':
        from model.faster_rcnn.vgg16_SCL import vgg16
        from model.faster_rcnn.resnet_SCL import resnet

        if args.net == 'vgg16':
            fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

    # elif args.cd_method == 'GPA':
    # from model.faster_rcnn.vgg16_GPA import vgg16
    # from model.faster_rcnn.resnet_GPA import resnet
    # if args.net == 'vgg16':
    # fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic, mode=args.mode, rpn_mode=args.rpn_mode)
    # elif args.net == 'res50':
    # fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic, mode=args.mode, rpn_mode=args.rpn_mode)
    # elif args.net == 'res101':
    # fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic, mode=args.mode, rpn_mode=args.rpn_mode)
    # else:
    # print("network is not defined")
    # pdb.set_trace()

    else:
        print("No such cd method!")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("loading checkpoint from %s" % args.load_name)
    if args.cuda > 0:
        checkpoint = torch.load(args.load_name)
    else:
        checkpoint = torch.load(args.load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    #fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('loaded model successfully!')
    #pdb.set_trace()
    print("loaded checkpoint from %s" % args.load_name)

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True
    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True

    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)
        num_images = 0
    else:
        imglist = os.listdir(args.image_dir)
        num_images = len(imglist)
        print('Loaded Photo: {} images.'.format(num_images))

    while num_images >= 0:
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        else:
            im_file = os.path.join(args.image_dir, imglist[num_images])
            im_in = cv2.imread(im_file)

        im = im_in

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented!"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)  # NCHW <- NHWC
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)  # (N,C,H,W)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)  # (N,[H,W,Scale])
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        #pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)[:3]

        # from thop import profile
        # inputs = (im_data, im_info, gt_boxes, num_boxes)
        # flops, params = profile(fasterRCNN, inputs)
        # print('FLOPs = ' + str(flops/1e9) + '{}'.format('G'))
        # print('params = ' + str(params/1e6) + '{}'.format('M'))

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            if args.cuda > 0:
                pred_boxes = torch.from_numpy(np.tile(boxes, (1, scores.shape[1]))).cuda()
            else:
                pred_boxes = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        if vis:
            im2show = np.copy(im)
        for j in range(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                #cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s \r'.format(num_images + 1, len(imglist), detect_time, nms_time))
            sys.stdout.flush()

        if vis and webcam_num == -1:
            cv2.imshow('test', im2show)
            cv2.waitKey(0)
            result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
        else:
            # im2show = cv2.resize(im2show, (1200,600))
            cv2.imshow("frame", im2show)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            sys.stdout.write('Frame rate: {:.3f}fps \r'.format(frame_rate))
            sys.stdout.flush()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()
