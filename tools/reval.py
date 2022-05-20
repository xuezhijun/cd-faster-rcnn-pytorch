""" Reval = re-eval. Re-evaluate saved detections. """
from model.utils.config import cfg
from datasets.factory import get_imdb

import pickle
import os, sys, argparse
import numpy as np


def parse_args():
    """ Parse input argument. """
    parser = argparse.ArgumentParser(description='Re-evaluate results')

    parser.add_argument('output_dir', nargs=1, help='results directory',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to re-evaluate',
                        default='voc_2007_test', type=str)
    parser.add_argument('--matlab', dest='matlab_eval',
                        help='use matlab for evaluation',
                        action='store_true')
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--nms', dest='apply_nms', help='apply nms',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def nms_cpu(dets, thresh):
    # dets = dets.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    # return torch.IntTensor(keep)
    return keep


def apply_nms(all_boxes, thresh):
    """ Apply non-maximum suppression to all predicted boxes output by the test_net method. """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms_cpu(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def from_dets(imdb_name, output_dir, args):
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(args.comp_mode)
    imdb.config['matlab_eval'] = args.matlab_eval
    with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
        dets = pickle.load(f)

    if args.apply_nms:
        print('Applying NMS to all detections')
        nms_dets = apply_nms(dets, cfg.TEST.NMS)
    else:
        nms_dets = dets

    print('Evaluating detections')
    imdb.evaluate_detections(nms_dets, output_dir)


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir[0])
    imdb_name = args.imdb_name
    from_dets(imdb_name, output_dir, args)
