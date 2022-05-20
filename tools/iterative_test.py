import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='iteratively test models')

    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='start epoch',
                        default=1, type=int)
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='max epochs',
                        default=10, type=int)

    parser.add_argument('--test_script', dest='test_script',
                        help='test script',
                        default='tools/test_net.py', type=str)

    parser.add_argument('--dataset', dest='dataset',
                        help='val dataset',
                        default='cityscape_car', type=str)

    parser.add_argument('--prefix', dest='prefix',
                        help='path to load models',
                        default='models/vgg16/sim10k/', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        template = 'python {} --cuda --net vgg16 --dataset {} --load_name {}_session_1_epoch_{}_step_9999.pth'
        cmd = template.format(args.test_script, args.dataset, args.prefix, epoch)
        os.system(cmd)
