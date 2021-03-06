#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import caffe
import argparse
import pprint
import numpy as np
import sys
from datasets.eleven_all import eleven_all
from datasets.eleven_12cat_bag import eleven_12cat_bag

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        default='/storage/ImageNet/ILSVRC2012/model/vgg/faster_rcnn_end2end/prototxt/solver.prototxt',
                        help='solver prototxt', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=90000000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default='/storage/ImageNet/ILSVRC2012/model/vgg/faster_rcnn_end2end/imagenet_models/VGG16.v2.caffemodel',
                        type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='/storage/ImageNet/ILSVRC2012/model/vgg/faster_rcnn_end2end/cfgs/faster_rcnn_end2end_train.yml', 
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    #import pdb; pdb.set_trace()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_device(cfg.GPU_ID)
    caffe.set_mode_gpu()

    #import pdb; pdb.set_trace()
    #image_path_prefix = '/storage/product/detection/11st_Bag'
    #loader = eleven_12cat_bag(image_path_prefix, 'train')
    image_path_prefix = '/storage/product/detection/11st_All'
    loader = eleven_all(image_path_prefix, 'train')
    output_dir = get_output_dir(loader, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    sys.stdout.flush()

    train_net(cfg.TRAIN.SOLVER_PROTOTXT, loader, output_dir,
              pretrained_model=cfg.TRAIN.CAFFE_MODEL,
              max_iters=args.max_iters)
