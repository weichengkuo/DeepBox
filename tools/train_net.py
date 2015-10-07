#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Fast DeepBox
# Written by Weicheng Kuo, 2015.
# See LICENSE in the project root for license information.
# --------------------------------------------------------


import pdb
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', 'src')))

from fast_dbox_config import cfg, cfg_from_file
import fast_dbox_train
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a fast DeepBox')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver', help='solver prototxt',
                        default='./models/DboxNet/solver.prototxt', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=120000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default='./data/imagenet_models/CaffeNet.v2.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='coco_train2014', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using fast_dbox_config:')
    pprint.pprint(cfg)

    # fix the random seed for reproducibility
    np.random.seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    imdb_train = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb_train.name)
#    pdb.set_trace() 

    fast_dbox_train.train_net(args.solver, imdb_train,
                              pretrained_model=args.pretrained_model,
                              max_iters=args.max_iters)


