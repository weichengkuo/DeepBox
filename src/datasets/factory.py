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



__sets = {}

import datasets.coco_imdb
import numpy as np

def _proposals_top_k(split, year, top_k):
    imdb = datasets.coco_imdb(split, year)
    imdb.roidb_handler = imdb.proposals_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2014','2015']:
    for split in ['train', 'val','test','test-dev']:
        name = 'coco_{}{}'.format(split, year)
        __sets[name] = (lambda split=split, year=year:
                datasets.coco_imdb(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(500, 5000, 500):
    for year in ['2015','2014']:
        for split in ['test','test-dev','train', 'val']:            
            name = 'coco_{}{}_top_{:d}'.format(split, year, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _proposals_top_k(split, year, top_k))

def get_imdb(name):
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    return __sets.keys()
