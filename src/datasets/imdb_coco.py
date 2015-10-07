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


import os
import PIL
import utils.cython_bbox
import numpy as np
import scipy.sparse
from fast_dbox_config import cfg
import pdb

class imdb(object):
    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
#        self._obj_proposer = 'deep_box'
        self._obj_proposer = 'edge_box'
#        self._obj_proposer = 'slid_window' 
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # self.coco = []
        # Use this dict for storing dataset specific config options
        self.config = {'top_k':1000}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return 2

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index
    
#    @property
#    def coco(self):
#        return self.coco
    
    @property
    def roidb_handler(self):
        return self._roidb_handler
    
    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        return os.path.join(cfg.ROOT_DIR, 'data', 'cache')

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def append_flipped_roidb(self):
        num_images = self.num_images
        widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            #pdb.set_trace()
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
    def test_roidb_from_box_list(self, box_list):
        assert len(box_list) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({'boxes' : boxes,
                          'gt_classes' : np.zeros((num_boxes,),
                                                  dtype=np.int32),
                          'gt_overlaps' : overlaps,
                          'flipped' : False})
        #pdb.set_trace()
        return roidb

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            gt_boxes = gt_roidb[i]['boxes']
            gt_classes = gt_roidb[i]['gt_classes']
            gt_overlaps = \
                    utils.cython_bbox.bbox_overlaps(boxes.astype(np.float),
                                                    gt_boxes.astype(np.float))
            if gt_overlaps.shape[1]==0:
                overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
                overlaps = scipy.sparse.csr_matrix(overlaps)
            else:
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                #num_classes+1 to include background class
                overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
                overlaps = scipy.sparse.csr_matrix(overlaps)
            
            roidb.append({'boxes' : boxes,
                          'gt_classes' : np.zeros((num_boxes,),
                                                  dtype=np.int32),
                          'gt_overlaps' : overlaps,
                          'flipped' : False})
        #pdb.set_trace()
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        #pdb.set_trace()
        for i in xrange(len(a)):
            print 'img %d'%i
            if b[i]['gt_overlaps'] is None:
                raise ValueError('b[i] gt overlaps is None')
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
        #    pdb.set_trace()
            print type(a[i]['gt_overlaps'])
            print type(b[i]['gt_overlaps'])
            if b[i]['gt_overlaps'].toarray().size > 0 and a[i]['gt_overlaps'].toarray().size > 0:
                a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],b[i]['gt_overlaps']])
            if b[i]['gt_overlaps'].toarray().size > 0 and a[i]['gt_overlaps'].toarray().size == 0:
                a[i]['gt_overlaps'] = b[i]['gt_overlaps']

            #    a[i]['gt_overlaps'] = np.vstack((a[i]['gt_overlaps'],b[i]['gt_overlaps']))
        return a
