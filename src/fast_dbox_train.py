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



from fast_dbox_config import cfg, get_output_path
import numpy as np
import cv2
import caffe
import finetuning
import os
import pdb

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    def __init__(self, solver_prototxt, imdb, pretrained_model=None):
        self.imdb = imdb

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print 'Loading pretrained model weights from {:s}' \
                .format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
        
        #Randomly initialize fc layers
        np.random.seed(0)
        rand_fc3 = np.random.normal(0, 0.005, self.solver.net.params['fc3'][0].data.shape)
        np.random.seed(0)
        rand_cls_score = np.random.normal(0, 0.01, self.solver.net.params['cls_score'][0].data.shape)
        self.solver.net.params['fc3'][0].data[:,:] = rand_fc3
        self.solver.net.params['cls_score'][0].data[:,:] = rand_cls_score
        
        #pdb.set_trace()
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def snapshot(self):
        
        output_dir = get_output_path(self.imdb, None)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = self.solver_param.snapshot_prefix + infix + \
                   '_iter_{:d}'.format(self.solver.iter) + '.caffemodel'
        filename = os.path.join(output_dir, filename)

        self.solver.net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        # restore net to original state
       
    def train_model(self, roidb, max_iters):
        last_snapshot_iter = -1
        while self.solver.iter < max_iters:
            shuffled_inds = np.random.permutation(np.arange(len(roidb)))
            lim = (len(shuffled_inds) / cfg.TRAIN.IMS_PER_BATCH) * \
                    cfg.TRAIN.IMS_PER_BATCH
            shuffled_inds = shuffled_inds[0:lim]
            for shuffled_i in xrange(0, len(shuffled_inds),
                                     cfg.TRAIN.IMS_PER_BATCH):
                db_inds = shuffled_inds[shuffled_i:shuffled_i +
                            cfg.TRAIN.IMS_PER_BATCH]
                minibatch_db = [roidb[i] for i in db_inds]
                im_blob, rois_blob, labels_blob = finetuning.get_minibatch(minibatch_db)
                
               # print db_inds
                if rois_blob.shape[0]==0:
                    print 'rois_blobs is empty ...'
                    continue
                
                # Reshape net's input blobs
                net = self.solver.net
                
                base_shape = im_blob.shape
                num_rois = rois_blob.shape[0]

                net.blobs['data'].reshape(base_shape[0], base_shape[1],
                                          base_shape[2], base_shape[3])
                net.blobs['rois'].reshape(num_rois, 5, 1, 1)
                net.blobs['labels'].reshape(num_rois, 1, 1, 1)

                # Copy data into net's input blobs
                net.blobs['data'].data[...] = \
                    im_blob.astype(np.float32, copy=False)

                net.blobs['rois'].data[...] = \
                    rois_blob[:, :, np.newaxis, np.newaxis] \
                    .astype(np.float32, copy=False)

                net.blobs['labels'].data[...] = \
                    labels_blob[:, np.newaxis, np.newaxis, np.newaxis] \
                    .astype(np.float32, copy=False)

                # Make one SGD update
                self.solver.step(1)

                if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                    last_snapshot_iter = self.solver.iter
                    self.snapshot()

                if self.solver.iter >= max_iters:
                    break

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

def prepare_training_roidb(imdb):
    """
    Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

    return roidb

def train_net(solver_prototxt, imdb, pretrained_model=None, max_iters=40000):
    # enhance roidb to contain flipped examples
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_roidb()
        print 'done'

    # enhance roidb to contain some useful derived quanties
    print 'Preparing training data...'
    roidb = prepare_training_roidb(imdb)
    print 'done'

    sw = SolverWrapper(solver_prototxt, imdb, pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(roidb, max_iters=max_iters)
    print 'done solving'
