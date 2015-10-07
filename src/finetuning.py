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


import numpy as np
import cv2
import matplotlib.pyplot as plt
from fast_dbox_config import cfg
import utils.blob
import pdb

def get_minibatch(roidb):
    """
    Given a roidb, construct a minibatch sampled from it.
    """
    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    # Sample random scales to use for each image in this batch
    random_scale_inds = \
        np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    for im_i in xrange(num_images):
        labels, overlaps, im_rois = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image)

        # Add to ROIs blob
        rois = _scale_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        labels_blob = np.hstack((labels_blob, labels)) 
    
    return im_blob, rois_blob, labels_blob

def _sample_rois(roidb, fg_rois_per_image, rois_per_image):
    """
    Generate a random sample of ROIs comprising foreground and background
    examples.
    """
    # label = class ROI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground ROIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image,
                                   replace=False)

    # Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background ROIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image,
                                   replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background ROIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb['gt_overlaps'].shape[1]

    return labels, overlaps, rois

def _get_image_blob(roidb, scale_inds):
    """
    Build an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if im.shape[2]==1:
            im=np.dstack((im,im,im))
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = \
                utils.blob.prep_im_for_blob(im, cfg.PIXEL_MEANS,
                                            target_size, cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = utils.blob.im_list_to_blob(processed_ims)

    return blob, im_scales

def _scale_im_rois(im_rois, im_scale_factor):
    rois = im_rois * im_scale_factor
    return rois

