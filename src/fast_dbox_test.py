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
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
import cPickle
import heapq
import utils.blob
import os
import pdb
import scipy.io as sio

def _get_image_blob(im):
    im_pyra = []
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
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = utils.blob.im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    rois, levels = _scale_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))[:, :, np.newaxis, np.newaxis]
    return rois_blob.astype(np.float32, copy=False)

def _scale_im_rois(im_rois, scales):
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def im_obj_detect(net, im, boxes):
    blobs, im_scale_factors = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'][:, :, 0, 0] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :, :, :]
        boxes = boxes[index, :]

    # reshape network inputs
    base_shape = blobs['data'].shape
    num_rois = blobs['rois'].shape[0]
    net.blobs['data'].reshape(base_shape[0], base_shape[1],
                              base_shape[2], base_shape[3])
    net.blobs['rois'].reshape(num_rois, 5, 1, 1)

    rois=blobs['rois'].astype(np.float32, copy=False)
    data=blobs['data'].astype(np.float32, copy=False)
    
    blobs_out = net.forward(data=data,rois=rois)
    # use softmax estimated probabilities
    scores = blobs_out['cls_prob']
    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]

    return scores

def test_net(net, imdb):
    num_images = len(imdb.image_index)
    output_dir = get_output_path(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    score_file = os.path.join(output_dir, 'fast_dbox_output_scores')
    roidb = imdb.roidb
    score_list = []
    batch_size = 3000 
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    sum_time = 0
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        num_gt = len(np.nonzero(roidb[i]['gt_classes'])[0])
        boxes = roidb[i]['boxes'][num_gt:,:]
        print 'Process image {:d},{:d} boxes'.format(i,boxes.shape[0])
        _t['im_detect'].tic()
       
        #Pass the boxes through the net in batches 
        if boxes.shape[0] % batch_size == 0:
            rid = boxes.shape[0]/batch_size
        else:
            rid = boxes.shape[0]/batch_size+1

        for j in xrange(rid):
            start_ind = batch_size*j
            end_ind = min(start_ind+batch_size,boxes.shape[0])
            boxes_ = boxes[start_ind:end_ind]
            scores_ = im_obj_detect(net, im, boxes_)
            if (j==0):
                scores = scores_
            else:
                scores = np.concatenate((scores, scores_), axis=0)
        
        _t['im_detect'].toc()
        print 'image {:d}/{:d} {:.3f}s'.format(i+1,num_images,_t['im_detect'].average_time)
        sum_time = sum_time+_t['im_detect'].average_time
        score_list.append(scores)

    print 'Average time = {:.3f}s'.format(sum_time/num_images)
    #Save Edge boxes score in both .mat and .pkl format
    sio.savemat(score_file+'.mat', {'score_list':score_list})
    f_score = open(score_file+'.pkl','wb')
    cPickle.dump(score_list,f_score)
    f_score.close()
    
def demo_net(net, imdb, frame_num,num_boxes_vis):
    import matplotlib.pyplot as plt 
    num_images = len(imdb.image_index)
    roidb = imdb.roidb
    score_list = []
    batch_size = 3000 
    # timer
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    im = cv2.imread(imdb.image_path_at(frame_num))
    num_gt = len(np.nonzero(roidb[frame_num]['gt_classes'])[0])
    boxes = roidb[frame_num]['boxes'][num_gt:,:]
    print 'Process image {:d},{:d} boxes'.format(frame_num,boxes.shape[0])
    _t['im_detect'].tic()
   
    #Pass the boxes through the net in batches 
    if boxes.shape[0] % batch_size == 0:
        rid = boxes.shape[0]/batch_size
    else:
        rid = boxes.shape[0]/batch_size+1

    for j in xrange(rid):
        start_ind = batch_size*j
        end_ind = min(start_ind+batch_size,boxes.shape[0])
        boxes_ = boxes[start_ind:end_ind]
        scores_ = im_obj_detect(net, im, boxes_)
        if (j==0):
            scores = scores_[:,1]
        else:
            scores = np.concatenate((scores, scores_[:,1]), axis=0)
    
    _t['im_detect'].toc()
    print 'image {:d}/{:d} {:.3f}s'.format(frame_num+1,num_images,_t['im_detect'].average_time)
    indices = scores.argsort()[::-1]
    fboxes = boxes[indices,:]

    #Visualize top scoring boxes 
    im = im[:, :, (2, 1, 0)] 
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
   # pdb.set_trace()
    for i in xrange(num_boxes_vis):
        bbox = fboxes[i,:]
        score = scores[indices[i]]    
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                'Objectness:{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title('Fast DeepBox proposals on frame #{:d} of COCO val set'.format(frame_num),fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def demo_net_quick(net, num_boxes_vis):
    import matplotlib.pyplot as plt 
    score_list = []
    batch_size = 3000 
    num_images = 4
    frame_num = 0 
    # timer
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    filename = './data/demo/demo_boxes.mat'
    demo_boxes = sio.loadmat(filename)['boxes']
    for frame_num in range(4): 
        im = cv2.imread('./data/demo/im{:d}.jpg'.format(frame_num))
        boxes = demo_boxes[frame_num][0]
        print 'Process image {:d},{:d} boxes'.format(frame_num,boxes.shape[0])
        _t['im_detect'].tic()
        
        #Pass the boxes through the net in batches 
        if boxes.shape[0] % batch_size == 0:
            rid = boxes.shape[0]/batch_size
        else:
            rid = boxes.shape[0]/batch_size+1

        for j in xrange(rid):
            start_ind = batch_size*j
            end_ind = min(start_ind+batch_size,boxes.shape[0])
            boxes_ = boxes[start_ind:end_ind]
            scores_ = im_obj_detect(net, im, boxes_)
            if (j==0):
                scores = scores_[:,1]
            else:
                scores = np.concatenate((scores, scores_[:,1]), axis=0)
        
        _t['im_detect'].toc()
        print 'image {:d}/{:d} {:.3f}s'.format(frame_num+1,num_images,_t['im_detect'].average_time)
        indices = scores.argsort()[::-1]
        fboxes = boxes[indices,:]

        #Visualize top scoring boxes 
        im = im[:, :, (2, 1, 0)] 
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(im)
       # pdb.set_trace()
        for i in xrange(num_boxes_vis):
            bbox = fboxes[i,:]
            score = scores[indices[i]]    
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    'Objectness:{:.3f}'.format(score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title('Fast DeepBox proposals on demo frame #{:d}'.format(frame_num),fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
