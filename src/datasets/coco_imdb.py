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


import datasets.coco_imdb
import os
import datasets.imdb_coco
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import h5py
import cPickle
import subprocess
from fast_dbox_config import cfg
import sys
sys.path.insert(0,"./data/MSCOCO/PythonAPI")
from pycocotools.coco import COCO
import pdb

class coco_imdb(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        datasets.imdb.__init__(self, 'coco_' + image_set + year)
        self._year = year
        self._image_set = image_set
        
        #_devkit_path points to MSCOCO
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        #_data_path points to MSCOCO/images
        self._data_path = os.path.join(self._devkit_path, 'images')
        self._classes = ('__background__', 'object')
                         
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        # initialize COCO api for instance annotations
        
        if image_set in ['train','val']:
            self.coco=COCO('%s/annotations/instances_%s.json'%(self._devkit_path,self._image_set+self._year))
        
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.proposals_roidb
        # Use top-k proposals to build imdb
        self.config = {'top_k':1000}

        assert os.path.exists(self._devkit_path), \
                'COCO devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Change this function to read COCO images: PATH = images/train2014/ 
        Construct an image path from the image's "index" identifier.
        self._image_set is either 'train','val','test','test-dev'
        """
        if self._image_set in ['train','val']:
            img = self.coco.loadImgs(index)
            image_path = os.path.join(self._data_path, self._name[5:],img[0]['file_name'])
            assert os.path.exists(image_path), \
                    'Path does not exist: {}'.format(image_path)
        else:
            image_path = self._data_path+'/test2015/COCO_test2015_{0:012d}.jpg'.format(int(index[0]))
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes from COCO object
        """
        #Use all images
        #Use train/val2014 COCO Matlab ordering
        imgIds = sio.loadmat('./data/coco_matlab_data/matlab_'+self._image_set+self._year+'_imgIds.mat')
        image_index = imgIds['imgIds'].tolist()
        
        #Use COCO python ordering
        #image_index = self.coco.getImgIds()
        return image_index

    def _get_default_path(self):
        """
        Return the default path where MSCOCO is expected to be installed.
        """
        return os.path.join(cfg.ROOT_DIR, 'data', 'MSCOCO')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_coco_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
        
    def proposals_roidb(self):
        """
        Return the database of Edge box regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """ 
        cache_file = os.path.join(self.cache_path,self.name + '_'+self._obj_proposer+'_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                 roidb = cPickle.load(fid)
            print '{} eb roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        if self._image_set in ['train','val']: 
            gt_roidb = self.gt_roidb()
            proposals_roidb = self._load_proposals_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, proposals_roidb)
        else:
            roidb = self._load_image_info_roidb()

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote {} roidb to {}'.format(self._obj_proposer,cache_file)

        return roidb
            
        
    def _load_image_info_roidb(self):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                self._obj_proposer+'_data',
                                                self._image_set+self._year+'.mat'))
        assert os.path.exists(filename), \
               self._obj_proposer + ' data not found at: {}'.format(filename)
        #read -v7 matrix
        #raw_data = sio.loadmat(filename)['boxes'].ravel()
        #read -v7.3 matrix
        f = h5py.File(filename)
        raw_data = [np.transpose(f[element[0]]) for element in f['boxes']] 
        #raw data is a list so use len() as follows. 
        #///////////
        box_list = []
        print 'No permutation for validation'
        for i in xrange(len(self.image_index)):#Use raw_data.shape[0] if it's not a list
            print raw_data[i].shape
            box_list.append(raw_data[i] - 1)   
        
        return self.test_roidb_from_box_list(box_list) 

    def _load_proposals_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                self._obj_proposer+'_data',
                                                self._image_set+self._year+'.mat'))
        assert os.path.exists(filename), \
               self._obj_proposer + ' data not found at: {}'.format(filename)
        #read -v7 matrix
        #raw_data = sio.loadmat(filename)['boxes'].ravel()
        #read -v7.3 matrix
        f = h5py.File(filename)
        raw_data = [np.transpose(f[element[0]]) for element in f['boxes']] 
        #raw data is a list so use len() as follows. 
        #///////////
        box_list = []
        #if self._image_set == 'train':
        if False:#Just for testing on train set
            print 'random sampling for training'
            top_k = self.config['top_k']
            for i in xrange(len(self.image_index)):#Use raw_data.shape[0] if it's not a list
                print raw_data[i].shape
                sel_num = min(top_k,raw_data[i].shape[0])
                samp_idx = np.random.permutation(raw_data[i].shape[0])
                box_list.append(raw_data[i][samp_idx[:sel_num], :] - 1)
        else:
            print 'No permutation for validation'
            for i in xrange(len(self.image_index)):#Use raw_data.shape[0] if it's not a list
                print raw_data[i].shape
                box_list.append(raw_data[i] - 1)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_coco_annotation(self, index):
        img_ann = []
        annIds = self.coco.getAnnIds(index); 
        anns = self.coco.loadAnns(annIds); 
        num_objs = len(anns); 
        boxes = np.zeros((num_objs, 4))
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        """
        Load image and bounding boxes info from COCO object
        """
        for j in range(num_objs):
            #cls = anns[j]['category_id']
            cls = 1
            boxes[j,:]=anns[j]['bbox']#coco bbox annotations are 0-based 
            boxes[j,2]=boxes[j,0]+boxes[j,2]
            boxes[j,3]=boxes[j,1]+boxes[j,3]
            gt_classes[j]=cls
            overlaps[j,cls] = 1.0
        #pdb.set_trace()        
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,'gt_classes': gt_classes,'gt_overlaps' : overlaps,'flipped' : False}

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

if __name__ == '__main__':
    d = datasets.coco_imdb('val', '2014')
    #d = datasets.coco_imdb('test-dev', '2015')
    res = d.roidb
    from IPython import embed; embed()
