# DeepBox: Learning Objectness with Convolutional Networks

Created by Weicheng Kuo at UC Berkeley

### Introduction

Fast DeepBox is a bounding box proposal re-ranker using ConvNets. It produces state-of-the-art bounding box proposal within 0.5s using a light-weight 4-layer network. Experiments on both PASCAL and COCO showed that DeepBox performs significantly better than Edge boxes in terms of Area under Curve and that the gain carries over to detection mAP. This implementation is based on Ross's Fast-RCNN codebase, thereby written in Python and C++/Caffe.

DeepBox was initially described in an [arXiv paper](http://arxiv.org/abs/1505.02146) and later published at ICCV 2015.

### License

Fast DeepBox is released under the MIT License (refer to the LICENSE file for details).

### Citing Fast DeepBox

If you find Fast DeepBox useful in your research, please consider citing:

    @inproceedings{KuoICCV15DeepBox,
        Author = {Weicheng Kuo, Bharath Hariharan, Jitendra Malik},
        Title = {DeepBox:Learning Objectness with Convolutional Networks},
        Booktitle = {International Conference on Computer Vision ({ICCV})},
        Year = {2015}
    }
    
### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)
7. [Extra downloads](#extra-downloads)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  *Note:* Caffe doesn't have to be built with support for Python layers!

  You can download my [Makefile.config](http://www.cs.berkeley.edu/~wckuo/fast-dbox-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. MATLAB (required for running COCO evaluation)

### Requirements: hardware

1. A good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices

### Installation (sufficient for the demo)

1. Clone the Fast DeepBox repository
  ```Shell
  git clone https://github.com/weichengkuo/fast-dbox.git
  ```
  
2. We'll call the directory that you cloned Fast DeepBox into `FDBOX_ROOT`

3. Build the Cython modules
    ```Shell
    cd $FDBOX_ROOT/src
    make
    ```
    
4. Build Caffe and pycaffe
    ```Shell
    cd $FDBOX_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```
    
5. Download pre-computed Fast DeepBox models
    ```Shell
    cd $FDBOX_ROOT/output/default
    ./scripts/fetch_fast_dbox_models.sh
    ```

    This will populate the `$FDBOX_ROOT/output/default` folder with `coco_train2014`,which contains a variety of models. These include multiscale/single-scale Fast DeepBox models and a multiscale sliding window model for training comparison.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

**Python**

To run the demo
```Shell
cd $FDBOX_ROOT
python ./tools/demo.py
```
By default, visualization only shows top five proposals per image and their scores. Users can set the number of proposals they want to visualize by passing in --numboxes [Number of proposals] argument. There is another demo mode on full COCO dataset that user can choose by passing in --demo 0 argument. This allows users to visualize Fast DeepBox proposals on any frame of users' choice in COCO validation set, but requires users to download pre-computed Edge boxes proposals and install COCO dataset in full. 

### Beyond the basic demo: installation for full COCO demo, training and testing models
Here are the steps to set up full functionalities of DeeBox package.

1. Download pre-computed Edge boxes proposals
    ```Shell
    cd $FDBOX_ROOT/data
    ./scripts/fetch_edge_box_data.sh
    ```
    The Edge boxes proposals are pre-computed in order to reduce installation requirements. This step is necessary for demo.

2. Download COCO Matlab data 
    ```Shell
    cd $FDBOX_ROOT/data
    ./scripts/fetch_coco_matlab_data.sh
    ```
    This step downloads the Matlab COCO image ordering with which Edge boxes proposals are computed. Ground truth boxes would also be downloaded to enable evaluation.

3. Set up Microsoft COCO directory by
    ```Shell
    cd $FDBOX_ROOT/data
    ln -s PATH/TO/YOUR/COCO ./MSCOCO
    ```
    If you haven't installed COCO on your machine yet, you can follow the instructions on the github page below to download and compile all the data. 
   -[MSCOCO](https://github.com/pdollar/coco)

4.  Download pre-trained ImageNet models

    Pretrained ImageNet model for Alex net can be downloaded to initialize the DeepBox network training.
    ```Shell
    cd $FDBOX_ROOT
    ./data/scripts/fetch_imagenet_models.sh
    ```
    Alternatively, users can initialize the training with our multiscale sliding window model in
    ```Shell
    $FDBOX_ROOT/output/default/coco_train2014/fast-dbox-slidwindow-multiscale.caffemodel
    ```

### Usage

**Train** a Fast DeepBox proposer:

```Shell
python ./tools/train_net.py (Show all training options)
python ./tools/train_net.py --gpu 0 (Train on GPU 0)
```
Train output is written underneath `$FDBOX_ROOT/output/default/coco_train2014`.


**Test** a Fast DeepBox proposer:

```Shell
python ./tools/test_net.py (Show all testing options)
python ./tools/test_net.py --gpu 0 (Test on GPU 0)
```

Test output is written underneath `$FDBOX_ROOT/output/default/coco_val2014`.

**Full Demo** a pre-trained Fast DeepBox proposer on COCO val set

```Shell
python ./tools/demo.py (Show all demo options)
python ./tools/demo.py --demo 0 (full demo)
