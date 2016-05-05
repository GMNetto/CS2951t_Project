#!/bin/bash

source ../bin/activate


export PYTHONPATH=$PYTHONPATH:/ltmp/gustavo-2951t/dd_cv/local/lib/python2.7/site-packages:/usr/local/cuda-7.0/targets/x86_64-linux/lib/


export LD_LIBRARY_PATH=/usr/local/cuda-7.0/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHONPATH:/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/caffe-fast-rcnn/python
