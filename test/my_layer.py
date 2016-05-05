import sys
import os

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from copy import copy

plt.rcParams['figure.figsize'] = (32, 32)

caffe_root = '/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/caffe-fast-rcnn/'
sys.path.append(caffe_root + 'python')

import caffe
from caffe import layers as L, params as P

sys.path.append('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/caffe-fast-rcnn/examples/pycaffe/layers')
sys.path.append('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/caffe-fast-rcnn/examples/pycaffe')

import tools

cifar_root = osp.join('/ltmp/gustavo-2951t/dd_cv/cifar-10-batches-py/')

cifar = np.array([''])

caffe.set_mode_gpu()
caffe.set_device(0)

# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)












