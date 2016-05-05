import sys

import caffe

sys.path.append('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/lib/')

from fast_rcnn.config import cfg
from fast_rcnn.my_test import im_detect
from fast_rcnn.test import im_detect as get_scores_boxes
import cv2
import numpy as np

class Autoencoder:

    def __init__(self, prototxt, caffemodel):
        cfg.TEST.HAS_RPN = True
	caffe.set_mode_gpu()
	caffe.set_device(0)
	caffe.GPU_ID = 0
	self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    def autoencoder(self, image_path):
	print image_path
	im = cv2.imread(image_path)
	code = im_detect(self.net, im, 'fc8_gustavo_encode')
	return code

    def get_scores(self, imagepath):
        im = cv2.imread(imagepath)
	scores, boxes = get_scores_boxes(self.net, im)
	return scores, boxes

    def get_fc7(self, imagepath):
	im = cv2.imread(imagepath)
	code = im_detect(self.net, im, 'fc7')
	#return np.ones((300, 4096))
	return code
