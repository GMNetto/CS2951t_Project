import sys
import caffe
sys.path.append('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/lib/')

import scipy
from fast_rcnn.config import cfg
from fast_rcnn.my_test import im_detect
import cv2

class WrapperNet:

    def initialize_net(self, prototxt, caffemodel):
        cfg.TEST.HAS_RPN = True
        caffe.set_mode_gpu()
        caffe.set_device(0)
        caffe.GPU_ID = 0
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)


    def get_output_layer(self, image_path, layer):
        im = cv2.imread(image_path)
	code = im_detect(self.net, im, layer)
	return code

def get_distance_codes(code1, code2):
    return scipy.spacial.distance.euclidean(code1, code2)
