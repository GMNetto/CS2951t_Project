"""Create files relating image names and binary codes"""

import argparse
import pprint
import numpy as np
import sys
import ntpath
from tempfile import mkstemp
from shutil import move
from os import remove, close
from autoencoder import Autoencoder
import numpy as np

sys.path.append('../py-faster-rcnn/lib/utils')

from utils.timer import Timer


sys.path.append('../py-faster-rcnn/lib/')
sys.path.append('../py-faster-rcnn/tools/')

from fast_rcnn.nms_wrapper import nms

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--model', dest='model',
		        help='caffemodel', default=None,
			type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--action', dest='action',
                        help='create or update',
                        default=None, type=str)
    parser.add_argument('--filepath', dest='inputfile',
                        help='Input file(to get images or update)',
                        default=None, type=str)
    parser.add_argument('--diroutput', dest='diroutput',
                        help='dir to save files',
                        default=None, type=str)
    parser.add_argument('--number_in_file', dest='num',
		       help='number images in file', default=0, type=int)
    return parser.parse_args();


#images_dir = '/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/'
images_dir = '../py-faster-rcnn/data/VOCdevkit/VOC2012/JPEGImages/'

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

#classes_dir = "/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/"
classes_dir = "../py-faster-rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/"

number_features = 256

number_dim_pca = 256

number_best_features = 3

start_interval = 0
end_interval = 5000
step_interval = 1000

def get_one_dimension(code):
    return np.average(code, axis=0)

def get_binary(code):
    return code > 0.5

def filter_code3(scores, boxes):
    #Confidence for fc7 is 0.8 0.3 for nms, and 0.6 in auto
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    list_features = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
	cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
	cls_scores = scores[:, cls_ind]
	dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
	keep = nms(dets, NMS_THRESH)
	dets = dets[keep, :]
	inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
	for index in inds:
	    list_features.append((keep[index], cls_ind))
    return list_features

def attach_image(code, list_features, boxes):
    codes = np.zeros(0)
    for item in list_features:
	ind = item[0]
	cat = item[1]
        codes_aux = np.array([np.append(code[ind,: ], boxes[ind, 4*cat: 4*(cat + 1)])])
	if codes.shape[0] != 0:
	    codes = np.append(codes, codes_aux, axis=0)
	else:
	    codes = np.copy(codes_aux)
    return codes

def get_lines_file(file_open, num_lines, name, dir_output, get_features, transform_code, type_dir):
    output_file = dir_output + type_dir + str(name) + '_code'
    counter_codes = 0
    codes = np.zeros(0)
    for counter in range(0, num_lines):
        line = file_open.readline()
	print line
	code = get_features(images_dir + line[0: -1] + '.jpg')
	scores, boxes = my_autoencoder.get_scores(images_dir + line[0: -1] + '.jpg')
        code = attach_image(transform_code(code), filter_code3(scores, boxes), boxes)
	print code.shape
	name = np.zeros((code.shape[0], 2))
	name[:, 0] = int(line[0: -1].split('_')[0])
	name[:, 1] = int(line[0: -1].split('_')[1])
	if code.shape[0] != 0:
	    code = np.append(code, name, axis=1)
	    if codes.shape[0] == 0:
	        codes = np.copy(code)
	    else:
	        codes = np.append(codes, code, axis=0)

    np.save(output_file, codes)
    return output_file, codes.shape[0]
    
def create_pool_pca2(file_path, type_dir, dir_output, num, get_features, transform_code):
    out_file = dir_output + type_dir + '_code'
    list_files = []
    last_size = 0
    number_lines = 0
    images_per_file = step_interval
    number_aux = int(np.ceil(num/images_per_file))
    with open(file_path) as images_file:
        for i in range(0, number_aux):
	    if (num - i*images_per_file) < images_per_file:
                number_each_item = num - i*images_per_file
            else:
	        number_each_item = images_per_file
	    file_name, last_size  = get_lines_file(images_file, number_each_item, i*images_per_file, dir_output, get_features, transform_code, type_dir)
	    number_lines = number_lines + last_size
            list_files.append(file_name)
	    print number_each_item
    print number_aux, 

def create_pool_pca_from_files(file_dir, dir_output, s, t, i):
    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=number_dim_pca)
    for counter in range(s, t, i):
        features_file = np.load(file_dir + '/pca' + str(counter) + '_code.npy')
	ipca.partial_fit(features_file[:, 0:4096])
    for counter in range(s, t, i):
        out_file = dir_output + 'pca_red_' + str(counter) + '_code.npy'
	features_file = np.load(file_dir + '/pca' + str(counter) + '_code.npy') 
	features_red = ipca.transform(features_file[:, 0:4096])
	np.save(out_file, np.append(features_red, features_file[:, 4096:], axis=1))

def create_pool(file_path, type_dir, dir_output, num):
    out_file = dir_output + type_dir + '_code'
    codes = np.zeros((num, number_features + 1))
    counter = 0
    with open(file_path) as images_file:
	for line in images_file:
	    code = my_autoencoder.autoencoder(images_dir + line[0:-1] + '.jpg')
	    print code.shape
	    binary_code = get_one_dimension(code)
	    print binary_code.shape
	    binary_code = binary_code > 0.5
	    codes[counter, 0:number_features] = binary_code
	    codes[counter, number_features] = int(line[0:-1])
    np.save(out_file, codes)


def update_pool(code_file):
    temp, absolute_path = mkstemp()
    with open(absolute_path, 'w') as new_file:
        with open(code_file) as old_file:
	    for line in old_file:
		image, code = line.split()
		new_code = my_autoencoder.autoencoder(images_dir + image + '.jpg')
	        new_file.write(image + ' ' + new_code)
    close(fh)
    remove(code_file)
    move(absolute_path, code_file)


my_autoencoder = None
if __name__ == '__main__':
    
    _t = Timer()
    _t.tic()
    args = parse_args()
    my_autoencoder = Autoencoder(args.solver, args.model)
    if args.action == 'create':
	#create_pool(args.inputfile, 'train', args.diroutput, args.num)
	    create_pool_pca2(args.inputfile, 'train', args.diroutput, args.num, my_autoencoder.autoencoder, get_binary)
    elif args.action == 'update':
        update_pool(args.inputfile)
    elif args.action == 'pca':
	    create_pool_pca2(args.inputfile, 'pca', args.diroutput, args.num, my_autoencoder.get_fc7, lambda x:x)
    elif args.action == 'pca_pred':
	    create_pool_pca_from_files(args.inputfile, args.diroutput, start_interval, end_interval, step_interval)
    else:
	    print 'invalid'
    _t.toc()
    print 'Average time: ', _t.average_time
