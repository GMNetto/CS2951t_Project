import numpy as np
import argparse
from scipy import spatial
from autoencoder import Autoencoder
import sys
import xml.etree.ElementTree

sys.path.append('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/lib/')
sys.path.append('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/tools/')

images_dir = '/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/'

jpg_dir =  '/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/'

annotations_dir = '/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations/'

total_retrieved_correct = 0 
total_retrieved_false = 0
total_retrieved = 0


def parser():
    parser = argparse.ArgumentParser(description='Get PCA results')
    parser.add_argument('--codesdir' ,dest='codesdir', type=str, help='dir with pca features')
    parser.add_argument('--solver' ,dest='solver', type=str, help='dir with solver features')
    parser.add_argument('--model' ,dest='model', type=str, help='dir with model features')
    parser.add_argument('--testfile' ,dest='testfile', type=str, help='file with image test')
    return parser.parse_args()


analysis = {'__background__':[0,0,0,0], 'aeroplane':[0,0,0,0], 'bicycle':[0,0,0,0], 'bird':[0,0,0,0], 'boat':[0,0,0,0], 'bottle':[0,0,0,0], 'bus':[0,0,0,0], 'car':[0,0,0,0], 'cat':[0,0,0,0], 'chair':[0,0,0,0], 'cow':[0,0,0,0], 'diningtable':[0,0,0,0], 'dog':[0,0,0,0], 'horse':[0,0,0,0], 'motorbike':[0,0,0,0], 'person':[0,0,0,0], 'pottedplant':[0,0,0,0], 'sheep':[0,0,0,0], 'sofa':[0,0,0,0], 'test':[0,0,0,0], 'train':[0,0,0,0], 'tvmonitor':[0,0,0,0]}

def compare_test_and_retrieved(test_list_classes, image):
    if image == 0 or image == 0.0:
        return
    image = str(image)
    while len(image) < 6:
        image = '0' + image 
    retrieved_classes = get_object_class(annotations_dir + image)
    positive = 0
    for test_class in test_list_classes:
	if test_class in retrieved_classes:
	    positive = 1
	    analysis[test_class][1] += 1
	else:
	    analysis[test_class][2] += 1
	analysis[test_class][0] += 1
    global total_retrieved_correct
    global total_retrieved_false
    global total_retrieved
    total_retrieved += 1
    total_retrieved_correct += positive
    total_retrieved_false += abs(positive - 1)

def compare_test_and_k_retrieved(test_list_classes, list_images):
    for image in list_images:
	compare_test_and_retrieved(test_list_classes, str(int(image)))

def compare_image(image_name, tree, my_autoencoder, k_nearest, pca, codes_image):
    codes = my_autoencoder.get_fc7(jpg_dir + image_name + '.jpg') 
    #binary_code = codes[0:128] > 0.5
    #binary_code = binary_code.astype(int)
    codes = pca.transform(codes)
    results =  tree.query(codes, k=k_nearest, p=2)
    classes = get_object_class(annotations_dir + image_name)
    for result in results:
	list_images = codes_image[result[1].astype(int), -1]    
	print 'code', list_images
	if list_images != 0.0:
	    if isinstance(result, list):
                compare_test_and_k_retrieved(classes, list_images.tolist())
	    else:
	        compare_test_and_retrieved(classes, str(int(list_images)))


def create_structures(codes):
    tree = spatial.cKDTree(codes[:, 0:48])
    return tree, codes

def get_code_from_files(files_dir, s, t, i, total):
    print s, t, i, total
    codes = np.zeros((total, 53))
    for counter in range(s, t, i):
        features_file = np.load(files_dir + 'pca_red_' + str(counter) + '.npy')
	new_dimension = counter + features_file.shape[0]
	print features_file.shape[0]
	codes[counter:new_dimension, :] = features_file
    return codes

def get_object_class(image_file):
    print 'class', image_file
    annotation_tree = xml.etree.ElementTree.parse(image_file + '.xml')
    annotation_list = []
    annotation = annotation_tree.getroot()
    for i in range(6, len(annotation)):
	print annotation[i][0].text    
        annotation_list.append(annotation[i][0].text)
    return annotation_list


def get_pca(file_dir, s, t, i):
    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=48)
    for counter in range(s, t, i):
        features_file = np.load(file_dir + '/pca' + str(counter) + '_code.npy')
        ipca.partial_fit(features_file[:, 0:4096])
    return ipca

if __name__ == "__main__":
    args = parser()
    tree, codes_image = create_structures(get_code_from_files(args.codesdir, 0, 1000, 1000, 4975))
    my_autoencoder = Autoencoder(args.solver, args.model)
    code = my_autoencoder.get_fc7(jpg_dir + '009961' + '.jpg')
    pca = get_pca(args.codesdir, 0, 1000, 1000)
    with open(images_dir + 'test.txt') as f:
        for image in f:  
	    print 'new image', image
	    #compare_image(image.rstrip(), tree, my_autoencoder, 1, pca, codes_image)
    #print 'ok'
    code_red = pca.transform(code)
   # print code_red.shape
    #print 
    result = tree.query(code_red, k=1, p=2)
    print 'results: ', len(result), result[1], result[0], codes_image[result[1].astype(int), -1]
    print total_retrieved_correct, total_retrieved
    print total_retrieved_false
    print total_retrieved
    print analysis


