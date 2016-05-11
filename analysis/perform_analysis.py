import numpy as np
import argparse
from scipy import spatial
from autoencoder import Autoencoder
import sys
import xml.etree.ElementTree

sys.path.append('../py-faster-rcnn/lib/utils')

from utils.timer import Timer

sys.path.append('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/lib/')
sys.path.append('/../py-faster-rcnn/tools/')

class_to_index = {'__background__':0, 'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4, 'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9, 'cow':10, 'diningtable':11, 'dog':12, 'horse':13, 'motorbike':14, 'person':15, 'pottedplant':16, 'sheep':17, 'sofa':18, 'test':19, 'train':20, 'tvmonitor':21}

images_dir = '../py-faster-rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/'
jpg_dir = '../py-faster-rcnn/data/VOCdevkit/VOC2012/JPEGImages/'
annotations_dir = '../py-faster-rcnn/data/VOCdevkit/VOC2012/Annotations/'
my_autoencoder = None
total_retrieved_correct = 0 
total_retrieved_false = 0
total_retrieved = 0
number_features = 0
pca = None
timer = Timer()
timer_pca = Timer()
timer_total = Timer()
start_interval = 0
step_interval = 1000
end_interval = 5000
pca_dimensions = 256

def parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--codesdir', dest='codesdir', type=str,help='npy file with codes and image')
    parser.add_argument('--solver', dest='solver', type=str,help='test prototxt', default=None)
    parser.add_argument('--model', dest='model', type=str,help='caffemodel', default=None)
    parser.add_argument('--testfile', dest='testfile', type=str,help='image', default=None)
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='algorithm', default=None)
    parser.add_argument('--metric', dest='metric', type=str, help='metric', default=None)
    parser.add_argument('--nfeatures', dest='number_features', type=int, help='number of features', default=None)
    parser.add_argument('--action', dest='action', type=str, help='fc7, pca, auto', default=None)
    return parser.parse_args()


analysis = {'__background__':[0,0,0,0], 'aeroplane':[0,0,0,0], 'bicycle':[0,0,0,0], 'bird':[0,0,0,0], 'boat':[0,0,0,0], 'bottle':[0,0,0,0], 'bus':[0,0,0,0], 'car':[0,0,0,0], 'cat':[0,0,0,0], 'chair':[0,0,0,0], 'cow':[0,0,0,0], 'diningtable':[0,0,0,0], 'dog':[0,0,0,0], 'horse':[0,0,0,0], 'motorbike':[0,0,0,0], 'person':[0,0,0,0], 'pottedplant':[0,0,0,0], 'sheep':[0,0,0,0], 'sofa':[0,0,0,0], 'test':[0,0,0,0], 'train':[0,0,0,0], 'tvmonitor':[0,0,0,0]}

def compare_test_and_retrieved(test_list_classes, retrieved_classes):
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
'''Gather everything in a list and send the list to compare'''

def compare_test_and_k_retrieved(test_list_classes, list_images):
    retrieved_classes = []
    for image in list_images:
        if image == 0 or image == 0.0:
            return
        image = image.split('_')
        while len(image[1]) < 6:
            image[1] = '0' + image[1] 
        retrieved_classes = get_object_class(annotations_dir + image[0] + '_' + image[1]) + retrieved_classes
    compare_test_and_retrieved(test_list_classes, retrieved_classes)
        

def get_autoencoder(image_name):
    codes = my_autoencoder.autoencoder(jpg_dir + image_name + '.jpg') 
    binary_code = codes[:, 0:number_features] > 0.5
    binary_code = binary_code.astype(int)
    return binary_code

def get_fc7(image_name):
    return my_autoencoder.get_fc7(jpg_dir + image_name + '.jpg') 

def get_pca(image_name):
    code = my_autoencoder.get_fc7(jpg_dir + image_name + '.jpg')
    return pca.transform(code)

def train_pca(file_dir, s, t, i):
    from sklearn.decomposition import IncrementalPCA
    global timer_pca
    timer_pca = Timer()	
    timer_pca.tic()
    ipca = IncrementalPCA(n_components=pca_dimensions)
    for counter in range(s, t, i):
        features_file = np.load(file_dir + '/pca' + str(counter) + '_code.npy')
	ipca.partial_fit(features_file[:, 0:4096])
	timer_pca.toc()
    return ipca

def compare_image(image_name, tree, my_autoencoder, k_nearest, codes_image, number_features, get_feature):
    binary_code = get_feature(image_name)
    results = [0,0]
    timer.tic()
    results[0], results[1] =  tree.kneighbors(binary_code, n_neighbors=k_nearest)
    timer.toc()
    classes = get_object_class(annotations_dir + image_name)
    for index_feature in range(0, results[1].shape[0]):
	
	list_images1 = codes_image[results[1][index_feature, :].astype(int), -1].tolist()

	list_images0 = codes_image[results[1][index_feature, :].astype(int), -2].tolist()
	resulting = [str(int(a)) + '_' + str(int(b)) for a,b in zip(list_images0, list_images1)]
        compare_test_and_k_retrieved(classes, resulting)

def get_best_matches(image_name, tree, my_autoencoder, k_nearest, codes_image, number_features, get_feature):
    binary_code = get_feature(image_name)
    results = [0,0]
    timer.tic()
    results[0], results[1] =  tree.kneighbors(binary_code, n_neighbors=k_nearest)
    timer.toc()
    classes = get_object_class(annotations_dir + image_name)
    histogram_knn = {}
    for index_feature in range(0, results[1].shape[0]):
	
	list_images1 = codes_image[results[1][index_feature, :].astype(int), -1].tolist()

	list_images0 = codes_image[results[1][index_feature, :].astype(int), -2].tolist()
	resulting = [str(int(a)) + '_' + str(int(b)) for a,b in zip(list_images0, list_images1)]
	for result in resulting:
            histogram_knn.setdefault(result, 0)
	    histogram_knn[result] += 1
    import operator
    return sorted(histogram_knn.items(), key=operator.itemgetter(1))

def get_codes_from_parameters(codesdir, s, t, i, total, number_features, prefix):
    codes = np.zeros((total, number_features + 6))
    last_row = 0
    for counter in range(s, t, i):
        features_file = np.load(codesdir + prefix + str(counter) + '_code.npy')
	new_dimension = last_row + features_file.shape[0]
	codes[last_row:new_dimension, :] = features_file
	last_row = new_dimension
    return codes

def get_total_number_features(codesdir, s, t, i, prefix):
    total = 0
    for counter in range(s, t, i):
        features_file = np.load(codesdir + prefix + str(counter) + '_code.npy')
	total = features_file.shape[0] + total
    return total

def histogram_dataset(codesdir, s, t, i, prefix):
    histogram = {}
    total_images = 0
    for counter in range(s, t, i):
	features_file = np.load(codesdir + prefix + str(counter) + '_code.npy')
	list_images1 = features_file[:, -1].tolist()
	list_images0 = features_file[:, -2].tolist()
	total_images += len(list_images1)
        for i in range(0, len(list_images0)):
            year = str(int(list_images0[i]))
	    image_number = str(int(list_images1[i]))
	    while len(image_number) < 6:
                image_number = '0' + image_number
	    retrieved_classes = get_object_class(annotations_dir + year + '_' + image_number)
	    for object_class in retrieved_classes:
	        histogram[object_class] = histogram.setdefault(object_class, 0) + 1
    histogram_file = open("histogram.txt", 'w')
    for object_class in histogram.keys():
	histogram_file.write(object_class + ' ' + str(histogram[object_class]) + '\n')
    histogram_file.close()
    print total_images


def create_neighbors_structure(codes, size_feature, metric_name, algorithm_name):
    from sklearn.neighbors import NearestNeighbors
    neigh =  NearestNeighbors(1, metric=metric_name, algorithm=algorithm_name)
    codes = codes.astype(int)
    neigh = neigh.fit(codes[:, 0:size_feature])
    return neigh, codes

def get_object_class(image_file):
    annotation_tree = xml.etree.ElementTree.parse(image_file + '.xml').getroot()
    annotation_list = []
    year = image_file[-11:-7]
    if year == "2008":
        for i in range(5, len(annotation_tree)):
            annotation_list.append(annotation_tree[i][0].text)
    else:
        for i in range(2, len(annotation_tree)):
            if len(annotation_tree[i]) == 0:
                 break
            annotation_list.append(annotation_tree[i][0].text)
		
    return annotation_list


def compare_classes(annotations, test):
    number_same = 0
    number_total = len(annotations)
    for annotation in annotations:
	if test[class_to_index[annotation]] == 1:
            number_same = number_same + 1
    return number_same, number_total
    
def test_general(test_codes_file, model, solver, codes_dir, algorithm, metric, number_features, get_features, prefix):
    for i in [1, 5, 10, 20]:
        global timer
        timer = Timer()
        k = i
        total = get_total_number_features(codes_dir, start_interval, end_interval, step_interval, prefix)
        print total
        neigh, codes_image = create_neighbors_structure(get_codes_from_parameters(codes_dir, start_interval, end_interval, step_interval, total, number_features, prefix), number_features, metric, algorithm)
        global my_autoencoder
        my_autoencoder = Autoencoder(solver, model)
        counter = 1
        with open(images_dir + test_codes_file) as f:
            for image in f:
                print counter
                counter = counter + 1 

                compare_image(image.rstrip(), neigh, my_autoencoder, k, codes_image, number_features, get_features)
	        #if counter == 4:
	        #    break
        persist_test('res_' + prefix + str(k) +'.txt')

def get_best_matches_images(test_codes_file, model, solver, codes_dir, algorithm, metric, number_features, get_features, prefix):
    global timer
    timer = Timer()
    k = 10
    total = get_total_number_features(codes_dir, start_interval, end_interval, step_interval, prefix)
    print total
    neigh, codes_image = create_neighbors_structure(get_codes_from_parameters(codes_dir, start_interval, end_interval, step_interval, total, number_features, prefix), number_features, metric, algorithm)
    global my_autoencoder
    my_autoencoder = Autoencoder(solver, model)
    counter = 1
    with open(test_codes_file) as f:
        for image in f:
            print counter
            counter = counter + 1 

            dictionary = get_best_matches(image.rstrip(), neigh, my_autoencoder, k, codes_image, number_features, get_features)
	        #if counter == 4:
	        #    break
            print "Dictionary for image #",counter
	    for i in range(len(dictionary)-1, 0, -1):
                if i == len(dictionary) - 10:
                      break
	        #print i
	        key = dictionary[i]
	        print key[0], "  ", key[1] 

def persist_test(file_name):
    file_r =  open(file_name, 'w')
    file_r.write(str(total_retrieved_correct)+'\n')
    file_r.write(str(total_retrieved_false)+'\n')
    file_r.write(str(total_retrieved)+'\n')
    for key in analysis:
        file_r.write(key+"\n")
        results = analysis[key]
        file_r.write(str(results[0]) + ' ' + str(results[1]) + ' ' + str(results[2]) + '\n')
        if results[0] != 0:
            file_r.write(str(float(results[1])/results[0]) + ' ' + str(float(results[2])/results[0]) + '\n')
        else:
            file_r.write("ZERO\n");
        file_r.write('\n')
	file_r.write(str(timer.average_time))
    file_r.close()



if __name__=="__main__":
    args = parser()
    global number_features
    number_features = args.number_features
    global timer_total
    timer_total = Timer()
    timer_total.tic()
    if args.action == 'fc7':
        test_general(args.testfile, args.model, args.solver, args.codesdir, args.algorithm, args.metric, args.number_features, get_fc7, 'pca')
    elif args.action == 'auto':
        test_general(args.testfile, args.model, args.solver, args.codesdir, args.algorithm, args.metric, args.number_features, get_autoencoder, 'auto')
    elif args.action == 'pca':
        global pca
        pca = train_pca(args.codesdir, start_interval, end_interval, step_interval)
        test_general(args.testfile, args.model, args.solver, args.codesdir, args.algorithm, args.metric, args.number_features, get_pca, 'pca_red_')
    elif args.action == 'getknn':
	get_best_matches_images(args.testfile, args.model, args.solver, args.codesdir, args.algorithm, args.metric, args.number_features, get_autoencoder, 'auto')
    elif args.action == 'histogram':
        histogram_dataset(args.codesdir, start_interval, end_interval, step_interval, 'auto')
   	timer_total.toc()
    print 'Average time search elements: ', timer.average_time
    print 'Average time building pca: ', timer_pca.average_time
    print 'Total time: ', timer_total.average_time


    #print 'results: ', len(result), result[1], result[0], codes_image[result[1].astype(int), -1]
