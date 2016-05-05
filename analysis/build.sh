#!/bin/bash

#python build_analysis.py --gpu 0 --action pca --filepath ../py-faster-rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/train_random.txt --diroutput /data/gen_data/gmarques/ --number_in_file 5000 --model /data/gen_data/gmarques/snapshots_auto_256/vgg16_faster_rcnn_iter_70000.caffemodel --solver ../test/my_test_vgg.prototxt
#../py-faster-rcnn/models/VGG16/faster_rcnn_end2end/test.prototxt
python build_analysis.py --gpu 0 --action pca_pred --filepath /data/gen_data/gmarques/pool_fc7_12 --diroutput /data/gen_data/gmarques/ --number_in_file 5000 --model /data/gen_data/gmarques/snapshots_auto_256/vgg16_faster_rcnn_iter_70000.caffemodel --solver ../test/my_test_vgg.prototxt

