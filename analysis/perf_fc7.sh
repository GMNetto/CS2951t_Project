#!/bin/bash

python perform_analysis.py --codesdir /data/gen_data/gmarques/pool_fc7_12/ --solver /ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/models/VGG16/faster_rcnn_end2end/test.prototxt --model /data/gen_data/gmarques/snapshot_end2end_no_autoencoder/vgg16_faster_rcnn_iter_70000.caffemodel --testfile val_random.txt --algorithm brute --metric euclidean --nfeatures 4096 --action fc7

