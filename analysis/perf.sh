#!/bin/bash

python perform_analysis.py --codesdir /data/gen_data/gmarques/pool_auto_256_12/ --solver ../test/my_test_vgg.prototxt --model /data/gen_data/gmarques/snapshots_auto_256/vgg16_faster_rcnn_iter_70000.caffemodel --testfile val_random.txt --algorithm ball_tree --metric manhattan --nfeatures 256 --action auto

