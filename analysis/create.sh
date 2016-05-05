#!/bin/bash

python build_analysis.py --gpu 0 --model  /ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_30000.caffemodel --solver /ltmp/gustavo-2951t/dd_cv/test/my_test.prototxt --action create --filepath  /ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt --diroutput /ltmp/gustavo-2951t/dd_cv/analysis/  --number_in_file 2501
