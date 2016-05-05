#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/default_faster_rcnn.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"
GPU_ID=$1
NET=$2
NET_lc=${NET,,}
OUTPUT=$3
ITERS=70000
DATASET_TRAIN=voc_2007_trainval
DATASET_TEST=voc_2007_test

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/faster_rcnn_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#NET_INIT=/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel

#NET_INIT=/data/gen_data/gmarques/snapshots_auto/vgg16_faster_rcnn_iter_50000.caffemodel

#NET_INIT=/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel 

NET_INIT=/data/gen_data/gmarques/snapshot_end2end_no_autoencoder/vgg16_faster_rcnn_iter_70000.caffemodel

time ./../py-faster-rcnn/tools/train_net.py --gpu ${GPU_ID} \
  --solver /ltmp/gustavo-2951t/dd_cv/test/my_solver_vgg.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg ./../py-faster-rcnn/experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./../py-faster-rcnn/tools/test_net.py --gpu ${GPU_ID} \
  --def ../test/my_test_vgg.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg ../py-faster-rcnn/experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
