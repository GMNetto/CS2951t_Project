set -x
set -e

export PYTHONUNBUFFERED="True"
GPU_ID=$1
NET=$2
MODEL_PATH=$3
NET_lc=${NET,,}
ITERS=20000
DATASET_TRAIN=voc_2007_trainval
DATASET_TEST=voc_2007_test

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="../py-faster-rcnn/experiments/logs/faster_rcnn_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#MODEL_PATH='/data/gen_data/gmarques/snapshot_end2end_no_autoencoder/vgg16_faster_rcnn_iter_10000.caffemodel'
#MODEL_PATH='../py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_50000.caffemodel' 
#MODEL_PATH='/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/vgg16_faster_rcnn_iter_5000.caffemodel' 
time ./../py-faster-rcnn/tools/test_net.py --gpu ${GPU_ID} \
  --def ${NET} \
  --net ${MODEL_PATH} \
  --imdb ${DATASET_TEST} \
  --cfg '../py-faster-rcnn/experiments/cfgs/faster_rcnn_end2end.yml' \
  ${EXTRA_ARGS}
