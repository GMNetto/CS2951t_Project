#
###

# import some modules
import sys, os, time
import os.path as osp

# TODO: change this to your own installation
#caffe_root = '/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/caffe-fast-rcnn/'
#sys.path.append(caffe_root+'python')
sys.path.append("/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/lib/")
sys.path.append("/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/tools/")

from sklearn.externals import joblib
import argparse
import caffe
import numpy as np

import print_funcs
import roi_data_layer.layer
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from fast_rcnn.config import cfg_from_file
from datasets.factory import get_imdb
import datasets.imdb
from fast_rcnn.train import get_training_roidb

caffe.set_mode_gpu()
caffe.set_device(0)


def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
	print 'Loaded dataset `{:s}` for training'.format(imdb.name)
	imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
	print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
	roidb = get_training_roidb(imdb)
	return roidb
    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
	    imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
	return imdb, roidb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pull out weights from snapshots of network")
    parser.add_argument("-s", "--start", help="starting snapshot iteration number", type=int)
    parser.add_argument("-t", "--stop", help="stopping snapshot iteration number", type=int)
    parser.add_argument("-i", "--iter", help="iteration step size", type=int)
    parser.add_argument("--snapshot_dir", help="caffe model params file", type=str)
    parser.add_argument("--save_dir", help="location to save weight files", type=str)
    args = parser.parse_args()
    os.chdir(args.snapshot_dir)
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Objects for logging solver training
    _train_loss = []
    _weight_params = {}

    start = args.start
    stop = args.stop
    timestr = 'snaps_{0}_{1}'.format(start, stop)
    
    cfg_from_file('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/experiments/cfgs/faster_rcnn_end2end.yml')

    if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
        # RPN can only use precomputed normalization because there are n
	# fixed statistics to compute a priori
        assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED


    import numpy as np
    import matplotlib.pyplot as plt

    imdb, roidb = combined_roidb('voc_2007_test')
    for itt in range(start, stop+1, args.iter):
        # TODO: change this to the name of your default solver file and shapshot file
	
	if(cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
	    assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED
	if(cfg.TRAIN.BBOX_REG):
	    bbox_means, bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
	    print 'done'
        

	#solver = caffe.SGDSolver(osp.join('/ltmp/gustavo-2951t/dd_cv/test/', 'my_solver.prototxt'))
	solver = caffe.SGDSolver(osp.join('/ltmp/gustavo-2951t/dd_cv/py-faster-rcnn/models/VGG16/faster_rcnn_end2end/', 'solver.prototxt'))
	solver.net.layers[0].set_roidb(roidb)

        solver.restore(osp.join(args.snapshot_dir, 'vgg16_faster_rcnn_iter_{}.solverstate'.format(itt)))
        solver.net.forward()
        _train_loss.append(solver.net.blobs['loss_cls'].data) # this should be output from loss layer
        print_funcs.print_layer_params(solver, _weight_params)
        print '******************************************************** Loss train for iter {0}: {1}'.format(itt, _train_loss)
        joblib.dump(_weight_params, osp.join(args.save_dir, 'network_parameters_%s.jbl' % timestr), compress=6)
        joblib.dump(_train_loss, osp.join(args.save_dir, 'network_loss_%s.jbl'% timestr), compress=6)


        del solver
