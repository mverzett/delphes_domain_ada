from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys
import os
from pdb import set_trace
from sklearn.model_selection import train_test_split


def save(df, fname):
	dname = os.path.dirname(fname)
	if not os.path.isdir(dname):
		os.makedirs(dname)
	records = df.to_records(index=False)
	records.dtype.names = [str(i) for i in records.dtype.names]
	np.save(fname, records)

def run_scan(
	outdir, indir, addsv=False, 
	lr=[0.001], weight=[50], lmb=[0.04], 
	igpu=0, gpu_fraction=0.25, postfix='1'):
	#
	# TF crap preamble
	#
	import keras
	from keras import backend as K
	import sys
	import tensorflow as tf
	import os
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)
	print('running on GPU '+str(igpu))

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	K.set_session(sess)
	print('using gpu memory fraction: '+str(gpu_fraction))

	#
	# Load Data
	#
	from make_samples import make_sample
	from keras.layers import Input
	Inputs = Input((21,))
	X_traintest, isB_traintest , isMC_traintest = make_sample(indir, addsv)
	X_all, X_test, isB_all, isB_test, isMC_all, isMC_test = train_test_split(
		X_traintest, isB_traintest , isMC_traintest, 
		test_size=0.1, random_state=42
		)
	
	from keras.optimizers import Adam
	from models import modelIverseGrad
	from itertools import product
	#
	# Run scan
	#
	for ilmb, ilr, iweight in product(lmb, lr, weight):
		print 'Testing lambda: %f, learning rate: %f, weight: %f' % (ilmb, ilr, iweight)
		model = modelIverseGrad(Inputs, rev_grad=ilmb)
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer=Adam(lr=ilr), 
			loss_weights=[1., 0., iweight, iweight]
			)
		history = model.fit(
			X_all, 
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=5000, epochs=75,  verbose=0, validation_split=0.2, 
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				1-isB_all.ravel()*0.75, 
				1+isB_all.ravel()*0.75],
			)
		
		history = pd.DataFrame(history.history)
		save(history, '%s/domain_adaptation_lr%f_w%f_l%f/%s/history.npy' % (outdir, ilr, iweight, ilmb, postfix))

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('outputdir')
	parser.add_argument("inputdir")
	parser.add_argument('postfix')
	parser.add_argument("--addsv", action='store_true')
	parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
	parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=-1)
	parser.add_argument("--lr", help="learning rates")
	parser.add_argument("--weight", help="domain adaptation weights")
	parser.add_argument("--lmb", help="domain adaptation lambdas")
	args = parser.parse_args()
	str2floats = lambda x: [float(i) for i in x.split(',')]

	run_scan(
	args.outputdir, args.inputdir, 
	addsv=args.addsv, 
	lr=str2floats(args.lr), 
	weight=str2floats(args.weight), 
	lmb=str2floats(args.lmb), 
	igpu=args.gpu, gpu_fraction=args.gpufraction, 
	postfix=args.postfix)

	
