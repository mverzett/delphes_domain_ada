from argparse import ArgumentParser
import keras
from keras import backend as K
import sys
import tensorflow as tf
import os
from pdb import set_trace
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument('outputDir')
parser.add_argument(
	'method', choices = [
		'domain_adaptation_two_samples',
		'MC_training',
		'data_training',
		'domain_adaptation_one_sample',
		'domain_adaptation_one_sample_lambdap5',
		'domain_adaptation_two_samples_w50_l.25',
		'domain_adaptation_two_samples_w50_l.04',
		'domain_adaptation_two_samples_w25_l.5',
		'domain_adaptation_two_samples_w05_l1',
	]
)
parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=-1)
args = parser.parse_args()

loss_weigth = 50
lambda_reversal = .1
if args.method.startswith('domain_adaptation_two_samples_'):
	cfg = args.method[len('domain_adaptation_two_samples_'):]
	winfo, linfo = tuple(cfg.split('_'))
	loss_weigth = float(winfo[1:])
	lambda_reversal = float(linfo[1:])
	args.method = 'domain_adaptation_two_samples'

if args.gpu<0:
	import imp
	try:
		imp.find_module('setGPU')
		import setGPU
	except ImportError:
		found = False
else:
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	print('running on GPU '+str(args.gpu))

if args.gpufraction>0 and args.gpufraction<1:
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	K.set_session(sess)
	print('using gpu memory fraction: '+str(args.gpufraction))

from keras.engine import Layer
from Layers import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from make_samples import make_sample
#from DL4Jets.DeepJet.modules.Losses import  weighted_loss
from keras.layers import Dense, Concatenate ,Dropout
from keras.layers import Input
from keras.models import Model
import pandas as pd

def schedule(x):
	lr=0.001
	if x>75: lr=0.0001
	if x>125: lr=0.00001
	
	return lr

learning_rate = keras.callbacks.LearningRateScheduler(schedule)

def save(df, fname):
	dname = os.path.dirname(fname)
	if not os.path.isdir(dname):
		os.makedirs(dname)
	records = df.to_records(index=False)
	records.dtype.names = [str(i) for i in records.dtype.names]
	np.save(fname, records)

def modelIverseGrad(Inputs, rev_grad=.1):
	X = Dense(20, activation='relu',input_shape=(21,)) (Inputs)
	#X = Dropout(0.25)(X)
	X = Dense(10, activation='relu')(X)
	#X = Dropout(0.25)(X)
	X = Dense(10, activation='relu')(X)
	#X = Dropout(0.25)(X)
	X = Dense(10, activation='relu')(X)
	#X = Dropout(0.25)(X)
	Xa = Dense(10, activation='relu')(X)
	X = Dense(10, activation='relu')(Xa)
	X = Dense(1, activation='sigmoid', name = 'mc')(X)
	X1= Dense(1, activation='linear',use_bias=False, trainable=False,kernel_initializer='Ones', name = 'data') (X)
	Ad = GradientReversal(hp_lambda=rev_grad)(Xa)
	Ad = Dense(10, activation='relu')(Ad)
	Ad = Dense(10, activation='relu')(Ad)
	Ad = Dense(10, activation='relu')(Ad)
	Ad = Dense(10, activation='relu')(Ad)
	Ad = Dense(1, activation='sigmoid', name = 'Add' )(Ad)
	Ad1 = GradientReversal(hp_lambda=rev_grad)(Xa)
	Ad1 = Dense(10, activation='relu')(Ad1)
	Ad1 = Dense(10, activation='relu')(Ad1)
	Ad1 = Dense(10, activation='relu')(Ad1)
	Ad1 = Dense(10, activation='relu')(Ad1)
	Ad1 = Dense(1, activation='sigmoid', name = 'Add_1' )(Ad1)
	predictions = [X,X1,Ad,Ad1]
	model = Model(inputs=Inputs, outputs=predictions)
	return model



from keras.models import load_model


def run_model(outdir, Grad=1, known = 1,AdversOn=1,diffOn = 1):
	Inputs = Input((21,))
	global_loss_list={}
	global_loss_list['GradientReversal']=GradientReversal()
	X_traintest, isB_traintest , isMC_traintest = make_sample('/eos/cms/store/user/amartell/Pheno/converted/numpy_allx5')
	X_all, X_test, isB_all, isB_test, isMC_all, isMC_test = train_test_split(X_traintest, isB_traintest , isMC_traintest, test_size=0.1, random_state=42)
	advers_weight = 25.
	if AdversOn==0:
		advers_weight = 0.
	
	model = modelIverseGrad(Inputs)
	
	# gradiant loss
	if(Grad == 'domain_adaptation_two_samples'):
		model = modelIverseGrad(Inputs,rev_grad=lambda_reversal)
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer='adam', 
			loss_weights=[1., 0., loss_weigth, loss_weigth]
		)
		history = model.fit(
			X_all, 
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=5000, epochs=50,  verbose=1, validation_split=0.2, 
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				1-isB_all.ravel()*0.75, 
				1+isB_all.ravel()*0.75],
			callbacks = [learning_rate]
		)
	elif(Grad == 'MC_training'):
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer='adam', 
			loss_weights=[1.,0.,0.,0.]
		)
		history = model.fit(
			X_all,
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=5000, epochs=50, verbose=1, validation_split=0.2,
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				1+0.5*isB_all.ravel(), 
				1-0.5*isB_all.ravel()],
			callbacks = [learning_rate]
		)
	elif(Grad == 'data_training'):
		model.compile(
			loss=['binary_crossentropy']*4,
			optimizer='adam', 
			loss_weights=[0.,1.,0.,0.]
		)
		history = model.fit(
			X_all,
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=5000, epochs=50, verbose=1, validation_split=0.2,
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				1+0.5*isB_all.ravel(), 
				1-0.5*isB_all.ravel()],
			callbacks = [learning_rate]
		)
	elif(Grad == 'domain_adaptation_one_sample'):
		model = modelIverseGrad(Inputs,rev_grad=.25)
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer='adam', 
			loss_weights=[1.,0.,50.,50.]
		)
		history = model.fit(
			X_all, 
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=5000, epochs=50, verbose=1, validation_split=0.2, 
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				np.ones(isB_all.ravel().shape[0]),
				np.ones(isB_all.ravel().shape[0])],				
			callbacks = [learning_rate]
		)
	elif(Grad == 'domain_adaptation_one_sample_lambdap5'):
		model = modelIverseGrad(Inputs,rev_grad=.5)
		model.compile(
			loss = ['binary_crossentropy']*4,
			optimizer = 'adam', 
			loss_weights = [1.,0.,50.,50.]
		)
		history = model.fit(
			X_all, 
			[isB_all, isB_all, isMC_all, isMC_all], 
			batch_size=5000, epochs=50,  verbose=1, validation_split=0.2,
			sample_weight = [
				isMC_all.ravel(),
				1-isMC_all.ravel(), 
				np.ones(isB_all.ravel().shape[0]),
				np.ones(isB_all.ravel().shape[0])],				
			callbacks = [learning_rate]
		)
	else:
		raise ValueError('%s is an unknown run option' % Grad)

	predictions = model.predict(X_test)
	preds = pd.DataFrame()
	preds['prediction'] = predictions[0].ravel()
	preds['isB'] = isB_test
	preds['isMC'] = isMC_test
	save(preds, '%s/predictions.npy' %outdir)
	
	history = pd.DataFrame(history.history)
	save(history, '%s/history.npy' %outdir)
	return history

#print history.history.keys()
run_model(
	args.outputDir, 
	Grad=args.method, 
	known = 1, 
	AdversOn=1,
	diffOn = 1
)
###   #print ('damain adaptation with tw sources')
###   history2 = run_model(Grad=2, known = 1,AdversOn=1,diffOn = 1)
###   #print ('train on sources')
###   history3 = run_model(Grad=3, known = 1,AdversOn=1,diffOn = 1)
###   #history4 = run_model(Grad=4, known = 1,AdversOn=1,diffOn = 1)
###   #history5 = run_model(Grad=5, known = 1,AdversOn=1,diffOn = 1)
###   
###   
###   fig = plt.figure()
###   plt.plot(history1.history['val_data_loss'],label='data DA 0.25')
###   plt.plot(history1.history['val_mc_loss'],label='mc DA 0.25')
###   plt.plot(history2.history['val_data_loss'],label='data mc')
###   plt.plot(history2.history['val_mc_loss'],label='mc mc')
###   plt.plot(history3.history['val_data_loss'],label='data data')
###   plt.plot(history3.history['val_mc_loss'],label='mc data')
###   #plt.plot(history4.history['val_data_loss'],label='data DA 0.1')
###   #plt.plot(history4.history['val_mc_loss'],label='mc DA 0.1')
###   #plt.plot(history5.history['val_data_loss'],label='data DA 0.5')
###   #plt.plot(history5.history['val_mc_loss'],label='mc DA 0.5')
###   
###   
###   plt.ylabel('loss')
###   plt.xlabel('epochs')
###   plt.legend()
###   fig.savefig('myPlot')
###   #plt.figure(2)
###   
###   #plt.plot(history.history['val_dense_8_loss'],label='data')
###   #	plt.plot(history.history['val_dense_7_loss'],label='mc')
###   #	plt.legend()
###   #plt.plot(history.history['val_dense_12_loss'])
###   #plt.figure(3)
###   #plt.plot(history.history['val_loss'],label='full loss')
###   #plt.plot(history.history['val_dense_12_loss'])
###   #plt.legend()
###   #plt.show()

