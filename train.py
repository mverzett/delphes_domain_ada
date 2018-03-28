from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys
import os
from pdb import set_trace
from sklearn.model_selection import train_test_split
import json

parser = ArgumentParser()
parser.add_argument('outdir')
parser.add_argument("-i",  help="input directory", default='/data/ml/jkiesele/pheno_domAda/numpyx2', dest='indir')
parser.add_argument("--addsv", action='store_true')
parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
parser.add_argument("--nopred", help="do not compute and store predictions", action='store_true')
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--weight", help="domain adaptation weight", type=float, default=50)
parser.add_argument("--lmb", help="domain adaptation lambda", type=float, default=0.04)
parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=-1)
args = parser.parse_args()

if not os.path.isdir(args.outdir): os.makedirs(args.outdir)
with open('%s/options.json' % args.outdir, 'w') as jfile:
	jfile.write(
		json.dumps(
			args.__dict__, indent=2, separators=(',', ': ')
			)
		)

import keras
from keras import backend as K
import sys
import tensorflow as tf
import os
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

#
# Create Model
#
from DeepJetCore.training.training_base import training_base
from keras.layers import Dense, Dropout, Concatenate, LocallyConnected1D, Reshape, Flatten
from keras.models import Model

from Layers import GradientReversal
from Losses import binary_crossentropy_labelweights_Delphes, categorical_crossentropy_MConly_Delphes, categorical_crossentropy_dataonly_Delphes
import keras.backend as K
from keras.layers.core import Reshape
from pdb import set_trace

import keras.backend as K
from DeepJetCore.modeltools import set_trainable, fixLayersContaining
from keras.layers import Input

## t1 = (np.random.rand(100,1) > 0.5).astype(float)
## t2 = np.zeros((100, 3))
## bkg = np.abs(np.random.normal(scale=0.3, size=(100)))
## t2[(t1 == 0).ravel(), 0] = bkg[(t1 == 0).ravel()]
## t2[(t1 == 1).ravel(), 0] = 1 - bkg[(t1 == 1).ravel()]
## flav = (np.random.rand(100) > 0.5)
## t2[(t1 == 1).ravel() & flav, 1] = 0.4
## t2[(t1 == 1).ravel() & np.invert(flav), 2] = 0.
## 
## y_pred = tf.convert_to_tensor(t2)
## y_true = tf.convert_to_tensor(t1)
## isMCpred = y_pred[:,:1]
## Weightpred = y_pred[:,1:]
## isMCtrue = y_true[:,:1]
## weightsum = K.clip(K.flatten(isMCtrue) * K.sum(Weightpred, axis=-1) + 1, 0.2, 5)
## weighted_xentr = weightsum*K.flatten(K.binary_crossentropy(isMCtrue, isMCpred))
## print K.sum( weighted_xentr , axis=-1)/K.sum(weightsum, axis=-1).eval(session=sess)
## 
## print binary_crossentropy_labelweights_Delphes(tf.convert_to_tensor(t1), tf.convert_to_tensor(t2)).eval(session=sess)


Inputs = [Input((21,)), Input((2,))]
X = Dense(20, activation='relu',input_shape=(21,), name='common_dense_0') (Inputs[0])
X = Dense(10, activation='relu', name='common_dense_1')(X)
X = Dense(10, activation='relu', name='common_dense_2')(X)
X = Dense(10, activation='relu', name='common_dense_3')(X)
Xa= Dense(10, activation='relu', name='common_dense_4')(X)
X = Dense(10, activation='relu', name='btag_classifier_0')(Xa)

X = Dense(1, activation='sigmoid', name = 'btag_mc')(X)
X1= Dense(1, activation='linear',use_bias=False, trainable=False,
		kernel_initializer='Ones', name = 'data') (X)

Ad = Xa #GradientReversal(hp_lambda=0.04, name='Ad_gradrev')(Xa)
Ad = Dense(10, activation='relu', name='Ad_0')(Ad)
Ad = Dense(10, activation='relu', name='Ad_1')(Ad)
Ad = Dense(10, activation='relu', name='Ad_2')(Ad)
Ad = Dense(10, activation='relu', name='Ad_3')(Ad)
Ad = Dense(1, activation='sigmoid', name = 'Ad_out' )(Ad)

#make list out of it, three labels from truth - make weights
Weight = Reshape((2,1),name='weight_reshape')(Inputs[1])
#one-by-one apply weight to label
Weight = LocallyConnected1D(
	1,1, 
	activation='linear',use_bias=False, 
	kernel_initializer='zeros',
	name="weight_layer") (Weight)    
Weight= Flatten()(Weight)
Weight = GradientReversal(name='weight_reversal', hp_lambda=1.)(Weight)

Ad = Concatenate(name='Ad_pred')([Ad,Weight]) 


Ad1 = Xa #GradientReversal(hp_lambda=0.04, name='Ad1_gradrev')(Xa)
Ad1 = Dense(10, activation='relu', name='Ad1_0')(Ad1)
Ad1 = Dense(10, activation='relu', name='Ad1_1')(Ad1)
Ad1 = Dense(10, activation='relu', name='Ad1_2')(Ad1)
Ad1 = Dense(10, activation='relu', name='Ad1_3')(Ad1)
Ad1 = Dense(1, activation='sigmoid', name = 'Ad1_out' )(Ad1)

## #make list out of it, three labels from truth - make weights
## Weight1 = Reshape((2,1),name='weight1_reshape')(Inputs[1])
## #one-by-one apply weight to label
## Weight1 = LocallyConnected1D(
## 	1,1, 
## 	activation='linear',use_bias=False, 
## 	kernel_initializer='ones',
## 	name="weight1_layer") (Weight1)    
## Weight1 = Flatten()(Weight1)
## Weight1 = GradientReversal(name='weight1_reversal', hp_lambda=1.)(Weight1)

Ad1 = Concatenate(name='Ad1_pred')([Ad1, Weight]) #FIXME 


predictions = [X,X1,Ad,Ad1]
model = Model(inputs=Inputs, outputs=predictions)

#
# Load Data
#
from make_samples import make_sample
from keras.layers import Input
X_traintest, isB_traintest , isMC_traintest = make_sample(args.indir, args.addsv)

#
# Use MC only
#
mask = (isMC_traintest == 1).ravel()
X_all = X_traintest[mask]
isB_all = isB_traintest[mask]
train = (np.random.rand(isB_all.shape[0]) > 0.2)
val = np.invert(train)
isMC_all = (np.random.rand(isB_all.shape[0], 1) > 0.5).astype(float)

#X_all[:, -1] = isMC_all.ravel() #FIXME
#X_all[:, -2] = isB_all.ravel() #FIXME

inweights = np.zeros((isB_all.shape[0], 2))
inweights[:,0] += isMC_all.ravel()*isB_all.ravel()
inweights[:,1] += isMC_all.ravel()*(1 - isB_all.ravel())

# define two data-MC samples
data_mc_sample_1 = (np.random.rand(isB_all.shape[0]) > 0.5).astype(float)
data_mc_sample_2 = 1 - data_mc_sample_1
fraction_bias = 0.3 #30% more Bs in MC
domada_weights = np.ones(isB_all.shape[0]) + (isB_all*isMC_all*fraction_bias).ravel() 
mc_only_weights = isMC_all.ravel()# + (isB_all*isMC_all*(1+fraction_bias)).ravel()

#sample 1 - 60% fewer B than standard
sample1_weights = data_mc_sample_1*(1-isB_all.ravel()*0.6)*domada_weights
#sample 2 - 40% more B than standard
sample2_weights = data_mc_sample_2*(1+isB_all.ravel()*0.4)*domada_weights

#norm weights to data size
norm_data = sample1_weights[data_mc_sample_1.astype(bool).ravel() & np.invert(isMC_all.astype(bool)).ravel()].sum()
norm_mc   = sample1_weights[data_mc_sample_1.astype(bool).ravel() & isMC_all.astype(bool).ravel()].sum()
sample1_weights[data_mc_sample_1.astype(bool).ravel() & isMC_all.astype(bool).ravel()] *= norm_data/norm_mc
norm_data = sample2_weights[data_mc_sample_2.astype(bool).ravel() & np.invert(isMC_all.astype(bool)).ravel()].sum()
norm_mc   = sample2_weights[data_mc_sample_2.astype(bool).ravel() & isMC_all.astype(bool).ravel()].sum()
sample2_weights[data_mc_sample_2.astype(bool).ravel() & isMC_all.astype(bool).ravel()] *= norm_data/norm_mc

#
# Define # of epochs
#
class_epochs   = 10
domdiscr_epochs= 15
da_epochs      = 20
from Losses import binary_crossentropy_labelweights_Delphes
from keras.optimizers import Adam
weight_layers = [i for i in model.layers if i.name.startswith('weight') and i.name.endswith('_layer')]


##  set_trainable(model, ['btag_', 'weight_', 'weight1_'], False)
##  set_trainable(model, ['Ad_', 'common_'], True)
##  model.compile(
##  	loss = ['binary_crossentropy']*2+[binary_crossentropy_labelweights_Delphes]*2,
##  	optimizer=Adam(lr=0.001),
##  	loss_weights=[0., 0., 1., 1.],
##  	weighted_metrics = ['accuracy'],
##  	)
##  
##  print 'training data/MC classifier, weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]
##  history = model.fit(
##  	[X_all, inweights], 
##  	[isB_all, isB_all, isMC_all, isMC_all], 
##  	batch_size=10000, epochs=10,  verbose=1, validation_split=0.2,
##  	sample_weight = [
##  		mc_only_weights,
##  		1-isMC_all.ravel(), 
##  		sample1_weights,
##  		sample2_weights],				
##  	)
##  df_history = pd.concat(
##  	[df_history,
##  	 pd.DataFrame(history.history)
##  	 ], ignore_index=True
##  )
##  
##  print 'output weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]


#
# Compile and train btag classifier
#
from Losses import binary_crossentropy_labelweights_Delphes
from keras.optimizers import Adam
set_trainable(model, ['Ad_', 'weight_', 'Ad1_', 'weight1_', ], False)
model.compile(
	loss = ['binary_crossentropy']*2+[binary_crossentropy_labelweights_Delphes]*2,
	optimizer=Adam(lr=0.001),
	loss_weights=[1., 0., 0., 0.],
	weighted_metrics = ['accuracy'],
	)

print '\n\ntraining b-tag classifier, weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]
history = model.fit(
	[X_all, inweights], 
	[isB_all, isB_all, isMC_all, isMC_all], 
	batch_size=10000, epochs=class_epochs,  verbose=1, validation_split=0.2,
	sample_weight = [
		mc_only_weights,
		1-isMC_all.ravel(), 
		sample1_weights,
		sample2_weights],				
	)
df_history = pd.DataFrame(history.history)

print '\n\noutput weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]


#
# Compile and train data/MC classifier
#
set_trainable(model, ['common_', 'btag_', 'weight_', 'weight1_'], False)
set_trainable(model, ['Ad_', 'Ad1_'], True)
model.compile(
	loss = ['binary_crossentropy']*2+[binary_crossentropy_labelweights_Delphes]*2,
	optimizer=Adam(lr=0.001),
	loss_weights=[0., 0., 1., 1.],
	weighted_metrics = ['accuracy'],
	)

print '\n\ntraining data/MC classifier, weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]
history = model.fit(
	[X_all, inweights], 
	[isB_all, isB_all, isMC_all, isMC_all], 
	batch_size=10000, epochs=domdiscr_epochs,  verbose=1, validation_split=0.2,
	sample_weight = [
		mc_only_weights,
		1-isMC_all.ravel(), 
		sample1_weights,
		sample2_weights],				
	)
df_history = pd.concat(
	[df_history,
	 pd.DataFrame(history.history)
	 ], ignore_index=True
)

print '\n\noutput weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]


#
# release weights
#
set_trainable(model, ['weight_', 'weight1_'], True)
set_trainable(model, ['common_', 'btag_', 'Ad_', 'Ad1_'], False)
model.compile(
	loss = ['binary_crossentropy']*2+[binary_crossentropy_labelweights_Delphes]*2,
	optimizer=Adam(lr=0.001),
	loss_weights=[0., 0., 50., 50.],
	weighted_metrics = ['accuracy'],
	)

print '\n\ntraining everything, weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]
history = model.fit(
	[X_all, inweights], 
	[isB_all, isB_all, isMC_all, isMC_all], 
	batch_size=10000, epochs=da_epochs,  verbose=1, validation_split=0.2,
	sample_weight = [
		mc_only_weights,
		1-isMC_all.ravel(), 
		sample1_weights,
    sample2_weights],
	)
df_history = pd.concat(
	[df_history,
	 pd.DataFrame(history.history)
	 ], ignore_index=True
)

print '\n\noutput weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]

## #forcing the fucking correct weights
## sess.run(
## 	weight_layers[0].weights[0].assign(
## 		np.array([[[1./1.3]], [[1]]])
## 		)
## 	)
## 
## print 'training everything, weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]
## history = model.fit(
## 	[X_all, inweights], 
## 	[isB_all, isB_all, isMC_all, isMC_all], 
## 	batch_size=10000, epochs=da_epochs,  verbose=1, validation_split=0.2,
## 	sample_weight = [
## 		mc_only_weights,
## 		1-isMC_all.ravel(), 
## 		sample1_weights,
##     sample2_weights],
## 	)
## print 'output weights:', [i.weights[0].eval(sess).ravel() for i in weight_layers]
## 
## from utils import save
## save(df_history, '%s/history.npy' % args.outdir)
## 
## print df_history
## 
## mask = (data_mc_sample_1 == 1) & (np.random.rand(data_mc_sample_1.shape[0]) > 0.99)
## sam1 = [X_all[mask], inweights[mask]]
## sam1_target = isMC_all[mask]
## pred1 = model.predict(sam1)
## dmc_pred = pred1[2]
## sam1_w = sample1_weights[mask]
## dmcp_v1 = np.copy(dmc_pred)
## dmcp_v1[:,1:] *= sam1_w.reshape((sam1_w.shape[0],1))
## 
## set_trace()
## xntrp1 = binary_crossentropy_labelweights_Delphes(
## 	tf.convert_to_tensor(sam1_target),
## 	tf.convert_to_tensor(dmcp_v1.astype(float)),	
## 	)
