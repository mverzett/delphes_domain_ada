import numpy as np
from Losses import binary_crossentropy_labelweights_Delphes
from keras import losses
import tensorflow as tf
from pdb import set_trace
from keras import backend as K

s = tf.Session()
K.set_session(s)
#
# Test Loss
#
x = np.random.rand(10,1)
y = (np.random.rand(10,1) > 0.5).astype(float)

kl = losses.binary_crossentropy(tf.convert_to_tensor(y), tf.convert_to_tensor(x)).eval(session=s)
print kl
xprime = np.hstack((x, np.zeros((10,1))))
cl = binary_crossentropy_labelweights_Delphes(tf.convert_to_tensor(y.reshape((10,1))), tf.convert_to_tensor(xprime)).eval(session=s)
print cl

#
# create fake dataset
#
nsamples = 40000
fb_mc = 0.2
fb_da = 0.5
ismc = (np.random.rand(nsamples) > 0.5)
isb = np.zeros(nsamples)
isb[ismc] += (np.random.rand(ismc.sum()) > fb_mc).astype(float)
isb[np.invert(ismc)] += (np.random.rand(np.invert(ismc).sum()) > fb_da).astype(float)
isb = isb.astype(bool)
weights = ((isb == 1) & ismc).astype(float)
weights = weights.reshape((nsamples, 1))
x = isb.reshape((nsamples, 1)).astype(float)
y = ismc.reshape((nsamples, 1)).astype(float)

#
# Hand-made probabilities
#
p_b = isb.sum() / float(isb.shape[0])
p_MCIb = (ismc & isb).sum()/float(isb.sum())
p_MCIl = (ismc & np.invert(isb)).sum()/float(np.invert(isb).sum())
p_MC = ismc.sum()/float(nsamples)
print 'Hand-made output probs: p(MC | b) = %.2f  p(MC | l) = %.2f' % (p_MCIb, p_MCIl)
print 'Hand-made output expectetions: p(MC | b) * p(b) = %.2f  p(MC | l) * p(l) = %.2f' % (p_MCIb*p_b, p_MCIl*(1-p_b))
print 'Total: %.2f, true %.2f' % ((p_MCIb*p_b+p_MCIl*(1-p_b)), p_MC)

nb_mc = (ismc & isb).sum()
nl_mc = (ismc & np.invert(isb)).sum()
data_b_frac = (np.invert(ismc) & isb).sum() / float(np.invert(ismc).sum())
print 'MC: #B: %d #L: %d' % ((ismc & isb).sum(), (ismc & np.invert(isb)).sum())
exp_f = data_b_frac*nl_mc/(nb_mc*(1-data_b_frac))
exp_f -= 1
print 'Expected weight: %.3f' % exp_f

from DeepJetCore.training.training_base import training_base
from keras.layers import Dense, Dropout, Concatenate, LocallyConnected1D, Reshape, Flatten, Lambda
from keras.models import Model
from keras.layers import Input
from Layers import GradientReversal
from keras.optimizers import Adam

#
# Model 0, really stupid
#
predictions = None
for _ in range(0):
	Inputs = Input((1,))
	#X = Dense(2, activation='relu', name='common_dense_0')
	#X = Dense(2, activation='sigmoid', name = 'btag_mc')(X)
	X = Dense(1, activation='sigmoid', name = 'btag_mc') (Inputs)#(X)
	#X = Dense(2, activation='softmax', name = 'btag_mc')(X)
	#X = Lambda(lambda x: x[:,:1])(X)
	model = Model(inputs=Inputs, outputs=X)
	
	model.compile(
		loss = 'binary_crossentropy',
		optimizer=Adam(lr=0.01),
		)

	model.fit(
		x, y, batch_size=100, epochs=10, verbose=0, validation_split=0.2,
		)
	preds = model.predict(np.array([[1.], [0.]]))

	if predictions is None:
		predictions = preds
	else:
		predictions = np.hstack((predictions, preds))

if predictions is not None: print predictions.mean(axis=1), predictions.std(axis=1)
#
# Model 1, with weights
#
predictions = None
from DeepJetCore.modeltools import set_trainable
for _ in range(1):
	Inputs = [Input((1,)), Input((1,))]
	#X = Dense(2, activation='relu', name='common_dense_0') (Inputs[0])
	#X = Dense(2, activation='sigmoid', name = 'btag_mc')(X)
	X = Dense(1, activation='sigmoid', name = 'btag_mc') (Inputs[0])
	#X = Dense(2, activation='softmax', name = 'btag_mc')(X)
	#X = Lambda(lambda x: x[:,:1])(X)

	#make list out of it, three labels from truth - make weights
	Weight = Reshape((1,1),name='weight_reshape')(Inputs[1])
	Weight = LocallyConnected1D(
		1,1, 
		activation='linear',use_bias=False, 
		kernel_initializer='zeros',
		name="weight_layer") (Weight)    
	Weight= Flatten()(Weight)
	Weight = GradientReversal(name='weight_reversal', hp_lambda=1.)(Weight)
	out = Concatenate(name='out')([X,Weight]) 
	
	model = Model(inputs=Inputs, outputs=out)
	set_trainable(model, ['weight_'], False)
	
	model.compile(
		loss = binary_crossentropy_labelweights_Delphes,
		optimizer=Adam(lr=0.01),
		)
	
	model.fit(
		[x, weights], y, batch_size=100, epochs=1, verbose=1, validation_split=0.2,
		)
	
	weight_layer = [i for i in model.layers if i.name == 'weight_layer'][0]
	set_trainable(model, ['weight_'], True)	
	model.compile(
		loss = binary_crossentropy_labelweights_Delphes,
		optimizer=Adam(lr=0.01),
		)	
	for iepoch in range(4):
		##set_trainable(model, ['weight_'], False)	
		##model.compile(
		##	loss = binary_crossentropy_labelweights_Delphes,
		##	optimizer=Adam(lr=0.01),
		##	)
		##model.fit(
		##	[x, weights], y, batch_size=100, epochs=1, verbose=1, validation_split=0.2,
		##	)
		model.fit(
			[x, weights], y, batch_size=500, epochs=2, verbose=1, validation_split=0.2,
			)
		print '\nEpoch:', iepoch, 'weight:', weight_layer.weights[0].eval(session=s)
	##s.run(
	##	weight_layer.weights[0].assign(
	##		np.array([[[exp_f]]])
	##		)
	##	)
	preds = model.predict(
		[np.array([[1.], [0.]]), np.array([[1.], [0.]])])
	preds = preds.reshape((preds.ravel().shape[0], -1))
	if predictions is None:
		predictions = preds
	else:
		predictions = np.hstack((predictions, preds))

print predictions.mean(axis=1), predictions.std(axis=1)
