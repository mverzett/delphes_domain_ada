from keras.engine import Layer
from Layers import *
from keras.layers import Dense, Concatenate ,Dropout
from keras.layers import Input
from keras.models import Model

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
