import tensorflow as tf
from keras.engine import Layer
import keras.backend as K
from Layers import *
#from DeepJet.modules.Losses import moment_loss , nd_moment_factory
import matplotlib
import os
batch_mode = 'DISPLAY' not in os.environ
if batch_mode:
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from make_samples import make_sample
#from DL4Jets.DeepJet.modules.Losses import  weighted_loss

from keras.layers import Dense, Concatenate ,Dropout
from keras.layers import Input
from keras.models import Model

try:
	import setGPU
except ImportError:
	found = False



def modelIverseGrad(Inputs):

    X = Dense(10, activation='relu',input_shape=(20,)) (Inputs)
    X = Dropout(0.25)(X)
    X = Dense(20, activation='relu')(X)
    X = Dropout(0.25)(X)
    X = Dense(10, activation='relu')(X)
    X = Dropout(0.25)(X)
    X = Dense(10, activation='relu')(X)
    X = Dropout(0.25)(X)
    Xa = Dense(20, activation='relu')(X)
    X = Dense(10, activation='relu')(Xa)
    X = Dense(1, activation='sigmoid')(X)
    X1= Dense(1, activation='linear',use_bias=False, trainable=False,kernel_initializer='Ones') (X)
    Ad = GradientReversal()(Xa)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(1, activation='sigmoid')(Ad)
    Ad1 = GradientReversal()(Xa)
    Ad1 = Dense(10, activation='relu')(Ad1)
    Ad1 = Dense(10, activation='relu')(Ad1)
    Ad1 = Dense(10, activation='relu')(Ad1)
    Ad1 = Dense(1, activation='sigmoid')(Ad1)
    predictions = [X,X1,Ad,Ad1]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def run_model(Grad=1, known = 1,AdversOn=1,diffOn = 1):
    Inputs = Input((20,))
    global_loss_list={}
    global_loss_list['GradientReversal']=GradientReversal()
    X_all, isB_all , isMC_all = make_sample()
    advers_weight = 25.
    if AdversOn==0:
        advers_weight = 0.
    
    
    model = modelIverseGrad(Inputs)
    # gradiant loss
    if(Grad==1):
        model.compile(
					loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy'] , 
					optimizer='adam', 
					loss_weights=[0.,5.,0.,0.]
				)
        history = model.fit(
					X_all, [isB_all, isB_all, isMC_all, isMC_all], 
					batch_size=5000, epochs=10, verbose=1,
					validation_split=0.2, 
					sample_weight=[
						isMC_all.ravel(), 1-isMC_all.ravel(), 
						1-isMC_all.ravel()+1,1-isMC_all.ravel()+1]
				)
        model.compile(
					loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy'] , 
					optimizer='adam', 
					loss_weights=[5.,0.,.0,.0]
				)
        history2 = model.fit(
					X_all, [isB_all, isB_all, isMC_all, isMC_all], 
					batch_size=5000, epochs=10,  verbose=1,
					validation_split=0.3,
					sample_weight=[
						isMC_all.ravel(),
						1-isMC_all.ravel(), 
						1-isMC_all.ravel()+1, 
						1-isMC_all.ravel()+1]
				)
    
    print history.history.keys()
    plt.plot(history.history['val_dense_8_loss'],label='data')
    plt.plot(history.history['val_dense_7_loss'],label='mc')
    plt.legend()
    if batch_mode:
        plt.savefig('history_mc_training.png')		
    plt.figure(2)
    plt.plot(history2.history['val_dense_8_loss'],label='data')
    plt.plot(history2.history['val_dense_7_loss'],label='mc')
    plt.legend()
    #plt.plot(history.history['val_dense_12_loss'])
    if batch_mode:
        plt.savefig('history_data_training.png')		
    plt.figure(3)
    plt.plot(history2.history['val_loss'],label='full loss')
    #plt.plot(history.history['val_dense_12_loss'])
    plt.legend()
    if batch_mode:
        plt.savefig('history_data_training_full_loss.png')		
    else:
        plt.show
    return True


run_model(Grad=1, known = 1,AdversOn=1,diffOn = 1)

