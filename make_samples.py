import numpy as np
import sys, os
from pdb import set_trace

def make_sample():
    X_all = np.load('/afs/cern.ch/work/m/mverzett/public/delphes_training/large_dataset_features_0.npy')
    Y_all = np.load('/afs/cern.ch/work/m/mverzett/public/delphes_training/large_dataset_truth_1.npy')

    print X_all.shape, Y_all.shape

    Y_C = Y_all[:,2:3]<0.1
    print Y_C.shape

    Y_all = Y_all[Y_C.ravel()]
    X_all = X_all[Y_C.ravel()]

    print X_all.shape, Y_all.shape

    isB_all = Y_all[:,1:2]
    isMC_all = Y_all[:,0:1]
    
    

#np.save('X_all.npy',X_all)
#    np.save('isB_all.npy',isB_all)
#    np.save('isMC_all.npy',isMC_all)

#    np.save('X_mc.npy',X_all[(isMC_all>.1).ravel()])
#    np.save('isB_mc.npy',isB_all[(isMC_all>.1).ravel()])
#    np.save('isMC_mc.npy',isMC_all[(isMC_all>.1).ravel()])

#    np.save('X_data.npy',X_all[(isMC_all<.1).ravel()])
#    np.save('isB_data.npy',isB_all[(isMC_all<.1).ravel()])
#    np.save('isMC_data.npy',isMC_all[(isMC_all<.1).ravel()])

    return X_all, isB_all , isMC_all,
