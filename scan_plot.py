import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob

h_files = glob('../smearing_5x_75epochs/*/history.npy')
histories = [
	pd.DataFrame(np.load(i))
	for i in h_files
]

infos = pd.DataFrame()
infos['file'] = h_files

infos['val_data_loss_mean'] = [i['val_data_loss_mean'].iloc[-1] for i in histories]
infos['val_data_loss_std' ] = [i['val_data_loss_std' ].iloc[-1] for i in histories]
infos['val_mc_loss_mean'] = [i['val_mc_loss_mean'].iloc[-1] for i in histories]
infos['val_mc_loss_std' ] = [i['val_mc_loss_std' ].iloc[-1] for i in histories]
infos['delta'] = np.abs(infos.val_data_loss_mean - infos.val_mc_loss_mean)
infos['delta_err'] = np.sqrt(infos.val_data_loss_std**2 + infos.val_mc_loss_std**2)

data_train_loss = pd.DataFrame(np.load('../smearing_5x_50epochs/data_training/history.npy'))
data_loss = data_train_loss['val_data_loss_mean'].iloc[-1]
data_std = data_train_loss['val_data_loss_std'].iloc[-1]
infos['delta_data'] = infos.val_data_loss_mean - data_loss
infos['delta_data_err'] = np.sqrt(infos.val_data_loss_std**2 + data_std**2)

plt.clf()
plt.errorbar(
	infos.index, infos.delta, yerr=infos.delta_err,	
	ls=None, fmt='none', ecolor='blue', label='|data loss - mc loss| DA'
)
plt.errorbar(
	infos.index, infos.delta_data, yerr=infos.delta_data_err,	
	ls=None, fmt='none', ecolor='red', label='data loss (DA) - data loss (data)'
)
plt.legend(loc='best')
plt.ylim(0, 0.1)
plt.savefig('test.png')

#
# Best point found at
# '../smearing_5x_75epochs/domain_adaptation_lr0.000500_w300.000000_l0.040000/history.npy'
#
