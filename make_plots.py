import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

da_history = pd.DataFrame(np.load('../domain_adaptation_two_samples/history.npy'))
data_history = pd.DataFrame(np.load('../data_training/history.npy'))
mc_history = pd.DataFrame(np.load('../MC_training/history.npy'))

fig = plt.figure()
nepochs=da_history['val_data_loss_mean'].shape[0]
plt.plot(da_history['val_data_loss_mean'],label='data DA 0.25', c='blue')
plt.fill_between(
	range(nepochs), 
	da_history['val_data_loss_mean']-da_history['val_data_loss_std'], 
	da_history['val_data_loss_mean']+da_history['val_data_loss_std'], 
	color='blue',
	alpha=0.3
	)
plt.plot(da_history['val_mc_loss_mean'],label='mc DA 0.25', c='green')
plt.fill_between(
	range(nepochs), 
	da_history['val_mc_loss_mean']-da_history['val_mc_loss_std'], 
	da_history['val_mc_loss_mean']+da_history['val_mc_loss_std'], 
	color='green',
	alpha=0.3
	)
plt.plot(mc_history['val_data_loss_mean'],label='data mc', c='red')
plt.plot(mc_history['val_mc_loss_mean'],label='mc mc', c='blueviolet')
plt.plot(data_history['val_data_loss_mean'],label='data data', c='orange')
plt.plot(data_history['val_mc_loss_mean'],label='mc data', c='brown')


plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
fig.savefig('myPlot.png')
fig.savefig('myPlot.pdf')
