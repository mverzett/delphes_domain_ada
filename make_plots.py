import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#
# Losses
#
da_history = pd.DataFrame(np.load(  '../domada_100_epochs_SV/domain_adaptation_two_samples/history.npy'))
data_history = pd.DataFrame(np.load('../domada_100_epochs_SV/data_training/history.npy'))
mc_history = pd.DataFrame(np.load(  '../domada_100_epochs_SV/MC_training/history.npy'))

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
fig.savefig('losses.png')
fig.savefig('losses.pdf')

#
# ROCs
#
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace

da_predictions = pd.DataFrame(np.load(  '../domada_100_epochs_SV/domain_adaptation_two_samples/predictions.npy'))
data_predictions = pd.DataFrame(np.load('../domada_100_epochs_SV/data_training/predictions.npy'))
mc_predictions = pd.DataFrame(np.load(  '../domada_100_epochs_SV/MC_training/predictions.npy'))

def draw_roc(df, label, color, draw_unc=False):
	fpr, tpr, _ = roc_curve(df.isB, df.prediction_mean)
	if draw_unc:
		newx = np.arange(0,1,0.01)
		tprs = pd.DataFrame()
		scores = []
		auc = ''
		for idx in range(10):
			tmp_fpr, tmp_tpr, _ = roc_curve(df.isB, df['prediction_%d' % idx])
			plt.plot(tmp_fpr, tmp_tpr, ls='--')
			## scores.append(
			## 	roc_auc_score(df.isB, df['prediction_%d' % idx])
			## 	)
			## coords = pd.DataFrame()
			## coords['fpr'] = tmp_fpr
			## coords['tpr'] = tmp_tpr
			## clean = coords.drop_duplicates(subset=['fpr'])
			## spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr)
			## tprs[idx] = spline(newx)
		## scores = np.array(scores)
		## auc = '%.3f +/- %.3f' % (scores.mean(), scores.std())
		## plt.fill_between(
		## 	newx,
		## 	tprs.mean(axis=1) - tprs.std(axis=1),
		## 	tprs.mean(axis=1) + tprs.std(axis=1),
		## 	color=color,
		## 	alpha=0.3
		## 	)		
	else:
		auc = roc_auc_score(df.isB, df.prediction_mean)
		auc = '%.3f' % auc
	plt.plot(fpr, tpr, label=label+' AUC: %s' % auc, c=color)
	
plt.clf()
draw_roc(
	da_predictions[da_predictions.isMC == 0],
	'data DA 0.25',
	'blue',
	draw_unc = True,
	)
## draw_roc(
## 	da_predictions[da_predictions.isMC == 1],
## 	'mc DA 0.25',
## 	'green',
## 	draw_unc = True,
## 	)

## draw_roc(
## 	mc_predictions[mc_predictions.isMC == 0],
## 	'data mc', 'red',
## 	)
## draw_roc(
## 	mc_predictions[mc_predictions.isMC == 1],
## 	'mc mc', 'blueviolet',
## 	)
## 
## draw_roc(
## 	data_predictions[data_predictions.isMC == 0],
## 	'data data', 'orange',
## 	)
## draw_roc(
## 	data_predictions[data_predictions.isMC == 1],
## 	'mc data', 'brown',
## 	)

plt.ylabel('true positive rate')
plt.xlabel('false positive rate')
plt.legend()
fig.savefig('rocs.png')
fig.savefig('rocs.pdf')
