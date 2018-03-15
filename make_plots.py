import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#
# Losses
#
da_history = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.04/history.npy'))
#pd.DataFrame(np.load(  '../domada_50_epochs_newsample/domain_adaptation_two_samples/history.npy'))
data_history = pd.DataFrame(np.load('../domada_50_epochs_newsample/data_training/history.npy'))
mc_history = pd.DataFrame(np.load(  '../domada_50_epochs_newsample/MC_training/history.npy'))


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


## da_w50_l25_hist = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.25/history.npy'))
## plt.plot(da_w50_l25_hist['val_data_loss_mean'],label='data DA w50 l0.25', c='blue' , ls='--')
## plt.plot(da_w50_l25_hist['val_mc_loss_mean'  ],label='mc DA w50 l0.25'  , c='green', ls='--')
## 
## da_w50_l04_hist = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.04/history.npy'))
## plt.plot(da_w50_l04_hist['val_data_loss_mean'],label='data DA w50 l0.04', c='blue' , ls='-.')
## plt.plot(da_w50_l04_hist['val_mc_loss_mean'  ],label='mc DA w50 l0.04'  , c='green', ls='-.')
## 
## da_w25_l50_hist = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w25_l.5/history.npy'))
## plt.plot(da_w25_l50_hist['val_data_loss_mean'],label='data DA w25 l0.5' , c='blue' , ls=':')
## plt.plot(da_w25_l50_hist['val_mc_loss_mean'  ],label='mc DA w25 l0.5'   , c='green', ls=':')
## 
## da_w05_l01_hist = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w05_l1/history.npy'))
## plt.plot(da_w05_l01_hist['val_data_loss_mean'],label='data DA w5 l1'    , c='cyan' , ls='-')
## plt.plot(da_w05_l01_hist['val_mc_loss_mean'  ],label='mc DA w5 l1'      , c='limegreen', ls='-')

plt.plot(mc_history['val_data_loss_mean'],label='data mc', c='red')
plt.plot(mc_history['val_mc_loss_mean'],label='mc mc', c='blueviolet')
plt.plot(data_history['val_data_loss_mean'],label='data data', c='orange')
plt.plot(data_history['val_mc_loss_mean'],label='mc data', c='brown')

plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(ncol=2, loc='best')
fig.savefig('losses.png')
fig.savefig('losses.pdf')

#
# ROCs
#
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace

da_predictions = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.04/predictions.npy'))
## pd.DataFrame(np.load(  '../domada_50_epochs_newsample/domain_adaptation_two_samples/predictions.npy'))
data_predictions = pd.DataFrame(np.load('../domada_50_epochs_newsample/data_training/predictions.npy'))
mc_predictions = pd.DataFrame(np.load(  '../domada_50_epochs_newsample/MC_training/predictions.npy'))

def draw_roc(df, label, color, draw_unc=False, ls='-', draw_auc=True):
	newx = np.logspace(-4, 0, 100)#arange(0,1,0.01)
	tprs = pd.DataFrame()
	scores = []
	for idx in range(10):
		tmp_fpr, tmp_tpr, _ = roc_curve(df.isB, df['prediction_%d' % idx])
		scores.append(
			roc_auc_score(df.isB, df['prediction_%d' % idx])
			)
		coords = pd.DataFrame()
		coords['fpr'] = tmp_fpr
		coords['tpr'] = tmp_tpr
		clean = coords.drop_duplicates(subset=['fpr'])
		spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr)
		tprs[idx] = spline(newx)
	scores = np.array(scores)
	auc = ' AUC: %.3f +/- %.3f' % (scores.mean(), scores.std()) if draw_auc else ''
	if draw_unc:
		plt.fill_between(
			newx,
			tprs.mean(axis=1) - tprs.std(axis=1),
			tprs.mean(axis=1) + tprs.std(axis=1),
			color=color,
			alpha=0.3
			)		
	plt.plot(newx, tprs.mean(axis=1), label=label + auc, c=color, ls=ls)
	
plt.clf()
draw_roc(
	da_predictions[da_predictions.isMC == 0],
	'data DA 0.25',
	'blue',
	draw_unc = True,
	draw_auc=True,
	)
draw_roc(
	da_predictions[da_predictions.isMC == 1],
	'mc DA 0.25',
	'green',
	draw_unc = True,
	draw_auc=True,
	)

## da_w50_l25_pred = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.25/predictions.npy'))
## draw_roc(da_w50_l25_pred[da_w50_l25_pred.isMC == 0], 'data DA w50 l0.25', 'blue', ls='--', draw_auc=True)
## draw_roc(da_w50_l25_pred[da_w50_l25_pred.isMC == 1], 'mc DA w50 l0.25'  , 'green', ls='--', draw_auc=True)
## 
## da_w50_l04_pred = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.04/predictions.npy'))
## draw_roc(da_w50_l04_pred[da_w50_l04_pred.isMC == 0], 'data DA w50 l0.04', 'blue' , ls='-.', draw_auc=True)
## draw_roc(da_w50_l04_pred[da_w50_l04_pred.isMC == 1], 'mc DA w50 l0.04'  , 'green', ls='-.', draw_auc=True)
## 
## da_w25_l50_pred = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w25_l.5/predictions.npy'))
## draw_roc(da_w25_l50_pred[da_w25_l50_pred.isMC == 0], 'data DA w25 l0.5', 'blue' , ls=':', draw_auc=True)
## draw_roc(da_w25_l50_pred[da_w25_l50_pred.isMC == 1], 'mc DA w25 l0.5'  , 'green', ls=':', draw_auc=True)
## 
## da_w05_l01_pred = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w05_l1/predictions.npy'))
## draw_roc(da_w05_l01_pred[da_w05_l01_pred.isMC == 0], 'data DA w5 l1', 'cyan', ls='-', draw_auc=True)
## draw_roc(da_w05_l01_pred[da_w05_l01_pred.isMC == 1], 'mc DA w5 l1'  , 'limegreen', ls='-', draw_auc=True)

draw_roc(
	mc_predictions[mc_predictions.isMC == 0],
	'data mc', 'red', draw_auc=True
	)
draw_roc(
	mc_predictions[mc_predictions.isMC == 1],
	'mc mc', 'blueviolet', draw_auc=True
	)

draw_roc(
	data_predictions[data_predictions.isMC == 0],
	'data data', 'orange', draw_auc=True
	)
draw_roc(
	data_predictions[data_predictions.isMC == 1],
	'mc data', 'brown', draw_auc=True
	)

plt.xlim(0., 1)
plt.ylim(0.45, 1)
plt.grid(True)
plt.ylabel('true positive rate')
plt.xlabel('false positive rate')
plt.legend(loc='best')
fig.savefig('rocs.png')
fig.savefig('rocs.pdf')

plt.xlim(10**-4, 1)
plt.ylim(0., 1)
plt.gca().set_xscale('log')
fig.savefig('rocs_log.png')
fig.savefig('rocs_log.pdf')
