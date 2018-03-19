import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import keras
from keras import backend as K
import sys
import tensorflow as tf
import os
from pdb import set_trace
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument('inputdir')
parser.add_argument("-p",  dest='postfix', default='', help="plot postfix")
args = parser.parse_args()


#
# Losses
#
#pd.DataFrame(np.load(  '../domada_50_epochs_newsample/domain_adaptation_two_samples/history.npy'))
da_history   = pd.DataFrame(np.load('%s/domain_adaptation_two_samples_w50_l.04/history.npy' % args.inputdir))
data_history = pd.DataFrame(np.load('%s/data_training/history.npy' % args.inputdir))
mc_history   = pd.DataFrame(np.load('%s/MC_training/history.npy' % args.inputdir))


fig = plt.figure()
nepochs=da_history['val_data_loss_mean'].shape[0]
plt.plot(da_history['val_data_loss_mean'],label='data DA w 50 L .04', c='blue')
plt.fill_between(
	range(nepochs), 
	da_history['val_data_loss_mean']-da_history['val_data_loss_std'], 
	da_history['val_data_loss_mean']+da_history['val_data_loss_std'], 
	color='blue',
	alpha=0.3
	)
plt.plot(da_history['val_mc_loss_mean'],label='mc DA w 50 L .04', c='green')
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
fig.savefig('%s/losses%s.png' % (args.inputdir, args.postfix))
fig.savefig('%s/losses%s.pdf' % (args.inputdir, args.postfix))

#
# ROCs
#
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace

## pd.DataFrame(np.load(  '../domada_50_epochs_newsample/domain_adaptation_two_samples/predictions.npy'))
da_predictions   = pd.DataFrame(np.load('%s/domain_adaptation_two_samples_w50_l.04/predictions.npy' % args.inputdir))
data_predictions = pd.DataFrame(np.load('%s/data_training/predictions.npy' % args.inputdir))
mc_predictions   = pd.DataFrame(np.load('%s/MC_training/predictions.npy' % args.inputdir))

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
	'data DA w 50 L .04',
	'blue',
	draw_unc = True,
	draw_auc=True,
	)
draw_roc(
	da_predictions[da_predictions.isMC == 1],
	'mc DA w 50 L .04',
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
fig.savefig('%s/rocs%s.png' % (args.inputdir, args.postfix))
fig.savefig('%s/rocs%s.pdf' % (args.inputdir, args.postfix))

plt.xlim(10**-4, 1)
plt.ylim(0., 1)
plt.gca().set_xscale('log')
fig.savefig('%s/rocs_log%s.png' % (args.inputdir, args.postfix))
fig.savefig('%s/rocs_log%s.pdf' % (args.inputdir, args.postfix))


def plot_discriminator(df, name):
	plt.clf()
	plt.hist(
		[df[df.isB == 1].prediction_mean, df[df.isB == 0].prediction_mean],
		bins = 50, range=(0, 1.), histtype='bar', stacked=True,
		color=['green', 'blue'], label=['B jets', 'light jets']
		)
	plt.ylabel('occurrences')
	plt.xlabel('NN output (averaged)')
	plt.legend(loc='best')
	fig.savefig('%s/%s%s.png' % (args.inputdir, name, args.postfix))
	fig.savefig('%s/%s%s.pdf' % (args.inputdir, name, args.postfix))

plot_discriminator(da_predictions[da_predictions.isMC == 1], 'nn_out_da_mc')
plot_discriminator(da_predictions[da_predictions.isMC == 0], 'nn_out_da_data')

plot_discriminator(data_predictions[data_predictions.isMC == 1], 'nn_out_dataTraining_mc')
plot_discriminator(data_predictions[data_predictions.isMC == 0], 'nn_out_dataTraining_data')

plot_discriminator(mc_predictions[mc_predictions.isMC == 1], 'nn_out_mcTraining_mc')
plot_discriminator(mc_predictions[mc_predictions.isMC == 0], 'nn_out_mcTraining_data')
