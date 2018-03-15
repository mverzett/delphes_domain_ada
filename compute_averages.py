from argparse import ArgumentParser
import numpy as np
import pandas as pd
from glob import glob
import os
from pdb import set_trace

parser = ArgumentParser()
parser.add_argument('inputdir')
args = parser.parse_args()

def save(df, fname):
  dname = os.path.dirname(fname)
  if not os.path.isdir(dname):
    os.makedirs(dname)
  records = df.to_records(index=False)
  records.dtype.names = [str(i) for i in records.dtype.names]
  np.save(fname, records)

histories = glob('%s/*/history.npy' % args.inputdir)
histories = [pd.DataFrame(np.load(i)) for i in histories]
columns = list(histories[0].columns)
out = pd.DataFrame()
for column in columns:
	vals = pd.concat([i[column] for i in histories], axis=1)
	out['%s_mean' % column] = vals.mean(axis=1)
	out['%s_std'  % column] = vals.std(axis=1)

save(out, '%s/history.npy' % args.inputdir)

predictions = glob('%s/*/predictions.npy' % args.inputdir)
predictions = [pd.DataFrame(np.load(i)) for i in predictions]

#check that MC thruts are the same
assert(
	all(
		((predictions[0][['isB', 'isMC']] == i[['isB', 'isMC']]).all()).all() 
		for i in predictions
		)
	)

out = predictions[0][['isB', 'isMC']]
vals = pd.concat([i['prediction'] for i in predictions], axis=1)
out['prediction_mean'] = vals.mean(axis=1)
out['prediction_std']  = vals.std(axis=1)
for idx, df in enumerate(predictions):
	out['prediction_%d' % idx] = df['prediction']

save(out, '%s/predictions.npy' % args.inputdir)
