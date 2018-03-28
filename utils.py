import os
import numpy as np
import pandas as pd

def save(df, fname):
	dname = os.path.dirname(fname)
	if not os.path.isdir(dname):
		os.makedirs(dname)
	records = df.to_records(index=False)
	records.dtype.names = [str(i) for i in records.dtype.names]
	np.save(fname, records)
