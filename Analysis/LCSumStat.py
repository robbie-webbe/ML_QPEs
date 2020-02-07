

import numpy as np
from glob import glob
from SummaryStats1 import sumstat

fnames = glob(input('Enter files to be imported: '))
arrays = [np.loadtxt(f, delimiter=',') for f in fnames]
final_array = np.concatenate(arrays)

no_curves = int((np.shape(final_array)[0])/2)

analysed_curves = np.zeros((no_curves,8))

label = int(input('Label for set of curves (QPE=1): '))

for i in range(no_curves):
	analysed_curves[i] = sumstat(final_array[(2*i+1)],label)

np.savetxt(input('Enter save file name: '),analysed_curves, delimiter = ',')
