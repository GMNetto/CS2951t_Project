import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
%matplotlib inline  

params = joblib.load('/data/gen_data/gmarques/network_parameters_snaps_100_1000.jbl')

for key, value in params.items():
    for k2, v2 in value.items():        
        plt.figure()
	plt.title('%s %s' % (key,k2))
        plt.xlabel('Training Iterations 1000x')
	plt.plot(v2) 
plt.figure()
