
#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from collections import defaultdict
from helpers import reconstructionError

out = './ICA/'

np.random.seed(0)
cars = pd.read_hdf('./BASE/datasets.hdf','cars')
carsX = cars.drop('Class',1).copy().values
carsY = cars['Class'].copy().values

madelon = pd.read_hdf('./BASE/datasets.hdf','madelon')        
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values


madelonX = StandardScaler().fit_transform(madelonX)
carsX= StandardScaler().fit_transform(carsX)

clusters =  [2,4,6,8,10,15,20,25,30,35,40]
dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]
cars_dims = [2,4,6,8,10,12,14,16,18,20]
#raise
#%% data for 1

ica = FastICA(random_state=5)

## Reconstruction error from components
reconstruction_error = defaultdict(dict)
for dim in dims:
    ica.set_params(n_components=dim)
    ica.fit(carsX)
    reconstruction_error[dim] = reconstructionError(ica, carsX)
reconstruction_error = pd.DataFrame(reconstruction_error).T
reconstruction_error.to_csv('test_cars_reconstruction_error.csv')