# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.decomposition import PCA
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from matplotlib import cm
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product
from helpers import   nn_arch,nn_reg,ImportanceSelect
from sklearn.ensemble import RandomForestClassifier

### Components chosen because of their respective quality metrics (explained variance/kurtosis/correlation)
### and reconstruction error (RP chosen at 20 b/c past this point no reconstruction error likely noise in futher components)
np.random.seed(0)
cars = pd.read_hdf('./BASE/datasets.hdf','cars')
carsX = cars.drop('Class',1).copy().values
carsY = cars['Class'].copy().values

madelon = pd.read_hdf('./BASE/datasets.hdf','madelon')        
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values


madelonX = StandardScaler().fit_transform(madelonX)
carsX= StandardScaler().fit_transform(carsX)


#### ICA optimal Projections
out = 'ICA'
dim = 5
ica = FastICA(n_components=dim,random_state=10)

madelonX2 = ica.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'/datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 9
ica = FastICA(n_components=dim,random_state=10)
carsX2 = ica.fit_transform(carsX)
cars2 = pd.DataFrame(np.hstack((carsX2,np.atleast_2d(carsY).T)))
cols = list(range(cars2.shape[1]))
cols[-1] = 'Class'
cars2.columns = cols
cars2.to_hdf(out+'/datasets.hdf','cars',complib='blosc',complevel=9)


##### PCA Optimal Projections
out = 'PCA'
dim = 5
pca = PCA(n_components=dim,random_state=10)

madelonX2 = pca.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'/datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 7
pca = PCA(n_components=dim,random_state=10)
carsX2 = pca.fit_transform(carsX)
cars2 = pd.DataFrame(np.hstack((carsX2,np.atleast_2d(carsY).T)))
cols = list(range(cars2.shape[1]))
cols[-1] = 'Class'
cars2.columns = cols
cars2.to_hdf(out+'/datasets.hdf','cars',complib='blosc',complevel=9)

##### RP Optimal Components
out = 'RP'
dim = 20
rp = SparseRandomProjection(n_components=dim,random_state=5)

madelonX2 = rp.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'/datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 35
rp = SparseRandomProjection(n_components=dim,random_state=5)
carsX2 = rp.fit_transform(carsX)
cars2 = pd.DataFrame(np.hstack((carsX2,np.atleast_2d(carsY).T)))
cols = list(range(cars2.shape[1]))
cols[-1] = 'Class'
cars2.columns = cols
cars2.to_hdf(out+'/datasets.hdf','cars',complib='blosc',complevel=9)

##### RF Optimal Components
out = 'RF'
dim = 20
rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
filtr = ImportanceSelect(rfc,dim)


madelonX2 = filtr.fit_transform(madelonX,madelonY)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'/datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 20
filtr = ImportanceSelect(rfc,dim)
carsX2 = filtr.fit_transform(carsX,carsY)
cars2 = pd.DataFrame(np.hstack((carsX2,np.atleast_2d(carsY).T)))
cols = list(range(cars2.shape[1]))
cols[-1] = 'Class'
cars2.columns = cols
cars2.to_hdf(out+'/datasets.hdf','cars',complib='blosc',complevel=9)



