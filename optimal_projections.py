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



#### ICA optimal Projections
dim = 15
ica = FastICA(n_components=dim,random_state=10)

madelonX2 = ica.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 14
ica = FastICA(n_components=dim,random_state=10)
carsX2 = ica.fit_transform(carsX)
cars2 = pd.DataFrame(np.hstack((carsX2,np.atleast_2d(carsY).T)))
cols = list(range(cars2.shape[1]))
cols[-1] = 'Class'
cars2.columns = cols
cars2.to_hdf(out+'datasets.hdf','cars',complib='blosc',complevel=9)


##### PCA Optimal Projections
dim = 5
pca = PCA(n_components=dim,random_state=10)

madelonX2 = pca.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 12
pca = PCA(n_components=dim,random_state=10)
carsX2 = pca.fit_transform(carsX)
cars2 = pd.DataFrame(np.hstack((carsX2,np.atleast_2d(carsY).T)))
cols = list(range(cars2.shape[1]))
cols[-1] = 'Class'
cars2.columns = cols
cars2.to_hdf(out+'datasets.hdf','cars',complib='blosc',complevel=9)

##### RP Optimal Components
dim = 20
rp = SparseRandomProjection(n_components=dim,random_state=5)

madelonX2 = rp.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 35
rp = SparseRandomProjection(n_components=dim,random_state=5)
carsX2 = rp.fit_transform(carsX)
cars2 = pd.DataFrame(np.hstack((carsX2,np.atleast_2d(carsY).T)))
cols = list(range(cars2.shape[1]))
cols[-1] = 'Class'
cars2.columns = cols
cars2.to_hdf(out+'datasets.hdf','cars',complib='blosc',complevel=9)

##### RF Optimal Components
dim = 20
filtr = ImportanceSelect(rfc,dim)

madelonX2 = filtr.fit_transform(madelonX,madelonY)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 16
filtr = ImportanceSelect(rfc,dim)
carsX2 = filtr.fit_transform(carsX,carsY)
cars2 = pd.DataFrame(np.hstack((carsX2,np.atleast_2d(carsY).T)))
cols = list(range(cars2.shape[1]))
cols[-1] = 'Class'
cars2.columns = cols
cars2.to_hdf(out+'datasets.hdf','cars',complib='blosc',complevel=9)



