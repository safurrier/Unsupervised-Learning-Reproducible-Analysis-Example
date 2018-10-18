# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './PCA/'
cmap = cm.get_cmap('Spectral') 

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

pca = PCA(random_state=5)
pca.fit(madelonX)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,501))
tmp.to_csv(out+'madelon scree.csv')


pca = PCA(random_state=5)
pca.fit(carsX)
tmp = pd.Series(data = pca.explained_variance_, index = range(1,carsX.shape[1]+1))
tmp.to_csv(out+'cars scree.csv')


#%% Data for 2

grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(madelonX,madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Madelon dim red.csv')

cars_dims = [2,4,6,8,10,12,14,16,18]
grid ={'pca__n_components':cars_dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(carsX,carsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cars dim red.csv')
# raise
#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
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