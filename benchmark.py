# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './BASE/'
np.random.seed(0)
cars = pd.read_hdf('./BASE/datasets.hdf','cars')
carsX = cars.drop('Class',1).copy().values
carsY = cars['Class'].copy().values

madelon = pd.read_hdf('./BASE/datasets.hdf','madelon')
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values


madelonX = StandardScaler().fit_transform(madelonX)
carsX= StandardScaler().fit_transform(carsX)

#%% benchmarking for chart type 2

grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(madelonX,madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Madelon NN bmk.csv')


mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(carsX,carsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cars NN bmk.csv')

