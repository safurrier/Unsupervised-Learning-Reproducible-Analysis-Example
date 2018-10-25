# -*- coding: utf-8 -*-

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
import sys

out = '{}/'.format(sys.argv[1])

np.random.seed(0)
cars = pd.read_hdf(out+'datasets.hdf','cars')
carsX = cars.drop('Class',1).copy().values
carsY = cars['Class'].copy().values

madelon = pd.read_hdf(out+'datasets.hdf','madelon')        
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values


madelonX = StandardScaler().fit_transform(madelonX)
carsX= StandardScaler().fit_transform(carsX)


perplexity_results = pd.read_csv('results/tsne_perplexity_search/perplexity_kl_divergence_results.csv')
# Remove Perplexity of 0
perplexity_results = perplexity_results[perplexity_results['Perplexity'] != 0]
min_kl = perplexity_results.sort_values(by='KL_Divergence').drop_duplicates(subset=['Data_Perspective', 'Dataset'], keep='first')
min_kl_perplexity_car = min_kl.query('Data_Perspective == @out'
                                             '& Dataset == "Cars"').Perplexity.values[0]
min_kl_perplexity_madelon = min_kl.query('Data_Perspective == @out'
                                             '& Dataset == "Madelon"').Perplexity.values[0]


# %% For chart 4/5
# Madelon perplexity set to 50 b/c it's high dimensional and points likely not dense like Cars
madelonX2D = TSNE(verbose=10, perplexity = 30.0, random_state=5).fit_transform(madelonX)
carsX2D = TSNE(verbose=10, perplexity = 30.0, random_state=5).fit_transform(carsX)

madelon2D = pd.DataFrame(np.hstack((madelonX2D,np.atleast_2d(madelonY).T)),columns=['x','y','target'])
cars2D = pd.DataFrame(np.hstack((carsX2D,np.atleast_2d(carsY).T)),columns=['x','y','target'])

madelon2D.to_csv(out+'madelon2D.csv')
cars2D.to_csv(out+'cars2D.csv')