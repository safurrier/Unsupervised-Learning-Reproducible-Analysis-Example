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

def fit_tsne_output_record_kl_divergence(dataset, algorithm, dataset_name, perplexity):
    tsne_transformer = TSNE(verbose=0, perplexity = perplexity, random_state=5)
    data_2D = tsne_transformer.fit(dataset) 
    record = (algorithm, dataset_name, tsne_transformer.perplexity, tsne_transformer.kl_divergence_)
    record = pd.DataFrame([record], 
                          columns=['Data_Perspective', 'Dataset', 'Perplexity', 'KL_Divergence'])
    
    return record

cars_perplexities =  pd.concat([fit_tsne_output_record_kl_divergence(carsX, out.replace('/', ''), 'Cars', perplexity) for perplexity in np.arange(0, 150, 5)])
madelon_perplexities =  pd.concat([fit_tsne_output_record_kl_divergence(madelonX, out.replace('/', ''), 'Madelon', perplexity) for perplexity in np.arange(0, 150, 5)])

try:
    perplexity_results = pd.read_csv('results/tsne_perplexity_search/perplexity_kl_divergence_results.csv')
except FileNotFoundError:
    perplexity_results = pd.DataFrame(columns = ['Data_Perspective', 'Dataset', 'Perplexity', 'KL_Divergence'])

perplexity_results = pd.concat([perplexity_results, cars_perplexities, madelon_perplexities])
perplexity_results.to_csv('results/tsne_perplexity_search/perplexity_kl_divergence_results.csv', index=False)