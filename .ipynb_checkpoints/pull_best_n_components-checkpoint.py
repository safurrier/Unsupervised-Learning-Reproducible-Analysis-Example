
# coding: utf-8

# In[1]:


import pandas as pd
import ruamel.yaml as yaml
import os
import numpy as np
import sys
import sklearn.model_selection as ms

# ## Change to Root

# In[15]:


NO_CONFIG_ERR_MSG = """No config file found. Root directory is determined by presence of "config.yaml" file."""        

original_wd = os.getcwd()

# Number of times to move back in directory
num_retries = 10
for x in range(0, num_retries):
    # try to load config file    
    try:
        with open("config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)
    # If not found move back one directory level
    except FileNotFoundError:
        os.chdir('../')
        # If reached the max number of directory levels change to original wd and print error msg
        if x+1 == num_retries:
            os.chdir(original_wd)
            print(NO_CONFIG_ERR_MSG)
            
# Add directory to PATH
path = os.getcwd()

if path not in sys.path:
    sys.path.append(path)


# ## Load in CV results
algorithms = ['BASE','ICA', 'PCA', 'RP', 'RF']
datasets = ['Cars', 'Madelon']
output_path = 'results/best_N_components_by_test_score.csv'

records = []
for algorithm in algorithms:
    for dataset in datasets:
        # For base data benchmark
        if algorithm == 'BASE':
            benchmark = pd.read_csv(f'{algorithm}/{dataset} NN bmk.csv')
            best_acc = benchmark.sort_values(by='mean_test_score', ascending=False).mean_test_score.values[0]
            if dataset == 'Cars':
                best_n_components = 19
            if dataset == 'Madelon':
                best_n_components = 500
            records.append((algorithm, dataset, best_n_components, best_acc))
        # For feature transform/selected data
        elif algorithm != 'BASE':
            tmp_csv_scores = pd.read_csv(f'{algorithm}/{dataset} dim red.csv')
            best_n_components = tmp_csv_scores.sort_values(by='mean_test_score', ascending=False).filter(regex='components|filter').values[0][0]
            best_acc = tmp_csv_scores.sort_values(by='mean_test_score', ascending=False).mean_test_score.values[0]
            records.append((algorithm, dataset, best_n_components, best_acc))
        
cols = ['Data_Perspective', 'Dataset', 'N_Components', 'Best_Test_Acc']        
best_N_components = pd.DataFrame(records, columns=cols)

# ## add results from previous analysis using base data
# baseline_results = pd.DataFrame([('BASE', 'cars', '19', .5159), ('BASE', 'madelon', '31', .7628)], columns=cols)
# best_N_components = pd.concat([best_N_components, baseline_results])

best_N_components.to_csv(output_path, index=False)


print(f'\n\nOutput the number of components for algorithms in {algorithms} for each dataset in {datasets} to {output_path}\n\n')

