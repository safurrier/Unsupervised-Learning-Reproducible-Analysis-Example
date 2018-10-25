import pandas as pd
import ruamel.yaml as yaml
import os
import numpy as np
import sys
import altair as alt

# ## Change to Root

# In[15]:

print('Testing')

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
        if x + 1 == num_retries:
            os.chdir(original_wd)
            print(NO_CONFIG_ERR_MSG)

# Add directory to PATH
path = os.getcwd()

if path not in sys.path:
    sys.path.append(path)

clustering_results = pd.read_csv('results/clustering_results.csv')
grid_search_results = pd.read_csv('results/grid_search_results.csv')
tsne_results = pd.read_csv('results/tsne_results.csv')
component_quality_results = pd.read_csv('results/component_quality.csv')

## Fix Redundant naming of clustering algorithms
clustering_alg_map = {
    'K_Means':'K-Means',
     'EM':'EM',
     'GMM':'EM',
     'Kmeans':'K-Means',
}

clustering_results['Clustering_Algorithm'] = clustering_results['Clustering_Algorithm'].map(clustering_alg_map)
clustering_results['Clustering_Algorithm'].value_counts()

## Pull Mean Train Time and Test Scores marginalized across other params
mean_test_score_col = 'mean_test_score'
mean_fit_time_col = 'mean_fit_time'

anchor_cols = [
 'N_Components/Clusters/Features',
 'Data_Perspective',
 'Dataset',
 'Clustering_Algorithm',
 ]

test_score_grid_search_results = (grid_search_results.groupby(by=anchor_cols)
                                  .mean()[mean_test_score_col] # Take the average across all other parameters
                                  .reset_index()
                                  .rename(columns={mean_test_score_col:'Test_Score'}) # Rename so that it populates the Metric Column
                                  .melt(id_vars=anchor_cols, var_name='Metric', value_name='Value'))

train_time_grid_search_results = (grid_search_results.groupby(by=anchor_cols)
                                  .mean()[mean_fit_time_col] # Take the average across all other parameters
                                  .reset_index()
                                  .rename(columns={mean_fit_time_col:'Fit_Time'}) # Rename so that it populates the Metric Column
                                  .melt(id_vars=anchor_cols, var_name='Metric', value_name='Value'))
tidy_grid_search_results = pd.concat([test_score_grid_search_results, train_time_grid_search_results])


alt.Chart(component_quality_results).mark_line().encode(
    x='N_Components:O',
    y='Value:Q',
    color='Data_Perspective:N',
    column='Dataset:N'
).transform_filter(
    alt.datum.Metric == 'Reconstruction_Error'
).properties(
    title='Reconstruction Error Across Number of Components'
).interactive()
