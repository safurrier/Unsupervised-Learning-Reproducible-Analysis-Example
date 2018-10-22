
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

def main():
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


    # ## Clean results so that similar outputs are organized by UL technique, dataset and metric/result

    # First clean based on one set and then expand to all others

    # ## 1) Cleaning the Clustering Quality Metrics (SSE (minimize) for K Means, log likelihood (maximize) for GMM)

    def clean_clustering_metrics(fpath):
        metric_df = pd.read_csv(fpath)
        metric = fpath.split('/')[1].split('.')[0]
        # Correct Spelling mistake
        if metric == 'logliklihood':
            metric='log-likelihood'
        metric_df.columns = ['N_Clusters', 'Madelon_'+metric, 'Cars_'+metric]
        clean_metric_df = (metric_df
                       .melt(id_vars=['N_Clusters'], value_name='Value')
                       .assign(Dataset= lambda X: X.variable.str.split('_').str.get(0))
                       .assign(Metric= lambda X: X.variable.str.split('_').str.get(1))
                       .drop(columns=['variable'])
        )
        if clean_metric_df.Metric.unique()[0] == 'log-likelihood':
            clean_metric_df['Clustering_Algorithm'] = 'EM'
        elif clean_metric_df.Metric.unique()[0] == 'SSE':
            clean_metric_df['Clustering_Algorithm'] = 'K_Means'
        else:
            clean_metric_df['Clustering_Algorithm'] = np.nan

        column_order = ['Clustering_Algorithm', 'N_Clusters', 'Dataset', 'Metric', 'Value',]

        return clean_metric_df[column_order]

    def clean_clustering_validation_metrics(algorithm_file_dir_prefix):
        sse_fpath = f'{algorithm_file_dir_prefix}/SSE.csv'
        em_fpath = f'{algorithm_file_dir_prefix}/logliklihood.csv'

        clean_sse = clean_clustering_metrics(sse_fpath)
        clean_em = clean_clustering_metrics(em_fpath)

        clean_clustering_metics_df = (pd.concat([clean_sse, clean_em],
                                             sort=False)
                                   .reset_index(drop=True)
                                  )
        return clean_clustering_metics_df



    # ## 2) Cleaning the classification/quality metrics of the clusters (accuracy when using cluster labels as predictions and Mutual information between cluster labels and target labels

    def clean_clustering_classification_metric_df(fpath):
        metric_df = pd.read_csv(fpath)
        metric = fpath.split('/')[1].split('.')[0].split(' ')[1]
        # Correct Spelling mistake
        clean_metric_df = (metric_df.rename(columns={'Unnamed: 0':'Clustering_Algorithm'})
         .melt(id_vars='Clustering_Algorithm', var_name='N_Clusters', value_name='Value',)
        )
        if metric == 'acc':
            clean_metric_df['Metric'] = 'Accuracy'
        elif metric == 'adjMI':
            clean_metric_df['Metric'] = 'Mutual_Information'
        else:
            clean_metric_df['Metric'] = np.nan

        return clean_metric_df

    def clean_clustering_classification_metrics(algorithm_file_dir_prefix, collection_of_dataset_names):
        clean_metric_dfs = []
        for dataset in collection_of_dataset_names:
            acc_fpath = f'{algorithm_file_dir_prefix}/{dataset} acc.csv'
            adjmi_fpath = f'{algorithm_file_dir_prefix}/{dataset} adjMI.csv'
            clean_acc = clean_clustering_classification_metric_df(acc_fpath)
            clean_acc['Dataset'] = dataset
            clean_adjmi = clean_clustering_classification_metric_df(adjmi_fpath)
            clean_adjmi['Dataset'] = dataset


            clean_classification_metric_df = (pd.concat([clean_acc, clean_adjmi],
                                                 sort=False)
                                       .reset_index(drop=True)
                                      )
            clean_metric_dfs.append(clean_classification_metric_df)

        # Concat all together
        clean_metric_df = (pd.concat(clean_metric_dfs,
                                                 sort=False)
                                       .reset_index(drop=True)
                                      )
        column_order = ['Clustering_Algorithm', 'N_Clusters', 'Dataset', 'Metric', 'Value',]


        return clean_metric_df[column_order]

    def clean_all_clustering_non_grid_search_metrics(algorithm_file_dir_prefix, collection_of_dataset_names):
        """Clean the clustering metrics (Accuracy, Mutual Info, SSE for Kmeans and Likelihood for EM
        and pull into a single clean dataframe"""
        clean_clustering_validation_metrics_df = clean_clustering_validation_metrics(algorithm_file_dir_prefix)
        clean_clustering_classification_metrics_df = clean_clustering_classification_metrics(algorithm_file_dir_prefix, 
                                                                                             collection_of_dataset_names)
        clean_metric_df = (pd.concat([clean_clustering_validation_metrics_df, clean_clustering_classification_metrics_df],
                                                 sort=False)
                                       .reset_index(drop=True)
                                      )
        clean_metric_df['Data_Perspective'] = algorithm_file_dir_prefix
        return clean_metric_df


    # Getting one such pairing of clustering metrics


    # In[67]:


    all_algorithm_clustering_methods = pd.concat([clean_all_clustering_non_grid_search_metrics(algorithm, ['Cars', 'Madelon'])                                   for algorithm in ['BASE', 'ICA', 'PCA', 'RP', 'RF']])


    # # Export to results HDF

    # In[68]:


    all_algorithm_clustering_methods.to_hdf('results/results.hdf', key='clustering', complib='blosc',complevel=9)
    all_algorithm_clustering_methods.to_csv('results/clustering_results.csv', index=False)


    # ## Compile T-SNE Results to one table

    # In[42]:


    def pull_tsne(algorithm, dataset):
        tsne = pd.read_csv(f'{algorithm}/{dataset}2D.csv').drop(columns=['Unnamed: 0'])
        tsne['Dataset'] = dataset
        tsne['Data_Perspective'] = algorithm
        return tsne

    tsne_dfs = [pull_tsne(algorithm, 'cars')
     for algorithm 
     in ['BASE', 'ICA', 'PCA', 'RP', 'RF']]+ [pull_tsne(algorithm, 'madelon')
     for algorithm 
     in ['BASE', 'ICA', 'PCA', 'RP', 'RF']]




    # In[72]:


    tsne_df = pd.concat(tsne_dfs).rename(columns={'x':'X', 'y':'Y', 'target':'Target'})

    # Export


    tsne_df.to_hdf('results/results.hdf', key='tsne', complib='blosc',complevel=9)
    tsne_df.to_csv('results/tsne_results.csv', index=False)


    # ## Compile GridSearch Results to one table

    # In[152]:


    def pull_grid_search(algorithm, dataset, clustering=True):
        """Given a data unsupervised learning algorithm, dataset name and whether it's 
        cluster related or not, pull grid search results"""
        # Load the clustering data if necesary
        if clustering:
            # Load in Grid Searches from Clustering efforts
            cluster_alg = 'GMM'
            cluster1_df = pd.read_csv(f'{algorithm}/{dataset} cluster {cluster_alg}.csv').drop(columns=['Unnamed: 0'])
            cluster1_df['Data_Perspective'] = algorithm        
            cluster1_df['Dataset'] = dataset
            cluster1_df['Clustering_Algorithm'] = cluster_alg
            cluster1_df['Clustered_Data'] = 1
            # Rename N components so that the concatenation works with columns aligned
            n_components_colname = cluster1_df.filter(regex='(_n_|filt)').columns.values.tolist()[0]
            cluster1_df.rename(columns={n_components_colname:'N_Components/Clusters/Features'}, inplace=True)
            # Remove Individual Split columns
            split_columns = cluster1_df.filter(regex='split').columns.values.tolist()
            cluster1_df = cluster1_df.drop(columns=split_columns)

            cluster_alg = 'Kmeans'
            cluster2_df = pd.read_csv(f'{algorithm}/{dataset} cluster {cluster_alg}.csv').drop(columns=['Unnamed: 0'])
            cluster2_df['Data_Perspective'] = algorithm        
            cluster2_df['Dataset'] = dataset        
            cluster2_df['Clustering_Algorithm'] = cluster_alg
            cluster2_df['Clustered_Data'] = 1    
            n_components_colname = cluster2_df.filter(regex='(_n_|filt)').columns.values.tolist()[0]
            cluster2_df.rename(columns={n_components_colname:'N_Components/Clusters/Features'}, inplace=True)
            # Remove Individual Split columns
            split_columns = cluster2_df.filter(regex='split').columns.values.tolist()
            cluster2_df = cluster2_df.drop(columns=split_columns)


        # There's no dimension reduction for BASE data
        if algorithm != 'BASE':
            grid_search_df = pd.read_csv(f'{algorithm}/{dataset} dim red.csv').drop(columns=['Unnamed: 0'])
            grid_search_df['Data_Perspective'] = algorithm        
            grid_search_df['Dataset'] = dataset
            grid_search_df['Clustering_Algorithm'] = 'None'
            grid_search_df['Clustered_Data'] = 0 
            n_components_colname = grid_search_df.filter(regex='(_n_|filt)').columns.values.tolist()[0]
            grid_search_df.rename(columns={n_components_colname:'N_Components/Clusters/Features'}, inplace=True) 
            # Remove Individual Split columns
            split_columns = grid_search_df.filter(regex='split').columns.values.tolist()
            grid_search_df = grid_search_df.drop(columns=split_columns)        


        if clustering & (algorithm != 'BASE'):         
            clean_grid_search =  pd.concat([cluster1_df, cluster2_df, grid_search_df])
            return clean_grid_search
        elif clustering & (algorithm == 'BASE'):
            clean_grid_search =  pd.concat([cluster1_df, cluster2_df])
            return clean_grid_search
        else:
            return 'Either clustering or grid search not found'

    grid_search_dfs = [pull_grid_search(algorithm, 'Cars')
     for algorithm 
     in ['BASE', 'ICA', 'PCA', 'RP', 'RF']] + [pull_grid_search(algorithm, 'Madelon')
     for algorithm 
     in ['BASE', 'ICA', 'PCA', 'RP', 'RF']]

    # ## Export grid search columns

    # In[158]:


    clean_grid_search_df = pd.concat(grid_search_dfs)


    # Export


    clean_grid_search_df.to_hdf('results/results.hdf', key='grid_search', complib='blosc',complevel=9)
    clean_grid_search_df.to_csv('results/grid_search_results.csv', index=False)
    
    def pull_cluster_quality_metrics(algorithm, dataset):
        """Given a data unsupervised learning algorithm, dataset name and whether it's 
        cluster related or not, pull grid search results"""
        # Load the clustering data if necesary

        # There's no dimension reduction for BASE data
        if (algorithm != 'BASE') & (algorithm != 'RF'):
            if algorithm == 'PCA':
                scree1 = pd.read_csv(f'{algorithm}/{dataset} scree.csv', header=None)
                metric = 'Explained_Variance'
                scree1.columns=['N_Components', 'Value']
                scree1['Metric'] = metric

                scree2 = (pd.read_csv(f'{algorithm}/{dataset} scree2.csv').drop(columns='Unnamed: 0')
                          .melt(id_vars='N_Components', var_name='Metric', value_name='Value'))

                alg_metrics = pd.concat([scree1, scree2], sort=True)
            elif algorithm == 'ICA':
                scree1 = pd.read_csv(f'{algorithm}/{dataset} scree.csv', header=None)
                metric = 'Avg_Component_Absolute_Kurtosis'
                scree1.columns=['N_Components', 'Value']
                scree1['Metric'] = metric

                scree2 = (pd.read_csv(f'{algorithm}/{dataset} scree2.csv').drop(columns='Unnamed: 0')
                          .melt(id_vars='N_Components', var_name='Metric', value_name='Value'))

                alg_metrics = pd.concat([scree1, scree2], sort=True)            

            elif algorithm == 'RP':
                ### Correlation Metric
                scree1 = pd.read_csv(f'{algorithm}/{dataset} scree.csv')            
                # Compute average across 10 random projections
                rp_metrics_avg_corr = pd.concat([scree1.iloc[:, 0], 
                                                 scree1.iloc[:, 1:].apply(np.mean, axis=1)],
                axis=1)
                rp_metrics_avg_corr.columns = ['N_Components', 'Value']
                # Compute Std. across 10 random projections
                rp_metrics_avg_corr['Metric'] = 'Projected_Pairwise_Distance_10_Trial_Correlation_Avg'
                rp_metrics_std_corr = pd.concat([scree1.iloc[:, 0], 
                           scree1.iloc[:, 1:].apply(np.std, axis=1)],
                            axis=1)
                rp_metrics_std_corr.columns = ['N_Components', 'Value']
                rp_metrics_std_corr['Metric'] = 'Projected_Pairwise_Distance_10_Trial_Correlation_Std'
                rp_metrics = pd.concat([rp_metrics_avg_corr, rp_metrics_std_corr])

                ### Reconstruction Error
                scree2 = pd.read_csv(f'{algorithm}/{dataset} scree2.csv')            
                # Compute average across 10 random projections
                rp_avg_reconstruction_error = pd.concat([scree2.iloc[:, 0], 
                                                 scree2.iloc[:, 1:].apply(np.mean, axis=1)],
                axis=1)
                rp_avg_reconstruction_error.columns = ['N_Components', 'Value']
                # Compute Std. across 10 random projections
                rp_avg_reconstruction_error['Metric'] = 'Reconstruction_Error'
                rp_std_reconstruction_error = pd.concat([scree2.iloc[:, 0], 
                           scree2.iloc[:, 1:].apply(np.std, axis=1)],
                            axis=1)
                rp_std_reconstruction_error.columns = ['N_Components', 'Value']
                rp_std_reconstruction_error['Metric'] = 'Reconstruction_Error_Std'
                rp_reconstruction_error_metrics = pd.concat([rp_avg_reconstruction_error, rp_std_reconstruction_error],
                                                           sort=True)  

                alg_metrics = pd.concat([rp_metrics, rp_reconstruction_error_metrics], sort=True) 

            alg_metrics = alg_metrics.reset_index(drop=True)
            alg_metrics['Data_Perspective'] = algorithm
            alg_metrics['Dataset'] = dataset

            column_order = ['Data_Perspective', 'Dataset',  'N_Components', 'Metric', 'Value', ]

            return alg_metrics[column_order]

        else:
            print(f'No Component Metrics found for {algorithm} and {dataset}')
            
    component_quality_df = [pull_cluster_quality_metrics(algorithm, 'Cars') 
                            for algorithm 
                            in ['ICA', 'PCA', 'RP']] + [pull_cluster_quality_metrics(algorithm, 'Madelon')
                                                                      for algorithm 
                                                                      in ['ICA', 'PCA', 'RP']]



    component_quality_df = pd.concat(component_quality_df).reset_index(drop=True)

    component_quality_df.to_hdf('results/results.hdf', key='component_quality', complib='blosc',complevel=9)
    component_quality_df.to_csv('results/component_quality.csv', index=False)
    
    
    print('\n\nOutputted clustering metrics, tsne projections and grid search results (on dimensionality reduction and clusters)'
         'to results.results.hdf using keys "clustering", "tsne", "grid_search", and "component_quality" \n\n'
         )
if __name__ == "__main__":
    main()

