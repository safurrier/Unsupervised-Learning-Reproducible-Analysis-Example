Unsupervised-Learning-Reproducible-Analysis-Example
==============================
Author: Alex Furrier

This is an example repo for a data science project focused
on unsupervised learning analysis. End analysis for the results can be 
found in the root folder under 'analysis.pdf'.

This example takes datasets from the
UCI machine learning lab repository and examines the performance of 
three unsupervised data learning algorithms, one feature selection method and
two clustering methods. The datasets provide 
contrasting problems: one a high dimensional, low signal dataset 
with continuous features (madelon) and the other a low dimensional,
high signal dataset with multi-level categoric data (cars). 


The three unsupervised learning algorithms examined are:

	* Principal Component Analysis (PCA)
	
	* Independent Component Analysis (ICA)

	* Random Projection (RP)

The feature selection method examined is:

	* Random Forest Classifer Feature Importance Selection
	
The clustering methods examined are:

	* K-Means
	
	* Expectation Maximization (Gaussian Mixture Model)

All clustering and projection algorithms were implemented via the python machine learning package sci-kit learn. 

For each feature transformation algorithm, both datasets were transformed across a varying number of dimensions. 

For each dimension size, two metrics were computed: one a quality of component metric and the other the reconstruction error 
measured by projecting the components back to the original dimensions and measuring squared pairwise distance. 

A 5 CV grid search across varying learning rates, hidden layer sizes, and projection components used was performed on each datasets. 

Based on subjective selection for the ‘best’ number of components, each projection was selected to be used for creating clusters. Using the
'best' number of components the data was projected and visualized in two dimensions using data was using T-SNE.

Both the original (BASE in the code) data and projected data was used for clustering the data.

Clusters across varying number of cluster sizes were computed using K-Means and Expectation Maximization (implemented as a Gaussian Mixture Model). 
For each cluster method were measured using three metrics: a cluster quality metric (SSE for K-Means and likelihood for EM), 
mutual information with the target class and accuracy when used for classification. Clustering predictions were found by taking the cluster 
label and assigning as a prediction to all points in that cluster the majority target class label. 
The clusters were used as features in a 5 CV grid search across varying learning rates, hidden layer sizes, and 
projection components used was performed on each datasets. 

To create the necesary conda env to reproduce the analysis, run 

> . ./setup_env.sh

To run all analysis steps (detailed below), enter the folloiwng command.

WARNING: This is computationally expensive and not parallelized. It will likely
take a long time to run. 

> make analysis

Warning: the above is computationally expensive and can take a great deal
of time. 


All scripts are contained in the top level folder and can be run sequentially 
following the steps enumerated below. 

Submission analysis is in top level folder in file 'analysis.pdf'

Steps to run. 

1) Run parse.py to parse and load the Madelon and Cars data appropriately
2) Run 'benchmark.py' and 'clustering.py' to generate K Means and EM (Gaussian Mixture Models) for each of the datasets (Cars and Madelon)
Depending on the script argument (BASE, ICA, PCA, etc) it will train these clustering algorithms using that data and output the following files
with different metrics on the clusters:

* SSE
    * The SSE scoring for the K Means clustering for Cars and Madelon
* loglikelihood
    * The loglikelihood scoring for the EM (Guassian Mixture Modelling) clustering for Cars and Madelon    
* {dataset} acc
    * The accuracy of the predictions using the clusters as features 
    * The way the accuracy is computed:
        * For each cluster label, the majority class label for observations of that cluster are chosen as predictions
        * Accuracy is simple accuracy sum(prediction == true_label)/# of true labels
    * Index is clustering method, columns are number of clusters, values are accuracy
* {dataset} adjMI
    * The adjusted mutual information between the cluster labels and the target class.
    * See [sklearn docs](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html) for details
    * Adjusted for chance in that clusters with higher counts usually have higher MI
    * Actual Metric is AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]
    * Index is clustering method, columns are number of clusters, values are adjusted mutual information
* {dataset} GMM
    * The NN grid search results using the EM (GaussianMixtureModel) clusters. Includes fit time, mean cv score and the hyperparameters for 
    the NN (alpha, hidden layer sizes) and clustering (number of components)
* {dataset} Kmeans
    * The NN grid search results using the EM (GaussianMixtureModel) clusters. Includes fit time, mean cv score and the hyperparameters for 
    the NN (alpha, hidden layer sizes) and clustering (number of components)    
* {dataset} 2D
    * TSNE projection of the dataset into 2 dimensions. To visualize on human understandable 2 dimensions. Default hyper parameters
    
3) Run each dimension reduction/unsupervised learning script to produce files with different metrics based on which algorithm was run:

* PCA
    * {dataset} scree.csv
        * The first column is the rank of the compoents
        * Second column is eigvenvalue (aka the amount of explained variance by that component)
    * {dataset} scree2.csv
        * N_Components is the number of the components used in the projection method
        * Reconstruction error is computed by:
                1) Taking the Comp (Moore-Penrose) pseudo-inverse of the projection matrix.
                2) Using the psuedo inverse to reconstruct the matrix with the projection:
                    * (psinv dot product with projection) and taking the dot product of that with the original data X
                3) Reconstruction error is the (original data - reconstructed)^2         
    * {dataset} dim red.csv
        * NN grid search results over number of components used for PCA    
        
* ICA
    * {dataset} scree.csv
        * The first column is the number of the compoents
        * Second column is mean absolute kurtosis of those components. The higher the kurtosis the more non-Gaussian the component is.
    * {dataset} scree2.csv
        * N_Components is the number of the components used in the projection method
        * Reconstruction error is computed by:
                1) Taking the Comp (Moore-Penrose) pseudo-inverse of the projection matrix.
                2) Using the psuedo inverse to reconstruct the matrix with the projection:
                    * (psinv dot product with projection) and taking the dot product of that with the original data X
                3) Reconstruction error is the (original data - reconstructed)^2       
    * {dataset} dim red.csv
        * NN grid search results over number of components used for ICA
        
* RP
    This script does a random sparse projection and then takes the correlation between the euclidean pairwise distances of the random projection and the original data
    * {dataset} scree1.csv
        * The first column/index is the number of the components used in the random projection
        * Columns 2-10 are 10 iterations of the correlation between the euclidean pairwise distances of the random projection and the original data
        * Avg Columns 2-10 to get an avg correlation between pairwise euclidean distance
        * 10 runs done to reduce variance from the random part of random projection
    * {dataset} scree2.csv
        * The first column/index is the number of the components used in the random projection
        * Columns 2-10 are 10 iterations of the reconstruction error of the projected data
            * Reconstruction error is computed by:
                1) Taking the Comp (Moore-Penrose) pseudo-inverse of the projection matrix.
                2) Using the psuedo inverse to reconstruct the matrix with the projection:
                    * (psinv dot product with projection) and taking the dot product of that with the original data X
                3) Reconstruction error is the (original data - reconstructed)^2 

        * Avg Columns 2-10 to get an avg correlation between pairwise euclidean distance
        * 10 runs done to reduce variance from the random part of random projection        
    * {dataset} dim red.csv
        * NN grid search results over number of components used for RP        
        
* RF (Random Forest Feature Selection) Uses a random forest based feature selection. Feature importance is based on the number
of splits a feature is involved along with its average information gained across the many bagged trees constructed
    * {dataset} scree.csv
        * The first column is the rank of the Random Forest selected feature importance
        * Second column is the feature importance score
    * {dataset} dim red.csv
        * NN grid search results over number of features based on rank (e.g. 5 will be the 5 most important features selected)
        * Number of features is denoted by column param__filter_n
        
Each of the three unsupervised learning techniques and RF raises after the grid search.
At this point proceed to step four:

4) Run pull_best_n_components.py to find the optimal number of components for each unsupervised learning/feature transformation technique. By default the criteria for this is the number of components that resulted in the highest out of fold test accuracy using 10 fold cross validation.  This will load the transformation using the ideal number of components (find from grid search, likely the optimal CV score from the NN using that number of components). It is suggested that these optimal components be taken into consideration as well as their respective quality metrics (Correlation, Avg kurtosis, variance explained etc). Set the optimal number of components as dims in the 'optimal_projections.py' script.

5) Run optimal_projections.py to output the best projections. This will record the optimal projection of the data using the number of components specified in step 4.

6) Run 'tsne.py' to create two dimensional embeddings of the optimally projected data. If so desired, run 'tsne_perplexity_search.py' to output the KL-divergence of the t-sne proejctions across a wide range of the parameter 'perplexity', which approximates the density of neighbourhoods in high dimensional space. 

7) Once the ideal number of components for PCA/ICA/RP have been set and created, run the clustering.py with the first argument as PCA/ICA/RP to generate the datasets from step 1 but with PCA/ICA/RP. This can be done in parallel, or by running the shell script 'cluster_projections.sh' (iterative)

8) To output all results in tidy data format in a single hdf table, run clean_results.py. The metrics for the results are broken down into 3 categories: clustering, which has metrics for cluster validation on the base data as well as clusters generated using UL algorithms, tsne for visualizing high dimensional clusters in a low dimensional settings, and grid_search which has the grid search results across the base data, clusters from base data, UL projected data and clusters made using the projected data.

9) The notebook in the notebooks folder '04-Analysis.ipynb' contains the code used to generate all plots used in the analysis file. 


Project Organization
------------

    ├── analysis.pdf                <- Project Analysis
    ├── Makefile                    <- Makefile with commands like `make data` or `make analysis`
    ├── README.md                   <- The top-level README for developers using this project.
    ├── data         
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   └── raw                     <- The original data. Mostly contains datasets from UCI
    │         
    ├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                  description, e.g. `01.0-initial-data-exploration`.
    │         
    ├── results                     <- Tables containing results of running unsupervised learning / clustering scripts. 
    │         
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                  generated with `pip freeze > requirements.txt`
	|         
    ├── setup_env.sh                    <- Script to initiailize git repo, setup a conda virtual environment  
    │                                  and install dependencies.
    │                          
    ├── helpers.py                  <- Utility code for various purposes and packages
    ├── benchmark.py                <- Create benchmark modelling results by NN grid search across both datasets
    ├── clustering.py               <- Script to generate K-Means and GMM clusters and NN grid search across a variety of N clusters.
    │                                  Run with one argument, the unsupervised learning/feature selection method. E.g. clustering PCA
    ├── PCA/ICA/RP/RF.py            <- Script to transform the data using named method across variety of N components. 
    ├── optimal_projections.py      <- Script to to project the data using specified optimal components determined by analysis
    ├── pull_best_n_components.py   <- Find the best number of components for PCA/ICA/RP/RF by finding the max test score in each grid search.
    ├── tsne.py                     <- Output T-SNE projections of each unsupervised learning method
    ├── tsne_perplexity_search.py   <- Output output the KL-divergence of the t-sne proejctions across a wide range of the parameter 'perplexity'



--------
Learner training and evaluation done using pandas, numpy, and scikit-learn.

Visualizations done with altair. 

Code for model training and model evaluation adapted from Jonathan Tay (https://github.com/JonathanTay) 



