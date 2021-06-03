# GDL
Python code for the paper [Online Graph Dictionary Learning]{https://arxiv.org/abs/2102.06555}.

**Abstract**
*Dictionary learning is a key tool for representation learning, that explains the data as linear combination of few basic elements. Yet, this analysis is not amenable in the context of graph learning, as graphs usually belong to different metric spaces. We fill this gap by proposing a new online Graph Dictionary Learning approach, which uses the Gromov Wasserstein divergence for the data fitting term. In our work, graphs are encoded through their nodes’ pairwise relations and modeled as convex combination of graph atoms, i.e. dictionary elements, estimated thanks to an online stochastic algorithm, which operates on a dataset of unregistered graphs with potentially different number of nodes. Our approach naturally extends
to labeled graphs, and is completed by a novel upper bound that can be used as a fast approximation of Gromov Wasserstein in the embedding space. We provide numerical evidences showing the interest of our approach for unsupervised embedding of graph datasets and for online graph subspace estimation and tracking.*


This repository contains implementations of our methods  which led to results detailed in numerical experiments. Namely vanilla GDL for unlabeled, its extension where we simultaneously learn graphs structure and their nodes distirbution, and its extension to labeled graphs as 
detailed in the supplementary material of the paper.


**Prerequisites**

- python >= 3.7.7
- numpy >= 1.18.5
- pandas >= 1.0.5
- networkx >= 2.4
- scikit-learn >= 0.24.0
- scikit-learn-extra >= 0.1.0b2
- scipy >= 1.5.0
- joblib == 0.15.1 




