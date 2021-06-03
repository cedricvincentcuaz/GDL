#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cvincentcuaz
"""

from tqdm import tqdm
import numpy as np
import dataloader
import os
import ot
import GDL_utils as gwu

#%%



def gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun='square_loss',
                       max_iter=1000, tol=1e-9, verbose=False,  init_C=None):
    """
    Returns the gromov-wasserstein barycenters of S measured similarity matrices
    (Cs)_{s=1}^{s=S}
    The function solves the following optimization problem with block
    coordinate descent:
    .. math::
        C = argmin_C\in R^NxN \sum_s \lambda_s GW(C,Cs,p,ps)
    Where :
    - Cs : metric cost matrix
    - ps  : distribution
    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S np.ndarray of shape (ns, ns)
        Metric cost matrices
    ps : list of S np.ndarray of shape (ns,)
        Sample weights in the S spaces
    p : ndarray, shape (N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the S spaces' weights
    loss_fun :  tensor-matrix multiplication function based on specific loss function
    update : function(p,lambdas,T,Cs) that updates C according to a specific Kernel
             with the S Ts couplings calculated at each iteration
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0).
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | ndarray, shape(N,N)
        Random initial value for the C matrix provided by user.
    Returns
    -------
    C : ndarray, shape (N, N)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    S = len(Cs)

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        # XXX : should use a random state and not use the global seed
        xalea = np.random.randn(N, 2)
        C = ot.utils.dist(xalea, xalea)
        C /= C.max()
    else:
        C = init_C

    iter_= 0
    prev_loss = 10**9
    curr_loss = 10**8
    best_loss = np.inf
    
    convergence_criterion=np.inf
    
    while (convergence_criterion> tol) and (iter_ < max_iter):
        Cprev = C
        prev_loss=curr_loss
        res = [gwu.np_GW2(Cs[s], C) for s in range(S)]
        T = [x[1] for x in res]
        dists= [x[0] for x in res]
        curr_loss = np.mean(dists)
        if curr_loss<best_loss:
            best_loss=curr_loss
            best_C = Cprev
        C = ot.gromov.update_square_loss(p, lambdas, T, Cs)

        convergence_criterion= np.abs(prev_loss - curr_loss)/np.abs(prev_loss)
        
        iter_+=1
    return best_C


class GW_Kmeans():
    """
    Our implementation of GW kmeans algorithm 
    """
    def __init__(self,
                 dataset_name:str, 
                 mode:str, 
                 experiment_repo:str, 
                 experiment_name:str,data_path:str='../data/'):
        self.experiment_repo= experiment_repo
        self.experiment_name = experiment_name
        print('dataset_name:', dataset_name)
        str_to_method = {'ADJ': 'adjacency', 'SP':'shortest_path','LAP':'laplacian'}
            
        if dataset_name in  ['mutag','cuneiform', 'enzymes', 'ptc','bzr','cox2','protein','imdb-b','imdb-m','imdb-b','nci1']:
            self.dataset_name= dataset_name
            self.mode = mode
            X,self.y=dataloader.load_local_data(data_path,dataset_name)
                
            if self.mode in str_to_method.keys():
                self.C_target = [np.array(X[t].distance_matrix(method=str_to_method[mode]),dtype=np.float64) for t in range(X.shape[0])]
            else:
                raise 'unknown graph representation for FGW dictionary learning'
        else:
            raise 'unknown dataset '
        
        self.shapes = np.array([x.shape[0] for x in self.C_target])
        self.T = len(self.C_target)
        
    def training(self, 
                 n_centroids:int, 
                 centroids_shape:int, 
                 tol:float=0.001, 
                 max_iter:int=300,
                 seed:int=0,
                 empty_cluster:bool=True,
                 max_fails:int = 10):
        """
        Find the clusters of kmeans algorithm

        Parameters
        ----------
        n_centroids : int - Number of kmeans centroids to look for
        centroids_shape : int - Fix the number of nodes for graphs centroids at the same value
        tol : float, optional
            DESCRIPTION. The default is 0.001.
        max_iter : int, optional
            DESCRIPTION. The default is 300.
        seed : int, optional
            DESCRIPTION. The default is 0.
        empty_cluster : bool, optional
            Use a random sampling strategy if there are clusters which become empty during training. 
            The default is True.
        max_fails : int, optional
            Maximum Number of updates which lead to a less discriminant clustering. The default is 10.
        """
        self.settings={'n_centroids':n_centroids, 'centroids_shape': centroids_shape,
                       'tol':tol, 'max_iter':max_iter,'seed':seed,'sampling_mode':'kmeans++','empty_cluster':empty_cluster}
        print('check for empty cluster: ', empty_cluster)
        self.n_centroids= n_centroids
        self.centroids_shape= centroids_shape
        self.seed=seed
        np.random.seed(self.seed)
        self.init_centroids()
        iter_ =0
        #Evaluate initial configuration
        init_loss = self.compute_assignment()
        # if we are forced to get a very bad initialization because chosen shapes
        # check check for empty assignments right away
        self.loss=[init_loss]
        convergence_criterion = np.inf
        prev_loss = np.inf
        curr_loss = init_loss
        best_loss = curr_loss
        iter_ = 0
        while (convergence_criterion > tol) and (iter_<max_iter):
            prev_loss = curr_loss
            self.update_centroids(bar_tol=tol)
            if not empty_cluster:
                curr_loss=self.compute_assignment()
            else:
                print('check for empty cluster')
                curr_loss= self.compute_assignment()
                # Create centroids if there are empty clusters
                check = (len(np.unique(self.assignments)) == self.n_centroids)
                attempt=0
                max_attemps=20
                while (check==False and attempt<max_attemps):
                    #print('found empty cluster - attempt = %s'%attempt)
                    curr_loss = self.handle_empty_cluster()
                    #print('new curr loss = ', curr_loss)
                    check = (len(np.unique(self.assignments)) == self.n_centroids)
                    #print('check =',check)
                    attempt+=1
                print('Succesfully handled empty clusters')
            convergence_criterion = np.abs(prev_loss - curr_loss)/np.abs(curr_loss)
            iter_+=1
            self.loss.append(curr_loss)
            #print('step= %s / loss = %s / convergence_criterion = %s'%(iter_,curr_loss, convergence_criterion))
            if curr_loss<best_loss:
                fails =0
                self.save_elements()
            else:
                fails+=1
                if fails >max_fails:
                    break
                
    def handle_empty_cluster(self):
        init_cluster_comp= np.unique(self.assignments, return_counts=True)
        filled_clusters = init_cluster_comp[0]
        size_clusters = init_cluster_comp[1]
        for k in range(self.n_centroids):
            if not (k in filled_clusters):
                # we look for a graph with the proper size among the biggest clusters
                found =False
                local_iter = 0
                while not found:
                    sorted_size_clusters = np.argsort(size_clusters)[::-1]
                    #print('size_clusters sorted: ', size_clusters[sorted_size_clusters])
                    curr_biggest = filled_clusters[sorted_size_clusters[local_iter]]
                    #print('curr_biggest = %s / cluster size = %s'%(curr_biggest, size_clusters[sorted_size_clusters[local_iter]]))
                    idx_cluster = [idx for idx in range(len(self.assignments)) if (self.assignments[idx]==curr_biggest and self.shapes[idx]==self.centroids_shape)]
                    if len(idx_cluster)>0:
                        #print('found in biggest cluster :', curr_biggest)
                        sampled_idx = np.random.choice(idx_cluster)
                        self.assignments[sampled_idx] = k
                        found=True
                        #print('previous cluster size = ', size_clusters[sorted_size_clusters[local_iter]])
                        size_clusters[sorted_size_clusters[local_iter]]-=1
                        #print('new cluster size = ', size_clusters[sorted_size_clusters[local_iter]])
                    else:
                        #print('not found in biggest cluster:', curr_biggest)
                        local_iter+=1
                self.C_centroids[k] = self.C_target[sampled_idx]
                print('centroids update')
        loss= self.compute_assignment()
        return loss
    
    def update_centroids(self,bar_tol=10**(-5)):
        
        for k in range(self.n_centroids):
            idx_k = np.argwhere(self.assignments == k )[:,0]
            list_Cs = [self.C_target[i] for i in idx_k]
            list_ps = [np.ones(self.shapes[i])/self.shapes[i] for i in idx_k]
            p = np.ones(self.centroids_shape)/self.centroids_shape
            lambdas = np.ones(len(idx_k))/len(idx_k)
            
            print('computing barycenter - cluster = %s / #graphs= %s'%(k,len(idx_k)))
            C=gromov_barycenters(self.centroids_shape, list_Cs, list_ps,p,lambdas,init_C=None,verbose=False,tol=bar_tol,max_iter =100)
            self.centroids[k]=C
            
            
    def compute_assignment(self):
        loss = 0
        for i in tqdm(range(self.T),desc='computing assignment'):
            
            local_dist = []
            for k in range(self.n_centroids):
                local_dist.append(gwu.np_GW2(self.C_target[i], self.centroids[k])[0])
            idx_min_dist = np.argmin(local_dist)    
            self.assignments[i] = idx_min_dist
            loss+= local_dist[idx_min_dist]
        return loss
    
    def init_centroids(self,seed=0,subsampling_size=100):
        """ Initialize centroids by kmeans++ strategy on a subsample of the dataset"""
        print('initializing centroids')
        self.centroids_idx = []
        sampled_idx = np.random.choice(range(self.T), size = np.min(subsampling_size,self.T),replace=False )
        self.D = np.zeros((subsampling_size,subsampling_size))
        for i in tqdm(range(subsampling_size),desc='computing pairwise GW matrix for init'):
            for j in range(i+1, subsampling_size):
                self.D[i,j] =  self.D[j,i] = gwu.np_GW2(self.C_target[sampled_idx[i]],self.C_target[sampled_idx[j]])[0]
        picked_centroids = 0
        centroids_subidx=[]
        while picked_centroids< self.n_centroids:
            if picked_centroids ==0:
                subidx = np.random.choice(range(subsampling_size))
                centroids_subidx.append(subidx)
                self.centroids_idx.append(sampled_idx[subidx])
            else:
                dist_distrib = self.D[centroids_subidx[-1],:]
                dist_distrib[dist_distrib<10**(-15)]=0
                p = dist_distrib/ np.sum(dist_distrib)
                subidx = np.random.choice(range(subsampling_size), p=p)
                centroids_subidx.append(subidx)
                self.centroids_idx.append(sampled_idx[subidx])
            picked_centroids+=1
        self.assignments = np.zeros(len(self.C_target))
        self.prev_centroids =[None]* self.n_centroids
        
    def save_elements(self,save_settings=False):
            
        path = os.path.abspath('../')+self.experiment_repo
        print('path',path)
        if not os.path.exists(path+self.experiment_name):
            os.makedirs(path+self.experiment_name)
            print('made dir', path+self.experiment_name)
        
        np.save(path+'%s/loss.npy'%self.experiment_name, np.array(self.loss))
        np.save(path+'%s/centroids.npy'%self.experiment_name, np.array(self.centroids))
        np.save(path+'%s/assignments.npy'%self.experiment_name, np.array(self.assignments))
        np.save(path+'%s/init_subpairwiseGWmatrix.npy'%self.experiment_name, self.D)
        

class GW_Kmeans_extendedDL():
    """
    Our implementary of kmeans algorithm when masses are not fixed to uniform distribuiton
    Barycenters masses are still fixed to uniform distributions.
    """
    def __init__(self,
                 list_C, 
                 list_h, 
                 labels,
                 experiment_repo, 
                 experiment_name,data_path='../data/'):
        self.experiment_repo= experiment_repo
        self.experiment_name = experiment_name
        self.C_target = list_C
        self.h_target=list_h
        self.mode='ADJ'
        self.y=labels
        
        self.shapes = np.array([x.shape[0] for x in self.C_target])
        self.T = len(self.C_target)
        
    
    def training(self, n_centroids:int, 
                 centroids_shape:int, 
                 tol:float=0.001, 
                 max_iter:int=300,
                 seed:int=0,
                 empty_cluster:bool=True,
                 max_fails:int = 10):
        """
        Find the clusters of kmeans algorithm

        Parameters
        ----------
        n_centroids : int - Number of kmeans centroids to look for
        centroids_shape : int - Fix the number of nodes for graphs centroids at the same value
        tol : float, optional
            DESCRIPTION. The default is 0.001.
        max_iter : int, optional
            DESCRIPTION. The default is 300.
        seed : int, optional
            DESCRIPTION. The default is 0.
        empty_cluster : bool, optional
            Use a random sampling strategy if there are clusters which become empty during training. 
            The default is True.
        max_fails : int, optional
            Maximum Number of updates which lead to a less discriminant clustering. The default is 10.
        """
        self.settings={'n_centroids':n_centroids, 'centroids_shape': centroids_shape,
                       'tol':tol, 'max_iter':max_iter,'seed':seed,'empty_cluster':empty_cluster}
        print('check for empty cluster: ', empty_cluster)
        self.n_centroids= n_centroids
        self.centroids_shape= centroids_shape
        self.seed=seed
        np.random.seed(self.seed)
        self.init_centroids()
        iter_ =0
        #Evaluate initial configuration
        init_loss = self.compute_assignment()
        self.loss=[init_loss]
        convergence_criterion = np.inf
        prev_loss = np.inf
        curr_loss = init_loss
        best_loss = curr_loss
        iter_ = 0
        while (convergence_criterion > tol) and (iter_<max_iter):
            prev_loss = curr_loss
            self.update_centroids(bar_tol=tol)
            if not empty_cluster:
                curr_loss=self.compute_assignment()
            else:
                #print('check for empty cluster')
                curr_loss= self.compute_assignment()
                # Create centroids if there are empty clusters
                check = (len(np.unique(self.assignments)) == self.n_centroids)
                max_attempts = 20
                attempt=0
                while (check==False and attempt<max_attempts):
                    #print('found empty cluster - attempt = %s'%attempt)
                    curr_loss = self.handle_empty_cluster()
                    #print('new curr loss = ', curr_loss)
                    check = (len(np.unique(self.assignments)) == self.n_centroids)
                    #print('check =',check)
                    attempt+=1
                #print('Succesfully handled empty clusters')
            convergence_criterion = np.abs(prev_loss - curr_loss)/np.abs(curr_loss)
            iter_+=1
            self.loss.append(curr_loss)
            #print('step= %s / loss = %s / convergence_criterion = %s'%(iter_,curr_loss, convergence_criterion))
            if curr_loss<best_loss:
                fails =0
                self.save_elements()
            else:
                fails+=1
                if fails >max_fails:
                    break
    def handle_empty_cluster(self):
        #print('Handling empty cluster')
        init_cluster_comp= np.unique(self.assignments, return_counts=True)
        filled_clusters = init_cluster_comp[0]
        size_clusters = init_cluster_comp[1]
        for k in range(self.n_centroids):
            if not (k in filled_clusters):
                # we look for a graph with the proper size among the biggest clusters
                
                found =False
                local_iter = 0
                while not found:
                    sorted_size_clusters = np.argsort(size_clusters)[::-1]
                    #print('size_clusters sorted: ', size_clusters[sorted_size_clusters])
                    curr_biggest = filled_clusters[sorted_size_clusters[local_iter]]
                    #print('curr_biggest = %s / cluster size = %s'%(curr_biggest, size_clusters[sorted_size_clusters[local_iter]]))
                    idx_cluster = [idx for idx in range(len(self.assignments)) if (self.assignments[idx]==curr_biggest and self.shapes[idx]==self.centroids_shape)]
                    if len(idx_cluster)>0:
                        #print('found in biggest cluster :', curr_biggest)
                        
                        sampled_idx = np.random.choice(idx_cluster)
                        self.assignments[sampled_idx] = k
                        found=True
                        #print('previous cluster size = ', size_clusters[sorted_size_clusters[local_iter]])
                        size_clusters[sorted_size_clusters[local_iter]]-=1
                        #print('new cluster size = ', size_clusters[sorted_size_clusters[local_iter]])
                    else:
                        #print('not found in biggest cluster:', curr_biggest)
                        local_iter+=1
                self.C_centroids[k] = self.C_target[sampled_idx]
                print('centroids update')
        loss= self.compute_assignment()
        return loss
            
    
    def update_centroids(self,bar_tol=10**(-5)):
        
        for k in range(self.n_centroids):
            idx_k = np.argwhere(self.assignments == k )[:,0]
            list_Cs = [self.C_target[i] for i in idx_k]
            list_ps = [self.h_target[i] for i in idx_k]
            p = np.ones(self.centroids_shape)/self.centroids_shape
            lambdas = np.ones(len(idx_k))/len(idx_k)
            C=gromov_barycenters(self.centroids_shape, list_Cs, list_ps,p,lambdas,init_C=None,verbose=False,tol=bar_tol,max_iter =300)
            self.centroids[k]=C
            
            
    def compute_assignment(self):
        loss = 0
        print('computing assignment')
        for i in tqdm(range(self.T)):
            
            local_dist = []
            for k in range(self.n_centroids):
                local_dist.append(gwu.np_GW2(self.C_target[i], self.centroids[k],p=self.h_target[i])[0])
            idx_min_dist = np.argmin(local_dist)    
            self.assignments[i] = idx_min_dist
            loss+= local_dist[idx_min_dist]
        return loss
    
    def init_centroids(self,seed=0,subsampling_size=100):
        """ Initialize centroids by kmeans++ strategy on a subsample of the dataset"""
        print('initializing centroids')
        self.centroids_idx = []
        sampled_idx = np.random.choice(range(self.T), size = np.min(subsampling_size,self.T),replace=False )
        self.D = np.zeros((subsampling_size,subsampling_size))
        for i in tqdm(range(subsampling_size),desc='computing pairwise GW matrix for init'):
            for j in range(i+1, subsampling_size):
                self.D[i,j] =  self.D[j,i] = gwu.np_GW2(self.C_target[sampled_idx[i]],self.C_target[sampled_idx[j]], p=self.h_target[sampled_idx[i]], q=self.h_target[sampled_idx[j]])[0]
        picked_centroids = 0
        centroids_subidx=[]
        while picked_centroids< self.n_centroids:
            if picked_centroids ==0:
                subidx = np.random.choice(range(subsampling_size))
                centroids_subidx.append(subidx)
                self.centroids_idx.append(sampled_idx[subidx])
            else:
                dist_distrib = self.D[centroids_subidx[-1],:]
                dist_distrib[dist_distrib<10**(-15)]=0
                p = dist_distrib/ np.sum(dist_distrib)
                subidx = np.random.choice(range(subsampling_size), p=p)
                centroids_subidx.append(subidx)
                self.centroids_idx.append(sampled_idx[subidx])
            picked_centroids+=1
        self.assignments = np.zeros(len(self.C_target))
        self.prev_centroids =[None]* self.n_centroids
        
    def save_elements(self,save_settings=False):
            
        path = os.path.abspath('../')+self.experiment_repo
        print('path',path)
        if not os.path.exists(path+self.experiment_name):
            os.makedirs(path+self.experiment_name)
            print('made dir', path+self.experiment_name)
        
        np.save(path+'%s/GWkmeans_loss.npy'%self.experiment_name, np.array(self.loss))
        np.save(path+'%s/GWkmeans_centroids.npy'%self.experiment_name, np.array(self.centroids))
        np.save(path+'%s/GWassignments.npy'%self.experiment_name, np.array(self.assignments))
        np.save(path+'%s/init_subpairwiseGWmatrix.npy'%self.experiment_name, self.D)
        



# =============================================================================
# FGW Kmeans
# =============================================================================
def fgw_barycenters(N, Ys, Cs, ps, lambdas, alpha, p=None, max_iter=100, tol=1e-5,
                    verbose=False, log=False, init_C=None, init_X=None):
    """Compute the fgw barycenter as presented eq (5) in [24].
    Parameters
    # @ POT github
    ----------
    N : integer
        Desired number of samples of the target barycenter
    Ys: list of ndarray, each element has shape (ns,d)
        Features of all samples
    Cs : list of ndarray, each element has shape (ns,ns)
        Structure matrices of all samples
    ps : list of ndarray, each element has shape (ns,)
        Masses of all samples.
    lambdas : list of float
        List of the S spaces' weights
    alpha : float
        Alpha parameter for the fgw distance
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0).
    init_C : ndarray, shape (N,N), optional
        Initialization for the barycenters' structure matrix. If not set
        a random init is used.
    init_X : ndarray, shape (N,d), optional
        Initialization for the barycenters' features. If not set a
        random init is used.
    Returns
    -------
    X : ndarray, shape (N, d)
        Barycenters' features
    C : ndarray, shape (N, N)
        Barycenters' structure matrix
    
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    S = len(Cs)
    d = Ys[0].shape[1]  # dimension on the node features
    if p is None:
        p = np.ones(N) / N

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    Ys = [np.asarray(Ys[s], dtype=np.float64) for s in range(S)]

    lambdas = np.asarray(lambdas, dtype=np.float64)

    if init_C is None:
        xalea = np.random.randn(N, 2)
        C = ot.utils.dist(xalea, xalea)
    else:
        C = init_C

    if init_X is None:
        X = np.zeros((N, d))
    else:
        X = init_X

    T = [np.outer(p, q) for q in ps]
    
    iter_= 0
    prev_loss = 10**9
    curr_loss = 10**8
    best_loss = np.inf
    
    convergence_criterion=np.inf
    while((convergence_criterion>tol) and iter_< max_iter):
        prev_loss=curr_loss
        Ys_temp = [y.T for y in Ys]
        X = ot.gromov.update_feature_matrix(lambdas, Ys_temp, T, p).T
        T_temp = [t.T for t in T]
        C = ot.gromov.update_sructure_matrix(p, lambdas, T_temp, Cs)
        res = [ gwu.numpy_FGW_loss(C, Cs[s],X,Ys[s],alpha=alpha, features_dist='l2', OT_loss='square_loss') for s in range(S)]
        dists = [x[0] for x in res]
        T = [x[1] for x in res]
        curr_loss = np.mean(dists)
        if curr_loss< best_loss:
            best_X= X
            best_C=C
        convergence_criterion = np.abs(prev_loss -curr_loss)/np.abs(curr_loss)
        iter_+=1
    return best_X, best_C

class FGW_Kmeans():
    def __init__(self,dataset_name, experiment_repo, experiment_name, mode, alpha,data_path='../data/'):
        assert dataset_name in ['mutag', 'enzymes', 'ptc',
                                'bzr','cox2','protein']
        self.dataset_name= dataset_name
        self.experiment_repo = experiment_repo
        self.experiment_name =experiment_name
        self.mode = mode
        str_to_method = {'ADJ': 'adjacency', 'SP':'shortest_path','LAP':'laplacian'}        
        
        if self.dataset_name in ['mutag', 'ptc']:
            X,self.y=dataloader.load_local_data(data_path,dataset_name,one_hot=True)
        elif self.dataset_name in ['cuneiform', 'enzymes','protein','cox2','bzr']:
            X,self.y=dataloader.load_local_data(data_path,dataset_name,one_hot=True)
            assert self.full_features[0] in ['raw']
        if self.mode in str_to_method.keys():
            self.C_target = [np.array(X[t].distance_matrix(method=str_to_method[mode]),dtype=np.float64) for t in range(X.shape[0])]
        self.A_target= [np.array(X[t].values(),dtype=np.float64) for t in range(X.shape[0])]

        self.shapes= np.array([X.shape[0] for X in self.C_target])
        self.alpha = alpha
        self.d = self.A_target[0].shape[-1]
        self.T = len(self.C_target)
    
    
    def training(self, n_centroids:int, 
                 centroids_shape:int, 
                 tol:float=0.001, 
                 max_iter:int=300,
                 seed:int=0,
                 empty_cluster:bool=True,
                 max_fails:int = 20):
        """
        Find the clusters of kmeans algorithm

        Parameters
        ----------
        n_centroids : int - Number of kmeans centroids to look for
        centroids_shape : int - Fix the number of nodes for graphs centroids at the same value
        tol : float, optional
            DESCRIPTION. The default is 0.001.
        max_iter : int, optional
            DESCRIPTION. The default is 300.
        seed : int, optional
            DESCRIPTION. The default is 0.
        empty_cluster : bool, optional
            Use a random sampling strategy if there are clusters which become empty during training. 
            The default is True.
        max_fails : int, optional
            Maximum Number of updates which lead to a less discriminant clustering. The default is 10.
        """
        self.settings={'n_centroids':n_centroids, 'centroids_shape': centroids_shape,
                       'tol':tol, 'max_iter':max_iter,'seed':seed,'alpha':self.alpha,'empty_cluster':empty_cluster}
        print('check for empty cluster: ', empty_cluster)
        self.n_centroids= n_centroids
        self.centroids_shape= centroids_shape
        self.seed=seed
        np.random.seed(self.seed)
        self.init_centroids()
        iter_ =0
        #Evaluate initial configuration
        init_loss = self.compute_assignment()
        self.loss=[init_loss]
        convergence_criterion = np.inf
        prev_loss = np.inf
        curr_loss = init_loss
        best_loss = curr_loss
        iter_ = 0
        while (convergence_criterion > tol) and (iter_<max_iter):
            prev_loss = curr_loss
            self.update_centroids(bar_tol=tol)
            if not empty_cluster:
                curr_loss=self.compute_assignment()
            else:
                #print('check for empty cluster')
                curr_loss= self.compute_assignment()
                # Create centroids if there are empty clusters
                check = (len(np.unique(self.assignments)) == self.n_centroids)
                max_attempts = 20
                attempt=0
                while (check==False and attempt<max_attempts):
                    #print('found empty cluster - attempt = %s'%attempt)
                    curr_loss = self.handle_empty_cluster()
                    #print('new curr loss = ', curr_loss)
                    check = (len(np.unique(self.assignments)) == self.n_centroids)
                    #print('check =',check)
                    attempt+=1
                #print('Succesfully handled empty clusters')
            convergence_criterion = np.abs(prev_loss - curr_loss)/np.abs(curr_loss)
            convergence_criterion = np.abs(prev_loss - curr_loss)/np.abs(curr_loss)
            iter_+=1
            self.loss.append(curr_loss)
            #print('step= %s / loss = %s / convergence_criterion = %s'%(iter_,curr_loss, convergence_criterion))
            if curr_loss<best_loss:
                fails =0
                self.save_elements()
            else:
                fails+=1
                if fails >max_fails:
                    break
            
    def update_centroids(self,bar_tol=10**(-5)):
        
        for k in range(self.n_centroids):
            idx_k = np.argwhere(self.assignments == k )[:,0]
            list_Cs = [self.C_target[i] for i in idx_k]
            list_As = [self.A_target[i] for i in idx_k]
            list_ps = [np.ones(self.shapes[i])/self.shapes[i] for i in idx_k]
            p = np.ones(self.centroids_shape)/self.centroids_shape
            lambdas = np.ones(len(idx_k))/len(idx_k)
            A,C = fgw_barycenters(self.centroids_shape, list_As, list_Cs, list_ps, lambdas, self.alpha, p=p, max_iter=100, tol=bar_tol,init_C=None, init_X=None)
            self.C_centroids[k]=C
            self.A_centroids[k]=A
                        
    def compute_assignment(self):
        loss = 0
        for i in tqdm(range(self.T), desc='computing cluster assignments'):
            
            local_dist = []
            for k in range(self.n_centroids):
                
                local_dist.append(gwu.numpy_FGW_loss(self.C_target[i], self.C_centroids[k],self.A_target[i], self.A_centroids[k],alpha=self.alpha, features_dist='l2', OT_loss='square_loss')[0])
            idx_min_dist = np.argmin(local_dist)    
            self.assignments[i] = idx_min_dist
            loss+= local_dist[idx_min_dist]
        return loss
    
    
    def init_centroids(self,seed=0,subsampling_size=100):
        """ Initialize centroids by kmeans++ strategy on a subsample of the dataset"""
        print('initializing centroids')
        self.centroids_idx = []
        sampled_idx = np.random.choice(range(self.T), size = np.min(subsampling_size,self.T),replace=False )
        self.D = np.zeros((subsampling_size,subsampling_size))
        for i in tqdm(range(subsampling_size),desc='computing pairwise FGW matrix for init'):
            for j in range(i+1, subsampling_size):
                self.D[i,j] =  self.D[j,i] = gwu.numpy_FGW_loss(self.C_target[sampled_idx[i]],self.C_target[sampled_idx[j]],self.A_target[sampled_idx[i]],self.A_target[sampled_idx[j]],alpha=self.alpha)[0]
        picked_centroids = 0
        centroids_subidx=[]
        while picked_centroids< self.n_centroids:
            if picked_centroids ==0:
                subidx = np.random.choice(range(subsampling_size))
                centroids_subidx.append(subidx)
                self.centroids_idx.append(sampled_idx[subidx])
            else:
                dist_distrib = self.D[centroids_subidx[-1],:]
                dist_distrib[dist_distrib<10**(-15)]=0
                p = dist_distrib/ np.sum(dist_distrib)
                subidx = np.random.choice(range(subsampling_size), p=p)
                centroids_subidx.append(subidx)
                self.centroids_idx.append(sampled_idx[subidx])
            picked_centroids+=1
        self.assignments = np.zeros(len(self.C_target))
        self.prev_centroids =[None]* self.n_centroids
        
    def handle_empty_cluster(self):
        #print('Handling empty cluster')
        init_cluster_comp= np.unique(self.assignments, return_counts=True)
        filled_clusters = init_cluster_comp[0]
        #print('# filled clusters = ', len(filled_clusters))
        size_clusters = init_cluster_comp[1]
        for k in range(self.n_centroids):
            if not (k in filled_clusters):
                # we look for a graph with the proper size among the biggest clusters
                
                found =False
                local_iter = 0
                while not found:
                    sorted_size_clusters = np.argsort(size_clusters)[::-1]
                    curr_biggest = filled_clusters[sorted_size_clusters[local_iter]]
                    idx_cluster = [idx for idx in range(len(self.assignments)) if (self.assignments[idx]==curr_biggest and self.shapes[idx]==self.centroids_shape)]
                    if len(idx_cluster)>0:
                        #print('found in biggest cluster :', curr_biggest)
                        
                        sampled_idx = np.random.choice(idx_cluster)
                        self.assignments[sampled_idx] = k
                        found=True
                        size_clusters[sorted_size_clusters[local_iter]]-=1
                        
                    else:
                        #print('not found in biggest cluster:', curr_biggest)
                        local_iter+=1
                self.C_centroids[k] = self.C_target[sampled_idx]
                self.A_centroids[k] = self.A_target[sampled_idx]
                #print('centroids update')
        loss= self.compute_assignment()
        return loss
    def save_elements(self,save_settings=False):
            
        path = os.path.abspath('../')+self.experiment_repo
        print('path',path)
        if not os.path.exists(path+self.experiment_name):
            os.makedirs(path+self.experiment_name)
            print('made dir', path+self.experiment_name)
        
        np.save(path+'%s/loss.npy'%self.experiment_name, np.array(self.loss))
        np.save(path+'%s/C_centroids.npy'%self.experiment_name, np.array(self.C_centroids))
        np.save(path+'%s/A_centroids.npy'%self.experiment_name, np.array(self.A_centroids))
        np.save(path+'%s/assignments.npy'%self.experiment_name, np.array(self.assignments))
        np.save(path+'%s/init_subpairwiseFGWmatrix.npy'%self.experiment_name, self.D)
        
