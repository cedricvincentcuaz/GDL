"""
@author: cvincentcuaz
"""

from tqdm import tqdm
import GDL_utils as gwu
import numpy as np
import random_toydatasets as rtoys

import dataloader
import os
import pandas as pd
import pylab as pl

#%%

# =============================================================================
# Gromov Wasserstein- Online Graph Dictionary Learning
#   1/  GW_Unsupervised_DictionaryLearning : Learning GDL over a fixed dataset on 
#       several epochs.
#   2/  extended_GW_Unsupervised_DictionaryLearning: Learning extended GDL over
#       a fixed dataset on several epochs. Where atoms consist in a pairwise 
#       relation matrix {C_s} and a weight vector {h_s} taking into account the relative 
#       importance of nodes
#   3/  Online_GW_Twitch_streaming: Online Learning of our GDL on the dataset Twitch - without features. 
# =============================================================================

class GW_Unsupervised_DictionaryLearning():
    """
    Graph Dictionary Learning for graph datasets without node features
    """
    def __init__(self,dataset_name:str,
                 mode:str, 
                 number_atoms:int, 
                 shape_atoms:int, 
                 experiment_repo:str, 
                 experiment_name:str,
                 data_path:str='./real_datasets/'):
        """
        Parameters
        ----------
        dataset_name : name of the dataset to experiment on.
                    To match our data loaders it is restricted to 
                    1) real datasets used for clustering experiments: ['imdb-b','imdb-m']
                    2) toy datasets from section 4.1 of the paper cf figures 2 and 3 :['balanced_clustertoy','clustertoy2C']
        mode : representations for input graphs. (e.g) 'ADJ':adjacency / 'SP': shortest path 
        number_atoms : number of atoms in our dictionary
        shape_atoms : number of nodes similar for each atom
        experiment_repo : subrepository to save results of the experiment 
        experiment_name : subrepository to save results of the experiment under the 'experiment repo' repository
        data_path : path where data is. The default is '../data/'.
        """
        
        self.experiment_repo= experiment_repo
        self.experiment_name = experiment_name
        
        str_to_method = {'ADJ': 'adjacency', 'SP':'shortest_path','LAP':'laplacian'}
        self.dataset_name= dataset_name
        self.mode = mode
        if dataset_name in ['imdb-b','imdb-m']:
            X,self.labels=dataloader.load_local_data(data_path,dataset_name)
                
            if self.mode in str_to_method.keys():
                #input datasets
                self.graphs = [np.array(X[t].distance_matrix(method=str_to_method[mode]),dtype=np.float64) for t in range(X.shape[0])]
            else:
                raise 'unknown graph representation for FGW dictionary learning' 
        elif dataset_name =='balanced_clustertoy':
            # Experiments used for figure 2 of the paper
            
            max_blocs = 3
            graph_by_group = 100
            graph_min_size, graph_max_size, graph_step_size=15,60,5
            graph_sizes = list(range(graph_min_size, graph_max_size+graph_step_size, graph_step_size))
            intra_p = 0.9
            inter_p = 0.1
            seed=0
            self.settings_toy= {'max_blocs':max_blocs, 'graph_by_group':graph_by_group,
                           'graph_min_size':graph_min_size,'graph_max_size':graph_max_size,'graph_step_size':graph_step_size,
                           'intra_p':intra_p,'inter_p':inter_p,'seed':seed}
            
            dataset_sbm = rtoys.toy_sbm3(max_blocs=max_blocs, graphs_by_group= graph_by_group, graph_sizes=graph_sizes, intra_p=intra_p, inter_p=inter_p, seed=seed)            
            self.graphs= [mat for group in range(1,max_blocs+1) for mat in dataset_sbm[group] ] 
            self.labels = np.array([bloc_number for bloc_number in range(1,max_blocs+1) for _ in range(graph_by_group)])
            if self.mode != 'ADJ':
                raise 'current version only support adjacency matrices for experiments on toy datasets'
        elif dataset_name == 'clustertoy2C':
            # Experiments used for figure 3 of the paper - 1D manifold
            
            graph_qt=150
            graph_min_size, graph_max_size, graph_step_size=15,60,5
            graph_sizes = list(range(graph_min_size, graph_max_size+graph_step_size, graph_step_size))
            cluster_perturbation = 0.25
            intra_p = 0.9
            inter_p = 0.1
            seed=0
            dataset_sbm,_= rtoys.toy_sbm2clusters_1Dinterpolation(graph_qt=graph_qt,cluster_perturbation=cluster_perturbation, graph_sizes=graph_sizes, intra_p=intra_p, inter_p=inter_p, seed=seed)            
        
            self.settings_toy= {'graph_qt':graph_qt,'graph_min_size':graph_min_size,'graph_max_size':graph_max_size,'graph_step_size':graph_step_size,
                           'intra_p':intra_p,'inter_p':inter_p,'seed':seed,'cluster_perturbation':cluster_perturbation}
            
            self.graphs= dataset_sbm
            self.labels = np.array([2]*graph_qt) # number of clusters in each graph
            
        self.dataset_size= len(self.graphs)
        self.shapes= np.array([X.shape[0] for X in self.graphs])
        self.number_atoms = number_atoms
        self.shape_atoms = shape_atoms
        
    def initialize_atoms(self,init_mode_atoms:int,seed:int):
        """
        Parameters
        ----------
        init_mode_atoms : 0= random initialization / 1= initialization by sampling in the dataset
        seed : seed of random generator to ensure reproductibility
        """
        np.random.seed(seed)
        if init_mode_atoms==0:
            # initialize atoms  with entries sampled from U([0,1])
            # Then project the random matrix on the set of symmetric matrices
            init_atoms = np.random.uniform(low= 10**(-15), high=1, size= (self.number_atoms,self.shape_atoms, self.shape_atoms))
            self.Cs = 0.5* (init_atoms + init_atoms.transpose((0,2,1))) 
        elif init_mode_atoms ==1:
            # Sample from the dataset:
            # 1. If enough samples with adequate shape exist in the dataset, we sample them randomly
            # 2. Else if samples with adequate shape exist but less than the required number, we add symmetric noise to the existing ones
            # 3. Else an exception is raised
            shape_idx = np.argwhere(self.shapes==self.shape_atoms)[:,0]
            if shape_idx.shape[0]>0:
                print('find samples with good shape')
                if shape_idx.shape[0]>self.number_atoms:
                    print('find enough samples with good shapes')
                    warmstart_idx = np.random.choice(shape_idx,size=self.number_atoms,replace=False)
                    print('SELECTED IDX FOR INITIALIZATION : ', warmstart_idx)
                    self.Cs= np.stack([self.graphs[idx] for idx in warmstart_idx])
                    
                else:
                    print('could not find enough samples with good shapes')
                    warmstart_idx = np.random.choice(shape_idx, size=self.number_atoms, replace=True)
                    #add noise to existing ones to reach the required number of atoms
                    self.Cs= np.stack([self.graphs[idx] for idx in warmstart_idx])
                    mean_ = np.mean(self.Cs)
                    std_ = np.std(self.Cs)
                    noise = np.random.normal(loc= mean_, scale = std_, size= (self.number_atoms,self.shape_atoms, self.shape_atoms ))
                    noise = 0.5* (noise+ noise.transpose((0,2,1))) #add symmetric noise
                    self.Cs+=noise
                   
            else:
                raise 'There is no graph with adequate shape in the dataset - change the value of init_mode_atoms from 1 to 0 for random initialization '
        
    def initialize_optimizer(self):
        #Initialization for our numpy implementation of adam optimizer
        self.atoms_adam_m = np.zeros((self.number_atoms, self.shape_atoms,self.shape_atoms))#Initialize first  moment vector
        self.atoms_adam_v = np.zeros((self.number_atoms, self.shape_atoms, self.shape_atoms))#Initialize second moment vector
        self.atoms_adam_count = np.zeros((self.number_atoms))
        
            
    def Stochastic_GDL(self,
                       l2_reg:float,
                       eps:float, 
                       max_iter_outer:int,
                       max_iter_inner:int,
                       lr:float,
                       batch_size:int,
                       epochs:int, 
                       algo_seed:int,
                       beta_1:float=0.9,
                       beta_2:float=0.99,
                       init_mode_atoms:int=1,
                       checkpoint_timestamp:int=None,
                       visualization_frame:int=None,
                       verbose:bool=False):
        """
        Stochastic Algorithm to learn dictionary atoms with our GW loss
        - refers to Equation 4 in the paper and algorithm 2.

        Parameters
        ----------
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter_outer : maximum number of outer loop iterations of our BCD algorithm            
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        lr : Initial learning rate of Adam optimizer
        batch_size : batch size 
        steps : number of updates of dictionaries 
        algo_seed : initialization random seed
        OT_loss : GW discrepency ground cost. The default is 'square_loss'.
        beta_1 : Adam parameter on gradient. The default is 0.9.
        beta_2 : Adam parameter on gradient**2. The default is 0.99.
        init_mode_atoms : 0= random initialization of {Cs} / 1= initialization of {Cs} by sampling in the dataset
        checkpoint_timestamp : checkpoint i.e number of epochs to run unmixing on the current dictionary state to save best model 
        visualization_frame: Can be set stricly positive below the number of epochs to plot the loss evolution periodically over epochs
        verbose : Check the good evolution of the loss. The default is False.
        """
        OT_loss = 'square_loss' # Euclidean ground cost for GW is the default for the method
        # dictionary of settings to save
        self.settings = {'number_atoms':self.number_atoms, 'shape_atoms': self.shape_atoms,
                         'eps':eps,'max_iter_inner':max_iter_inner,'max_iter_outer':max_iter_outer,
                         'lr':lr, 'epochs':epochs, 'init_mode_atoms':init_mode_atoms,'batch_size':batch_size,
                         'OT_loss':OT_loss,'algo_seed':algo_seed, 'beta1':beta_1, 'beta2':beta_2,'l2_reg':l2_reg}
        self.initialize_atoms(init_mode_atoms, algo_seed)
        self.initialize_optimizer()
        self.log ={}
        self.log['loss']=[]
        
        if not (checkpoint_timestamp is None):
            self.checkpoint_best_loss = np.inf
            self.checkpoint_atoms = self.Cs.copy() # save atoms state if the reconstruction over the whole dataset is minimized at current epoch
            self.log['checkpoint_loss']=[]
        hs= np.ones(self.shape_atoms)/self.shape_atoms
        self.hhs = hs[:,None].dot(hs[None,:])
        self.diagh= np.diag(hs)
        already_saved= False
        iter_by_epoch= int(self.dataset_size/batch_size)+1
        for epoch in tqdm(range(epochs)):
            cumulated_loss_over_epoch = 0
            for _ in range(iter_by_epoch):
                #batch sampling
                batch_idx = np.random.choice(range(self.dataset_size), size=batch_size, replace=False)
                #initialize weights {wk}
                w = np.ones((batch_size,self.number_atoms))/self.number_atoms
                best_w = w.copy()
                best_T = [None]*batch_size
                batch_best_loss= 0
                for k,idx in enumerate(batch_idx):
                    #BCD algorithm for GW unmixing problems solved independently on each graph of the batch
                    prev_loss_w = 10**(8)
                    current_loss_w = 10**(7)
                    best_loss_w = np.inf
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/prev_loss_w
                    outer_count=0
                    saved_transport=None
                    while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                        
                        prev_loss_w = current_loss_w
                        new_w,current_loss_w,saved_transport,_= self.BCD_step(idx,w[k], T_init=saved_transport, l2_reg=l2_reg,OT_loss=OT_loss,max_iter=max_iter_inner,eps=eps)
                        w[k] = new_w
                        if current_loss_w< best_loss_w:
                            best_w[k] = w[k]
                            best_T[k]= saved_transport
                            best_loss_w= current_loss_w
                        outer_count+=1
                        if prev_loss_w !=0:                        
                            convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w)
                        else:
                            break
                    batch_best_loss+= best_loss_w
                cumulated_loss_over_epoch+=batch_best_loss
                self.log['loss'].append(batch_best_loss)
                input_graphs = [self.graphs[idx] for idx in batch_idx]
                self.atoms_stochastic_update(input_graphs,best_w,best_T,beta_1,beta_2, lr, batch_size)

            if verbose:
                    print('epoch : %s / cumulated_loss_over_epoch : %s '%(epoch, cumulated_loss_over_epoch))
                
            # Visualization learning evolution 
            if not (visualization_frame is None):
                if (epoch%visualization_frame ==0):
                    pl.plot(self.log['loss'])
                    pl.xlabel('updates')
                    pl.ylabel('GW loss')
                    pl.title('loss evoluation over updates')
                    pl.show()
            #Save model        
            if not (checkpoint_timestamp is None):
                if ((epoch%checkpoint_timestamp ==0) and (epoch>0)) or (epoch==epochs-1):
                    # compute unmixing for the current dictionary state and stack
                    # on self.checkpoint_atoms the current dictionary state
                    checkpoint_unmixing,checkpoint_losses = self.GDL_unmixing(l2_reg,eps,max_iter_outer,max_iter_inner)
                    averaged_checkpoint_losses = np.mean(checkpoint_losses)
                    self.log['checkpoint_loss'].append(averaged_checkpoint_losses)
                    if averaged_checkpoint_losses < self.checkpoint_best_loss:
                        self.checkpoint_best_loss = averaged_checkpoint_losses
                        print('epoch : %s / new best checkpoint loss: %s'%(epoch, self.checkpoint_best_loss))
                        self.checkpoint_Cs = self.Cs.copy()
                        self.save_checkpoint(checkpoint_unmixing,checkpoint_losses)
            if not already_saved:
                print('saved settings:', self.settings)
                self.save_elements(save_settings=True)
                already_saved=True
            else:
                self.save_elements(save_settings=False)
                   
    def BCD_step(self,idx, w, T_init, l2_reg,eps,max_iter, OT_loss='square_loss'):
        """
        One step of the BCD algorithm to solve GW unmixing problem on a graph
        - refers to unmixing equation 2 and algorithm 1.
        Parameters
        ----------
        idx: (int) index of the graph in the dataset
        wt: (np.array) embeddings of the graph
        T_init: (np.array) matrix to initialize the solver of GW distance problem 
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        """
        graph= self.graphs[idx]
        sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, w)
        #Solve GW(Ck, \sum_s wk_s C_s)
        local_GW_loss,OT = gwu.np_GW2(graph,sum_ws_Cs,T_init=T_init)
        local_GW_loss-=l2_reg*np.sum(w**2)
        TCT = np.transpose(graph.dot(OT)).dot(OT)
        local_count = 0
        prev_criterion = 10**8
        curr_criterion = local_GW_loss   
        convergence_criterion = np.inf
        local_best_w = w.copy()
        best_loss = prev_criterion
        #Conditional gradient over wk given Tk
        while (convergence_criterion> eps) and (local_count<max_iter):
            prev_criterion=curr_criterion
            grad_w = np.zeros(self.number_atoms)
            for s in range(self.number_atoms):
                grad_w[s] = np.sum(self.Cs[s]*sum_ws_Cs*self.hhs - self.Cs[s]*TCT)
                grad_w[s] -= l2_reg*w[s]
                grad_w[s]*=2
            # Gradient direction : x= argmin_x x^T.grad_w
            x= np.zeros(self.number_atoms)
            sorted_idx = np.argsort(grad_w)#ascending order
            pos=0
            while (pos<self.number_atoms) and (grad_w[sorted_idx[pos]] == grad_w[sorted_idx[0]]) :
                x[sorted_idx[pos]] = 1
                pos+=1
            x/= pos
            #Line search step: solve argmin_{\gamma in (0,1)} a\gamma^2 + b\gamma +c
            sum_xs_Cs =  gwu.np_sum_scaled_mat(self.Cs,x)
            sum_xsws_Cs = sum_xs_Cs - sum_ws_Cs
            tr_xsws_xs = np.sum((sum_xsws_Cs*sum_xs_Cs)*self.hhs)
            tr_xsws_ws = np.sum((sum_xsws_Cs*sum_ws_Cs)*self.hhs)            
            a=tr_xsws_xs - tr_xsws_ws
            b= 2*tr_xsws_ws - 2*np.sum( sum_xsws_Cs*TCT)
            if l2_reg !=0:
                a -=l2_reg*np.sum((x-w)**2)
                b -= 2*l2_reg* (w.T).dot(x-w)
            
            if a>0:
                gamma = min(1, max(0, -b/(2*a)))
            elif a+b<0:
                gamma=1
            else:
                gamma=0            
            w+=gamma*(x-w)
            sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, w)
            curr_criterion+=a*(gamma**2) + b*gamma 
            convergence_criterion = abs(prev_criterion - curr_criterion)/prev_criterion
            
            local_count +=1
            if curr_criterion < best_loss:
                local_best_w = w
                best_loss = curr_criterion
        return  local_best_w,best_loss, OT,local_count
    
    
    def atoms_stochastic_update(self,graphs,batch_w,batch_T,beta_1,beta_2, lr,batch_size,proj='nsym', epsilon=10**(-15)):
        """
        Stochastic gradient step on atoms

        Parameters
        ----------
        graphs : (list) batch of graphs
        batch_w : (np.array) batch of corresponding embeddings
        batch_T : (np.array) batch of corresponding optimal transport plans
        beta_1 : (float) Adam parameter
        beta_2 : (float) Adam parameter
        lr : (float) learning rate
        batch_size : (int) batch size
        proj : (str) Projection of atoms - in {'nsym','sym'}. The default is 'nsym'.
        epsilon : (float) value to avoid division by 0. The default is 10**(-15).
        """
        
        grad_atoms = np.zeros_like(self.Cs)
        for k in range(batch_size):
            sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, batch_w[k])
            generic_term = sum_ws_Cs*self.hhs - (graphs[k].dot(batch_T[k])).T.dot(batch_T[k])
            for pos in range(self.number_atoms):
                grad_atoms[pos]+= batch_w[k,pos]*generic_term
        for pos in range(self.number_atoms):
            grad_atoms[pos]*=2/batch_size
            #stochastic update of atoms with adam optimizer
            C_t=self.atoms_adam_count[pos]+1
            C_m_k = beta_1*self.atoms_adam_m[pos] + (1-beta_1)*grad_atoms[pos]
            C_v_k = beta_2*self.atoms_adam_v[pos] + (1-beta_2)*(grad_atoms[pos]**2)
            unbiased_m_k = C_m_k/ (1-beta_1**C_t)
            unbiased_v_k = C_v_k/(1-beta_2**C_t)
            self.Cs[pos] -= lr*unbiased_m_k/ (np.sqrt(unbiased_v_k)+epsilon)
            if proj=='nsym':
                #projection on the set of non negative matrices is used by default while using adjacency matrices
                
                self.Cs[pos] = np.maximum(np.zeros((self.shape_atoms,self.shape_atoms)), (self.Cs[pos]+ self.Cs[pos].T )/2)
            elif proj=='sym':
                self.Cs[pos] =  (self.Cs[pos]+ self.Cs[pos].T )/2
            else:
                raise 'unknown projection of atoms'
            self.atoms_adam_count[pos]= C_t
            self.atoms_adam_m[pos] = C_m_k
            self.atoms_adam_v[pos] = C_v_k
                
    def GDL_unmixing(self,l2_reg:float,eps:float,max_iter_outer:int,max_iter_inner:int,verbose:bool=False):
        """
        For each graph - solve independently the GW unmixing problem with BCD algorithm
        
        Parameters
        ----------
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter_outer : maximum number of outer loop iterations of our BCD algorithm
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        verbose : Check the good evolution of the loss. The default is False.
        """
        w = np.ones((self.dataset_size,self.number_atoms))/self.number_atoms
        best_w = w.copy()
        saved_best_loss = np.zeros(self.dataset_size)        
        for idx in range(self.dataset_size):
        
            saved_transport=None
            
            prev_loss_w = 10**(8)
            current_loss_w = 10**(7)
            convergence_criterion = np.inf
            best_loss_w= prev_loss_w
            outer_count=0
            while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                outer_count+=1
                prev_loss_w = current_loss_w
                w[idx],current_loss_w,saved_transport,_= self.BCD_step(idx, w[idx], saved_transport, l2_reg,max_iter=max_iter_inner,eps=eps)
                convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w)
                if current_loss_w < best_loss_w:
                    best_w[idx]= w[idx]
                    best_loss_w= current_loss_w
                    saved_best_loss[idx]= current_loss_w
        return best_w,saved_best_loss
    
    
    def save_checkpoint(self, checkpoint_unmixing,checkpoint_losses):
        """
        Save embeddings, reconstructions, at different time steps.
        """
        path = os.path.abspath('../')+'/%s/%s/'%(self.experiment_repo,self.experiment_name)
        print('path',path)
        if not os.path.exists(path):
            os.makedirs(path)
            print('made dir', path)
        np.save(path+'/checkpoint_unmixings.npy', checkpoint_unmixing)
        np.save(path+'/checkpoint_reconstruction.npy', checkpoint_losses)
        np.save(path+'/checkpoint_Cs.npy', self.checkpoint_Cs)
        
        
    def save_elements(self,save_settings=False):
        """
        Save embeddings,reconstructions, dictionary state - when called. Used to keep the final state.
        """
        path = os.path.abspath('../')+self.experiment_repo
        print('path',path)
        if not os.path.exists(path+self.experiment_name):
            os.makedirs(path+self.experiment_name)
            print('made dir', path+self.experiment_name)
        
        np.save(path+'%s/Cs.npy'%self.experiment_name, self.Cs)
        for key in self.log.keys():
            try:
                np.save(path+'%s/%s.npy'%(self.experiment_name,key), np.array(self.log[key]))
            except:
                print('bug for  log component: %s'%key)
                pass
        if save_settings:
            pd.DataFrame(self.settings, index=self.settings.keys()).to_csv(path+'%s/settings'%self.experiment_name)
            if self.dataset_name in ['balanced_clustertoy','clustertoy2C']:
                pd.DataFrame(self.settings_toy, index=self.settings_toy.keys()).to_csv(path+'%s/toydataset_settings'%self.experiment_name)
            
    def load_elements(self, path=None,use_checkpoint=False):
        """
        Loading final state of the graph dictionary 
        """
        if path is None:
            path= os.path.abspath('../')
        path+='/%s/%s/'%(self.experiment_repo,self.experiment_name)
        if not use_checkpoint:
            self.Cs = np.load(path+'/Cs.npy')
            self.number_atoms= self.Cs.shape[0]
            self.shape_atoms = self.Cs.shape[-1]
            hs= np.ones(self.shape_atoms)/self.shape_atoms
            self.hhs = hs[:,None].dot(hs[None,:])
            self.diagh= np.diag(hs)
        else:
            self.Cs= np.load(path+'/checkpoint_Cs.npy')
            self.number_atoms= self.Cs.shape[0]
            self.shape_atoms = self.Cs.shape[-1]
            hs= np.ones(self.shape_atoms)/self.shape_atoms
            self.hhs = hs[:,None].dot(hs[None,:])
            self.diagh= np.diag(hs)



#%% 


class extended_GW_Unsupervised_DictionaryLearning():
    """
    Extended GDL: 
        Simultaneously learning structures and masses 
        with atoms {Cbar_s,hbar_s} as expressed in Equation 6 of the paper.
        
    """
    def __init__(self,
                 dataset_name:str, 
                 mode:str, 
                 number_atoms:int, 
                 shape_atoms:int, 
                 experiment_repo:str, 
                 experiment_name:str,
                 data_path='../data/'):
        self.experiment_repo= experiment_repo
        self.experiment_name = experiment_name
        str_to_method = {'ADJ': 'adjacency', 'SP':'shortest_path','LAP':'laplacian'}
        self.number_atoms = number_atoms
        self.shape_atoms = shape_atoms
        assert dataset_name in ['imdb-b','imdb-m']
        self.dataset_name= dataset_name
        self.mode = mode
        X,self.labels=dataloader.load_local_data(data_path,dataset_name)
        if self.mode in str_to_method.keys():
            self.graphs = [np.array(X[t].distance_matrix(method=str_to_method[mode]),dtype=np.float64) for t in range(X.shape[0])]
        else:
            raise 'unknown graph representation for GW dictionary learning'
        self.dataset_size = len(self.graphs)
        self.shapes= np.array([X.shape[0] for X in self.graphs])
        self.masses = [np.ones(n)/n for n in self.shapes]
        
    def initialize_atoms(self,
                         init_mode_atoms:int=1,
                         algo_seed:int=0):
        """
        Initialize atoms {Cbar_s}: 0= random initialization / 1= random sampling in the dataset 
        """
        np.random.seed(algo_seed)
        if init_mode_atoms==0:
            # initialize Cs components with uniform distribution
            init_Cs = np.random.uniform(low= 10**(-15), high=1, size= (self.number_atoms,self.shape_atoms, self.shape_atoms ))
            self.Cs = 0.5* (init_Cs + init_Cs.transpose((0,2,1))) 
        elif init_mode_atoms ==1:
            # Sample from the dataset:
            # 1. If enough samples with adequate shape exist in the dataset, we sample them randomly
            # 2. Else if samples with adequate shape exist but less than the required number, we add symmetric noise to the existing ones
            # 3. Else an exception is raised
            shape_idx = np.argwhere(self.shapes==self.shape_atoms)[:,0]
            if shape_idx.shape[0]>0:
                print('find samples with good shape')
                if shape_idx.shape[0]>self.number_atoms:
                    print('find enough samples with good shapes')
                    warmstart_idx = np.random.choice(shape_idx,size=self.number_atoms,replace=False)
                    print('SELECTED IDX FOR INITIALIZATION : ', warmstart_idx)
                    self.Cs= np.stack([self.graphs[idx] for idx in warmstart_idx])
                    
                else:
                    print('could not find enough samples with good shapes')
                    warmstart_idx = np.random.choice(shape_idx, size=self.number_atoms, replace=True)
                    #add noise to existing ones to reach the required number of atoms
                    self.Cs= np.stack([self.graphs[idx] for idx in warmstart_idx])
                    mean_ = np.mean(self.Cs)
                    std_ = np.std(self.Cs)
                    noise = np.random.normal(loc= mean_, scale = std_, size= (self.number_atoms,self.shape_atoms, self.shape_atoms ))
                    noise = 0.5* (noise+ noise.transpose((0,2,1))) #add symmetric noise
                    self.Cs+=noise
                   
            else:
                raise 'There is no graph with adequate shape in the dataset - change the value of init_mode_atoms from 1 to 0 for random initialization '
        
    def initialize_h_atoms(self,
                           init_hs_mode:int=1,
                           algo_seed:int=0):
        """
        initialize atoms {h_s}: 0= uniform / 1= randomly following ~ U([0.5,1])
        """
        np.random.seed(algo_seed)
        if init_hs_mode==0:
            # Forces the masses of each atom to be similar along learning
            print('initializing HS uniformly')
            self.hs = np.ones((self.number_atoms, self.shape_atoms))/self.shape_atoms
        elif init_hs_mode==1:
            # Initialize them close to the uniform distribution 
            # To ease the stochastic optimization procedure
            # Works better in practice
            print('initializing HS as random distributions ~ U([0.9,1])')
            self.hs =  np.random.uniform(low= 0.9, high=1, size= (self.number_atoms,self.shape_atoms))
            for i in range(self.number_atoms):
                scale = np.sum(self.hs[i,:])
                self.hs[i]/=scale
        else:
            raise 'unknown init hs mode'
            
    def initialize_optimizer(self):
        """
        Initialize adam optimizer
        """
        self.Cs_adam_m = np.zeros((self.number_atoms, self.shape_atoms,self.shape_atoms))#Initialize first  moment vector
        self.Cs_adam_v = np.zeros((self.number_atoms, self.shape_atoms, self.shape_atoms))#Initialize second moment vector
        self.Cs_adam_count = np.zeros((self.number_atoms))
        self.hs_adam_m = np.zeros((self.number_atoms, self.shape_atoms))#Initialize first  moment vector
        self.hs_adam_v = np.zeros((self.number_atoms,  self.shape_atoms))#Initialize second moment vector
        self.hs_adam_count = np.zeros((self.number_atoms))
        
            
    def Stochastic_algorithm(self,
                             l2_reg_c:float, 
                             eps:float,max_iter_outer:int,max_iter_inner:int ,
                             lr_Cs:float,lr_hs:float,batch_size:int,
                             epochs:int, algo_seed:int=0 ,
                             beta_1:float=0.9, beta_2:float=0.99,
                             init_mode_atoms:int=1,init_hs_mode:int=1,
                             centering:bool=True,
                             checkpoint_timestamp:int=None,
                             visualization_frame:int=None,
                             verbose:bool =False,use_optimizer:bool=True):
        """
        Stochastic algorithm to learn simultaneously {C_s} and {h_s}. Described in supplementary materials
        - refers to Equation 6 in the paper .
        - Algorithmic details provided in the supplementary 
            > Use subgradient highlighted in proposition 2
            > Extension of the BCD algorithm as detailed in supplementary material
        
        Parameters
        ----------
        l2_reg_c : regularization coefficient of the negative quadratic regularization on unmixings over structure
        eps : precision to stop our unmixing learning process based on relative variation of the loss
        max_iter_outer : maximum number of outer loop iterations of our BCD algorithm            
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        lr_Cs : Initial learning rate of Adam optimizer over atoms structure Cs
        lr_hs: Initial learning rate of Adam optimizer over atoms masses hs
        batch_size : batch size 
        steps : number of updates of dictionaries 
        algo_seed : initialization random seed
        OT_loss : GW discrepency ground cost. The default is 'square_loss'.
        beta_1 : Adam parameter on gradient. The default is 0.9.
        beta_2 : Adam parameter on gradient**2. The default is 0.99.
        init_mode_atoms : 0= random initialization of {Cs} / 1= initialization of {Cs} by sampling in the dataset
        init_hs_atoms : 0= uniform initialization of {hs} / 1= random initialization of {Cs} 
        checkpoint_timestamp : checkpoint i.e number of epochs to run unmixing on the current dictionary state to save best model 
        visualization_frame: Can be set stricly positive below the number of epochs to plot the loss evolution periodically over epochs
        centering: set permanently to True in order to center the dual potentials from Wasserstein cost for computing gradients over masses
        verbose : Check the good evolution of the loss. The default is False.
        """
        # Save settings
        self.settings = {'number_atoms':self.number_atoms, 'shape_atoms': self.shape_atoms, 
                         'eps':eps,'max_iter_inner':max_iter_inner,'max_iter_outer':max_iter_outer,
                         'lr_Cs':lr_Cs, 'lr_hs':lr_hs,'epochs':epochs,'batch_size':batch_size,'algo_seed':algo_seed,
                         'init_mode_atoms':init_mode_atoms,'init_hs_mode':init_hs_mode, 'beta1':beta_1, 'beta2':beta_2,
                         'l2_reg_c':l2_reg_c,'centering':centering, 'use_optimizer':use_optimizer}
        self.initialize_atoms(init_mode_atoms, algo_seed)
        self.initialize_h_atoms(init_hs_mode,algo_seed)
        if use_optimizer:
            self.initialize_optimizer()
        self.log={}
        self.log['loss']=[]
        if not (checkpoint_timestamp is None):
            self.checkpoint_best_loss = np.inf
            self.checkpoint_atoms = self.Cs.copy() # save atoms state if the reconstruction over the whole dataset is minimized at current epoch
            self.log['checkpoint_loss']=[]
        hs= np.ones(self.shape_atoms)/self.shape_atoms
        self.hhs = hs[:,None].dot(hs[None,:])
        self.diagh= np.diag(hs)
        already_saved= False
        iter_by_epoch= int(self.dataset_size/batch_size)+1
        for epoch in tqdm(range(epochs)):
            cumulated_loss_over_epoch = 0
            for _ in range(iter_by_epoch):
                #batch sampling
                batch_idx = np.random.choice(range(self.dataset_size), size=batch_size, replace=False)
                # wc : current unmixings on structures Cs
                wc = np.ones((batch_size,self.number_atoms))/self.number_atoms
                best_wc = wc.copy()
                # wh: current unmixings on masses hs
                wh = np.ones((batch_size,self.number_atoms))/self.number_atoms
                best_wh = wh.copy()
                best_T = [None]*batch_size
                best_ut = [None]*batch_size # just for tracking purposes
                best_vt = np.zeros((batch_size, self.shape_atoms))
                batch_best_loss=0
                #BCD ALGORITHM FOR SOLVING UNMIXING PROBLEMS OF EQUATION 6
                for k,idx in enumerate(batch_idx):
                    previous_loss = 10**(8)
                    current_loss = 10**(7)
                    best_loss= np.inf
                    Ct= self.graphs[idx]
                    ht= self.masses[idx]
                    convergence_criterion = np.inf
                    outer_count=0
                    while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                        previous_loss = current_loss
                        # 1. get optimal transport plan for current state
                        sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, wc[k])
                        sum_wh_hs = gwu.np_sum_scaled_mat(self.hs, wh[k])
                        #here the variable self.hhs of the basic algorithm depends on the selected graph as embeddings over the masses are involved
                        local_GW_loss,Tt,ut,vt = gwu.np_GW2_extendedDL(Ct,sum_ws_Cs,p=ht,q=sum_wh_hs,T_init=None,centering=centering)
                        local_GW_loss -= l2_reg_c*np.sum(wc[k]**2)
                        # 2. optimize wh given T and wc (optimize unmixings on masses before unmixings on structures)
                        wh[k], local_wh_loss = self.solver_masses_unregularized(  wh[k], vt)
                        if verbose:
                            const_utp = (ht.T).dot(ut)
                            local_wh_loss +=const_utp
                            local_wh_loss*=0.5
                            print('local wh loss = %s  / new wh = %s '%(local_wh_loss, wh[k]))
                        # 3. optimize wc given T and wh
                        wc[k],current_loss, _=self.structure_unmixings_FWupdate(Ct,ht,wc[k],wh[k], l2_reg_c,T_star=Tt, eps=eps, max_iter_inner=max_iter_inner,  centering=centering)
    
                        if current_loss< best_loss:
                            best_wc[k] = wc[k]
                            best_wh[k] = wh[k]
                            best_T[k]= Tt
                            best_ut[k] = ut
                            best_vt[k] = vt
                            best_loss= current_loss
                        outer_count+=1
                        if previous_loss!=0:                        
                            convergence_criterion = abs(previous_loss - current_loss)/abs(previous_loss)
                        else:
                            convergence_criterion = abs(previous_loss - current_loss)/abs(previous_loss+10**(-12))
                        
                    batch_best_loss+= best_loss
                cumulated_loss_over_epoch+=batch_best_loss
                self.log['loss'].append(batch_best_loss)
                    
                Ct = [self.graphs[idx] for idx in batch_idx]
                ht= [self.masses[idx] for idx in batch_idx]
                  
                self.atoms_simultaneous_stochastic_update(Ct,best_wc,best_wh,best_T, best_vt,beta_1,beta_2, lr_Cs, lr_hs,batch_size,use_optimizer=use_optimizer,verbose=verbose)
                
            if verbose:
                    print('epoch : %s / cumulated_loss_over_epoch : %s '%(epoch, cumulated_loss_over_epoch))
                
            # Visualization learning evolution 
            if not (visualization_frame is None):
                if (epoch%visualization_frame ==0):
                    pl.plot(self.log['loss'])
                    pl.xlabel('updates')
                    pl.ylabel('GW loss')
                    pl.title('loss evoluation over updates')
                    pl.show()
            #Save model        
            if not (checkpoint_timestamp is None):
                if ((epoch%checkpoint_timestamp ==0) and (epoch>0)) or (epoch==epochs-1):
                    # compute unmixing for the current dictionary state and stack
                    # on self.checkpoint_atoms the current dictionary state
                    checkpoint_unmixing_wc,checkpoint_unmixing_wh,checkpoint_losses = self.extendedGDL_unmixing(l2_reg_c,eps,max_iter_outer,max_iter_inner,seed=None,verbose=False,centering=centering)
                    averaged_checkpoint_losses = np.mean(checkpoint_losses)
                    self.log['checkpoint_loss'].append(averaged_checkpoint_losses)
                    if averaged_checkpoint_losses < self.checkpoint_best_loss:
                        self.checkpoint_best_loss = averaged_checkpoint_losses
                        print('epoch : %s / new best checkpoint loss: %s'%(epoch, self.checkpoint_best_loss))
                        self.checkpoint_Cs = self.Cs.copy()
                        self.checkpoint_hs = self.hs.copy()
                        
                        self.save_checkpoint(checkpoint_unmixing_wc,checkpoint_unmixing_wh,checkpoint_losses)
            if not already_saved:
                print('saved settings:', self.settings)
                self.save_elements(save_settings=True)
                already_saved=True
            else:
                self.save_elements(save_settings=False)
                   

    def atoms_simultaneous_stochastic_update(self,
                                             Ct:list,
                                             wc_t:list,
                                             wh_t:list,
                                             Tt:list,
                                             vt:list,
                                             beta_1:float=0.9,
                                             beta_2:float=0.99,
                                             lr_Cs:float=0.01,
                                             lr_hs:float=0.001,
                                             batch_size:int=16,
                                             use_optimizer:bool=True, 
                                             epsilon:float=10**(-15),
                                             verbose:bool=False):
        """
        Stochastic gradient steps for {C_s} and {h_s}
        
        parameters
        ------
        
        Ct: list of graphs
        wc_t: corresponding list of unmixings on structure Cs
        wh_t: corresponding list of unmixings on masses hs
        Tt: corresponding list of OT matrices 
        vt: corresponding list of Wasserstein potentials for computing gradients
        beta_1 : Adam parameter on gradient. The default is 0.9.
        beta_2 : Adam parameter on gradient**2. The default is 0.99.
        lr_Cs : Initial learning rate of Adam optimizer over atoms structure Cs
        lr_hs: Initial learning rate of Adam optimizer over atoms masses hs
        batch_size : batch size 
        use_optimizer: either to use Adam optimizer or not
        epsilon: precision to avoid division by 0
        """
        
        grad_Cs = np.zeros_like(self.Cs)
        grad_hs = np.zeros_like(self.hs)
        Cs_terms = []

        for k in range(batch_size):
            sum_wh_hs = gwu.np_sum_scaled_mat(self.hs, wh_t[k])
            #here the variable self.hhs of the basic algorithm depends on the selected graph as embeddings over the masses are involved
            hhs = sum_wh_hs[:,None].dot(sum_wh_hs[None,:])
            sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, wc_t[k])
            Cs_terms.append((sum_ws_Cs*hhs- (Ct[k].dot(Tt[k])).T.dot(Tt[k])))
        for pos in range(self.number_atoms):
            for k in range(batch_size):
                grad_Cs[pos] += wc_t[k,pos]*Cs_terms[k]
                grad_hs[pos] += wh_t[k,pos]*vt[k]
            
            grad_Cs[pos]*=2/batch_size
            grad_hs[pos]*=2/batch_size
            if verbose:
                print('pos = %s/ gradients Cs (norm):'%pos, np.linalg.norm(grad_Cs[pos],ord=2))
                print('pos = %s/ gradients Hs (norm):'%pos, np.linalg.norm(grad_hs[pos],ord=2))
            if use_optimizer:
                C_t=self.Cs_adam_count[pos]+1
                C_m_k = beta_1*self.Cs_adam_m[pos] + (1-beta_1)*grad_Cs[pos]
                C_v_k = beta_2*self.Cs_adam_v[pos] + (1-beta_2)*(grad_Cs[pos]**2)
                unbiased_m_k = C_m_k/ (1-beta_1**C_t)
                unbiased_v_k = C_v_k/(1-beta_2**C_t)
                #old_Cs= self.Cs[s].copy()
                self.Cs[pos] -= lr_Cs*unbiased_m_k/ (np.sqrt(unbiased_v_k)+epsilon)
                self.Cs[pos] = np.maximum(np.zeros((self.shape_atoms,self.shape_atoms)), (self.Cs[pos]+ self.Cs[pos].T )/2)
                #print('Cs variations (norm):', np.linalg.norm(self.Cs[s] - old_Cs,ord=2))
                #update Adam
                self.Cs_adam_count[pos]= C_t
                self.Cs_adam_m[pos] = C_m_k
                self.Cs_adam_v[pos] = C_v_k
                h_t=self.hs_adam_count[pos]+1
                h_m_k = beta_1*self.hs_adam_m[pos] + (1-beta_1)*grad_hs[pos]
                h_v_k = beta_2*self.hs_adam_v[pos] + (1-beta_2)*(grad_hs[pos]**2)
                unbiased_m_k = h_m_k/ (1-beta_1**h_t)
                unbiased_v_k = h_v_k/(1-beta_2**h_t)
                
                self.hs[pos] -= lr_hs*unbiased_m_k/ (np.sqrt(unbiased_v_k)+epsilon)
                if verbose:
                    print('pos= %s / HS before simplex projection:'%pos, self.hs[pos])
                self.hs[pos] = gwu.np_simplex_projection(self.hs[pos])
                if verbose:
                    print('pos= %s / HS after simplex projection:'%pos, self.hs[pos])
                #print('Cs variations (norm):', np.linalg.norm(self.Cs[s] - old_Cs,ord=2))
                #update Adam
                self.hs_adam_count[pos]= h_t
                self.hs_adam_m[pos] = h_m_k
                self.hs_adam_v[pos] = h_v_k
            else:
                self.Cs[pos] -= lr_Cs*grad_Cs[pos]
                self.Cs[pos] = np.maximum(np.zeros((self.shape_atoms,self.shape_atoms)), (self.Cs[pos]+ self.Cs[pos].T )/2)
                self.hs[pos] -= lr_hs*grad_hs[pos]
                if verbose:
                    print('pos= %s / HS before simplex projection:'%pos, self.hs[pos])
                self.hs[pos] = gwu.np_simplex_projection(self.hs[pos])
                if verbose:
                    print('pos= %s / HS after simplex projection:'%pos, self.hs[pos])
                
    def structure_unmixings_FWupdate(self,Ct,ht,wc,wh, l2_reg_c,T_star,eps, max_iter_inner, verbose = False, centering=True):
        """
        Frank-Wolfe algorithm for updating unmixing on structure Cs
        cf Algorithm 1. and vanilla GDL problem.
        
        """
        sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, wc)
        sum_wh_hs = gwu.np_sum_scaled_mat(self.hs, wh)
        #here the variable self.hhs of the basic algorithm depends on the selected graph as embeddings over the masses are involved
        hhs = sum_wh_hs[:,None].dot(sum_wh_hs[None,:])
    
        local_GW_loss,Tt = gwu.np_GW2(Ct,sum_ws_Cs,ht, sum_wh_hs, T_star=T_star)
        local_GW_loss -= l2_reg_c*np.sum(wc**2)
        if verbose:
            print('1st update wc - old T - GW loss:', local_GW_loss)
        TCT = np.transpose(Ct.dot(Tt)).dot(Tt)
        local_count = 0
        prev_criterion = 10**8
        curr_criterion = local_GW_loss   
        convergence_criterion = np.inf
        local_best_wc = wc.copy()
        local_best_loss = prev_criterion
        #Conditional gradient over wk given Tk
        while (convergence_criterion> eps) and (local_count<max_iter_inner):
            prev_criterion=curr_criterion
            grad_wc = np.zeros(self.number_atoms)
            for s in range(self.number_atoms):
                grad_wc[s] = np.sum(self.Cs[s]*sum_ws_Cs*hhs - self.Cs[s]*TCT)
                grad_wc[s] -= l2_reg_c*wc[s]
                grad_wc[s]*=2
            
            # Gradient direction : x= argmin_x x^T.grad_wt
            x= np.zeros(self.number_atoms)
            sorted_idx = np.argsort(grad_wc)#ascending order
            pos=0
            while (pos<self.number_atoms) and (grad_wc[sorted_idx[pos]] == grad_wc[sorted_idx[0]]) :
                x[sorted_idx[pos]] = 1
                pos+=1
            x/= pos
            
            #Line search step: solve argmin_{\gamma in (0,1)} a\gamma^2 + b\gamma +c
            sum_xs_Cs =  gwu.np_sum_scaled_mat(self.Cs,x)
            sum_xsws_Cs = sum_xs_Cs - sum_ws_Cs
            tr_xsws_xs = np.sum((sum_xsws_Cs*sum_xs_Cs)*hhs)
            tr_xsws_ws = np.sum((sum_xsws_Cs*sum_ws_Cs)*hhs)
            
            a=tr_xsws_xs - tr_xsws_ws
            b= 2*tr_xsws_ws - 2*np.sum( sum_xsws_Cs*TCT)
            if l2_reg_c !=0:
                a -=l2_reg_c*np.sum((x-wc)**2)
                b -= 2*l2_reg_c* np.sum(wc*(x-wc))
            if a>0:
                gamma = min(1, max(0, -b/(2*a)))
            elif a+b<0:
                gamma=1
            else:
                gamma=0
            wc +=gamma*(x-wc)
            sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, wc)
            #evaluate GW loss with new wk and same Tk
            curr_criterion += a*(gamma**2)+b*gamma
            if prev_criterion!=0:
                convergence_criterion = abs(prev_criterion - curr_criterion)/abs(prev_criterion)
            else:
                convergence_criterion = abs(prev_criterion - curr_criterion)/abs(prev_criterion+10**(-15))
            local_count +=1
            if curr_criterion < local_best_loss:
                local_best_wc = wc
                local_best_loss = curr_criterion
        return local_best_wc, local_best_loss, local_count
        
    def compute_cut_dual_objective(self, vt,wh_t):
        """
        evaluate the scalar product on potential to optimize as detailed in the supplementary material.
        """
        vec = np.zeros(self.number_atoms)
        for s in range(self.number_atoms):
            #print('local_hs shape: ', self.hs[s,:].shape)
            vec[s] = (vt.T).dot(self.hs[s,:])
        
        return (wh_t.T).dot(vec),vec
    
    def solver_masses_unregularized(self,  wh_t, vt):
        """
        Minimize the corresponding linear OT problem w.r.t embeddings on {h_s} (denotes wh_t here)
        """
        _, vec = self.compute_cut_dual_objective(vt,wh_t) 
        x= np.zeros(self.number_atoms)
        sorted_idx = np.argsort(vec)#ascending order
        pos=0
        while (pos<self.number_atoms) and (vec[sorted_idx[pos]] == vec[sorted_idx[0]]) :
            x[sorted_idx[pos]] = 1
            pos+=1
        x/= pos
        new_loss,_ = self.compute_cut_dual_objective(vt,x)
        return  x,new_loss
    
    def extendedGDL_unmixing(self,l2_reg_c,eps,max_iter_outer,max_iter_inner,seed=0,verbose=False,centering=True):
        """
        Solve the extended GW unmixing problem on the whole dataset.
        """
        if not (seed is None):
            np.random.seed(seed)
        
        wc = np.ones((self.dataset_size,self.number_atoms))/self.number_atoms
        best_wc = wc.copy()
        wh = np.ones((self.dataset_size,self.number_atoms))/self.number_atoms
        best_wh = wh.copy()
        best_loss = np.zeros(self.dataset_size)
        for t in range(self.dataset_size):
            
            local_best_loss=np.inf
            #print('batch:',batch_t)
        
            previous_loss = 10**(8)
            current_loss = 10**(7)
            
            Ct= self.C_target[t]
            ht= self.h_target[t]
            convergence_criterion = np.inf
            outer_count=0
            while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                previous_loss = current_loss
                # 1. get optimal transport plan for current state
                sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, wc[t])
                sum_wh_hs = gwu.np_sum_scaled_mat(self.hs, wh[t])
                #here the variable self.hhs of the basic algorithm depends on the selected graph as embeddings over the masses are involved
                local_GW_loss,Tt,ut,vt = gwu.np_GW2_extendedDL(Ct,sum_ws_Cs,p=ht,q=sum_wh_hs,T_init=None,verbose=verbose,centering=centering)
                local_GW_loss -= l2_reg_c*np.sum(wc[t]**2)
                # 2. optimize ah given T and ac
                wh[t], local_wh_loss = self.solver_masses_unregularized(  wh[t], vt, verbose=verbose)
                # 3. optimize wc given T and new wh 
                wc[t],current_loss, inner_count=self.structure_unmixings_FWupdate(Ct,ht,wc[t],wh[t], l2_reg_c,T_star=Tt, eps=eps, max_iter_inner=max_iter_inner, centering=centering)
                
                if current_loss< local_best_loss:
                    best_wc[t] = wc[t]
                    best_wh[t] = wh[t]
                    local_best_loss= current_loss
                    best_loss[t] = current_loss
                outer_count+=1
                if previous_loss!=0:                        
                    convergence_criterion = abs(previous_loss - current_loss)/abs(previous_loss)
                else:
                    convergence_criterion = abs(previous_loss - current_loss)/abs(previous_loss+10**(-15))
                #print('UNMIXING - current loss : ', current_loss_a)
        return best_wc,best_wh,best_loss
    
    
    
    def save_checkpoint(self, checkpoint_unmixing_wc, checkpoint_unmixing_wh,checkpoint_losses):
        """
        Save embeddings, reconstructions, at different time steps.
        """
        path = os.path.abspath('../')+'/%s/%s/'%(self.experiment_repo,self.experiment_name)
        print('path',path)
        if not os.path.exists(path):
            os.makedirs(path)
            print('made dir', path)
        np.save(path+'/checkpoint_unmixings_wc.npy', checkpoint_unmixing_wc)
        np.save(path+'/checkpoint_unmixings_wh.npy', checkpoint_unmixing_wh)
        np.save(path+'/checkpoint_reconstruction.npy', checkpoint_losses)
        np.save(path+'/checkpoint_Cs.npy', self.checkpoint_Cs)
        np.save(path+'/checkpoint_hs.npy', self.checkpoint_hs)
            
        
    def save_elements(self,save_settings=False):
            
        path = os.path.abspath('../')+self.experiment_repo
        print('path',path)
        if not os.path.exists(path+self.experiment_name):
            os.makedirs(path+self.experiment_name)
            print('made dir', path+self.experiment_name)
        
        np.save(path+'%s/Cs.npy'%self.experiment_name, self.Cs)
        np.save(path+'%s/hs.npy'%self.experiment_name, self.hs)
        
        for key in self.log.keys():
            try:
                np.save(path+'%s/%s.npy'%(self.experiment_name,key), np.array(self.log[key]))
            except:
                print('bug for  log component: %s'%key)
                pass
        if save_settings:
            pd.DataFrame(self.settings, index=self.settings.keys()).to_csv(path+'%s/settings'%self.experiment_name)
            if self.dataset_name in ['balanced_clustertoy','unbalanced_clustertoy']:
                pd.DataFrame(self.settings_toy, index=self.settings_toy.keys()).to_csv(path+'%s/toydataset_settings'%self.experiment_name)
                
                
    def load_elements(self, path=None,use_checkpoint=False):
        """
        Loading final state of the graph dictionary 
        """
        if path is None:
            path= os.path.abspath('../')
        path+='/%s/%s/'%(self.experiment_repo,self.experiment_name)
        if not use_checkpoint:
            self.Cs = np.load(path+'/Cs.npy')
            self.hs = np.load(path+'/hs.npy')
            self.number_atoms= self.Cs.shape[0]
            self.shape_atoms = self.Cs.shape[-1]
        else:
            self.Cs= np.load(path+'/checkpoint_Cs.npy')
            self.hs= np.load(path+'/checkpoint_hs.npy')
            self.number_atoms= self.Cs.shape[0]
            self.shape_atoms = self.Cs.shape[-1]
            
        
#%% ONLINE LEARNING - EMBEDDING TRACKING



class Online_GW_Twitch_streaming():
    
    """
    Similar process than GW_Unsupervised_DictionaryLearning except that it is adapted to the online experiments
    illustrated in Section 4.3:  Online graph subspace estimation and change

        > Online: we can not fit dataset in memory therefore we integrated a data sampler conditioned for the experiment
        > Change point detection events: simulated events as described on dataset TWITCH-EGOS
        > Optimizer: vanilla SGD is used instead of Adam
    """
    def __init__(self,dataset_name, dataset_mode,graph_mode, number_Cs, shape_Cs, 
                 experiment_repo, experiment_name,data_path='../data/'):
        assert 'twitch_egos' in dataset_name
        self.dataset_name= dataset_name
        self.dataset_mode = dataset_mode
        self.graph_mode = graph_mode # like 'ADJ' for adjacency matrices
        self.number_Cs = number_Cs #number of atoms
        self.shape_Cs = shape_Cs #shared number of nodes by the atoms
        
        local_path = os.path.abspath('../')
        self.data_path = local_path+'/data/%s/'%self.dataset_name
        self.experiment_repo= experiment_repo
        self.experiment_name = experiment_name
        
    def initialize_atoms(self,algo_seed=0):
        """
        Initialize Atoms {C_s}: randomly (i.e = 0) for online experiments
        """
        np.random.seed(algo_seed)
        init_Cs = np.random.uniform(low= 10**(-15), high=1, size= (self.number_Cs,self.shape_Cs, self.shape_Cs ))
        self.Cs= 0.5* (init_Cs + init_Cs.transpose((0,2,1))) 
        

    def twitch_streaming_mode(self, 
                              streaming_mode:int):
        """
        Monitor streaming events
        """
        if streaming_mode==0:
            self.labels_by_events = [[0],[1]]
        elif streaming_mode==1:
            self.labels_by_events=[[1],[0]]
        else:
            raise 'unknown streaming mode'
        self.all_labels=[0,1]
 
    def Online_Learning_fulldataset(self,
                                    l2_reg:float,
                                    eps:float = 10**(-5),
                                    sampler_batchsize:int=100,
                                    checkpoint_size:int=2000, 
                                    max_iter_outer:int=10,
                                    max_iter_inner:int=100,
                                    lr_Cs:float= 0.01,
                                    steps:int = 1000,
                                    algo_seed:int=0,
                                    event_steps:list=[10000], 
                                    streaming_mode:int=0,
                                    checkpoint_steps:int= 100,
                                    save_chunks:bool=False,
                                    verbose=False):
        """
        Online Learning of our GW GDL by streams of the full dataset Twitch_EGOS
        
        Stochastic Algorithm to learn dictionary atoms with our GW loss on the fly
        
        - refers to Equation 4 in the paper and algorithm 2. + Experiments ran in section 4.3

        Parameters
        ----------
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        sampler_batchsize: number of graphs stored in memory. refreshed after everyone contributed to SGD update one by one.
        checkpoint_size: number of graphs randomly sampled to evaluate unmixing reconstruction
        max_iter_outer : maximum number of outer loop iterations of our BCD algorithm            
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        lr_Cs : Learning rate for atoms Cs
        batch_size : batch size 
        steps : number of updates of dictionaries. 
        algo_seed : initialization random seed
        event_steps: list of iterations index to make a change occur in the stream (here change of class)
        streaming_mode: used to control which class is streamed first
        verbose : Check the good evolution of the loss. The default is False.
        """
        
        self.settings = {'number_Cs':self.number_Cs, 'shape_atoms': self.shape_Cs, 'eps':eps,'max_iter_outer':max_iter_outer,'max_iter_inner':max_iter_inner,
                         'lr_Cs':lr_Cs, 'steps':steps,'algo_seed':algo_seed,'l2_reg':l2_reg,
                         'checkpoint_steps':checkpoint_steps ,'sampler_batchsize':sampler_batchsize}
        
        self.initialize_atoms( algo_seed)# initialize randomly atoms and set generator seed
        self.save_chunks=save_chunks
        self.event_steps = event_steps
        print('STEPS= %s / EVENT STEPS: %s'%(steps,event_steps))
        self.twitch_streaming_mode(streaming_mode)
        self.log ={}
        self.log['loss']=[]
        self.log['steps']=[]
        if not self.save_chunks:
            #tracking of dictionary state 
            self.tracked_a=[]            
            self.tracked_losses = []
            self.tracked_atoms =[]
            self.tracked_idx=[]
        hs= np.ones(self.shape_Cs)/self.shape_Cs
        self.hhs = hs[:,None].dot(hs[None,:])
        self.diagh= np.diag(hs)
        already_saved= False
        np.random.seed(0)
        self.current_event = 0
        print('MAX ITER OUTER: %s / MAX ITER INNER : %s    / STOP RATE: %s'%(max_iter_outer,max_iter_inner,eps))
        tracking_new_event = self.event_steps[self.current_event]
        print('labels_by_events:', self.labels_by_events)
        #For twitch egos each class is seen after the other > So size_sampler is just equal to sampler_batchsize / can be adapted for different scenarios
        size_sampler = len(self.labels_by_events[self.current_event])*sampler_batchsize
        print('initial size_sampler:', size_sampler)
        reload_batch = False # Flag used to load a certain amount of graphs in memory and refresh it after each graph has been used to update the dictionary 
        for i in tqdm(range(steps)):
            if i > tracking_new_event:
                #monitor events depending on iteration i 
                if verbose:
                    print('initial current event = %s / tracking_new_event = %s'%(self.current_event, tracking_new_event))
                self.current_event +=1
                reload_batch=True
                if len(self.event_steps)== self.current_event:
                    tracking_new_event = np.inf
                else:
                    tracking_new_event = self.event_steps[self.current_event]
                if verbose:
                    print('updated current event = %s / tracking_new_event = %s'%(self.current_event, tracking_new_event))
            #sampling tracking
            if (i%size_sampler ==0) or (reload_batch==True):
                print('loading a new batch of graphs')
                #sample stream of graphs with adequate labels
                #corresponds to a batch saved in memory which will be used iteratively (~independently) to update atoms 
                self.graphs, self.y = dataloader.data_streamer(self.data_path,sampler_batchsize,self.labels_by_events[self.current_event])
                size_sampler= len(self.graphs)
                reload_batch = False
            t = np.random.choice(range(size_sampler))
            if verbose:
                print('idx : %s / label : %s'%(t,self.y[t]))
            w = np.ones(self.number_Cs)/self.number_Cs
            best_w = w.copy()
            best_T = None
            best_loss_w= np.inf
            total_steps = 0            
            prev_loss_w = 10**(8)
            current_loss_w = 10**(7)
            convergence_criterion = np.inf
            outer_count=0
            saved_transport=None
            
            while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                
                prev_loss_w = current_loss_w 
                w,current_loss_w,saved_transport,inner_count= self.BCD_step(self.graphs[t],w, T_init=saved_transport, l2_reg=l2_reg,max_iter=max_iter_inner,eps=eps)
                total_steps+=inner_count
                if current_loss_w< best_loss_w:
                    best_w = w
                    best_T= saved_transport
                    best_loss_w= current_loss_w
                outer_count+=1
                if prev_loss_w !=0:                        
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w)
                else:
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w+10**(-15))
            self.log['loss'].append(best_loss_w)
            self.log['steps'].append(total_steps)
            
            self.numpy_update_atoms(self.graphs[t],best_w,best_T, lr_Cs)
            if verbose:
                #check new loss after update of atoms
                sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, best_w)
                check_loss,_=gwu.np_GW2(self.graphs[t],sum_ws_Cs,T_star=best_T)
                print('--- Sanity check / GW2 loss: ',check_loss)
                check_loss,_=gwu.np_GW2(self.graphs[t],sum_ws_Cs)
                print('--- Sanity check - NEW TRANSPORT /  GW2  loss: ',check_loss)
            
            if (((i%checkpoint_steps)==0) or (i==(steps-1))) :
                print('run checkpoint - i =%s '%i)
                #load a new batch of all samples to evaluate reconstruction
                checkpoint_C, checkpoint_label = dataloader.data_streamer(self.data_path,checkpoint_size,self.all_labels)
                if verbose:
                    print('checkpoint labels: %s/ first label = %s / second label =%s'%(np.unique(checkpoint_label,return_counts=True), np.unique(checkpoint_label[:checkpoint_size]),np.unique(checkpoint_label[-checkpoint_size:])))
                checkpoint_w,checkpoint_losses=self.numpy_GW_mixture_learning(checkpoint_C,l2_reg,eps=eps,max_iter= max_iter_inner)
                self.save_elements(save_settings=(not already_saved),local_step=i,checkpoint_a=checkpoint_w,checkpoint_losses=checkpoint_losses)
                already_saved=True
                
    def BCD_step(self,graph, w, T_init, l2_reg,eps,max_iter, OT_loss='square_loss'):
        """
        One step of the BCD algorithm to solve GW unmixing problem on a graph
        - refers to unmixing equation 2 and algorithm 1.
        Parameters
        ----------
        idx: (int) index of the graph in the dataset
        wt: (np.array) embeddings of the graph
        T_init: (np.array) matrix to initialize the solver of GW distance problem 
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        """
        sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, w)
        #Solve GW(Ck, \sum_s wk_s C_s)
        local_GW_loss,OT = gwu.np_GW2(graph,sum_ws_Cs,T_init=T_init)
        local_GW_loss-=l2_reg*np.sum(w**2)
        TCT = np.transpose(graph.dot(OT)).dot(OT)
        local_count = 0
        prev_criterion = 10**8
        curr_criterion = local_GW_loss   
        convergence_criterion = np.inf
        local_best_w = w.copy()
        best_loss = prev_criterion
        #Conditional gradient over wk given Tk
        while (convergence_criterion> eps) and (local_count<max_iter):
            prev_criterion=curr_criterion
            grad_w = np.zeros(self.number_Cs)
            for s in range(self.number_Cs):
                grad_w[s] = np.sum(self.Cs[s]*sum_ws_Cs*self.hhs - self.Cs[s]*TCT)
                grad_w[s] -= l2_reg*w[s]
                grad_w[s]*=2
            # Gradient direction : x= argmin_x x^T.grad_w
            x= np.zeros(self.number_Cs)
            sorted_idx = np.argsort(grad_w)#ascending order
            pos=0
            while (pos<self.number_Cs) and (grad_w[sorted_idx[pos]] == grad_w[sorted_idx[0]]) :
                x[sorted_idx[pos]] = 1
                pos+=1
            x/= pos
            #Line search step: solve argmin_{\gamma in (0,1)} a\gamma^2 + b\gamma +c
            sum_xs_Cs =  gwu.np_sum_scaled_mat(self.Cs,x)
            sum_xsws_Cs = sum_xs_Cs - sum_ws_Cs
            tr_xsws_xs = np.sum((sum_xsws_Cs*sum_xs_Cs)*self.hhs)
            tr_xsws_ws = np.sum((sum_xsws_Cs*sum_ws_Cs)*self.hhs)            
            a=tr_xsws_xs - tr_xsws_ws
            b= 2*tr_xsws_ws - 2*np.sum( sum_xsws_Cs*TCT)
            if l2_reg !=0:
                a -=l2_reg*np.sum((x-w)**2)
                b -= 2*l2_reg* (w.T).dot(x-w)
                #check_a, check_b= self.check_line_coefficients(Ct,wt,x,curr_criterion, T_star=Tt,l2_reg=l2_reg)
                #print('a: %s / check_a : %s'%(a,check_a))
                #print('b:%s / check_b: %s '%(b,check_b))
            if a>0:
                gamma = min(1, max(0, -b/(2*a)))
            elif a+b<0:
                gamma=1
            else:
                gamma=0            
            w+=gamma*(x-w)
            sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, w)
            curr_criterion+=a*(gamma**2) + b*gamma 
            convergence_criterion = abs(prev_criterion - curr_criterion)/prev_criterion
            
            local_count +=1
            if curr_criterion < best_loss:
                local_best_w = w
                best_loss = curr_criterion
        return  local_best_w,best_loss, OT,local_count
        
    def numpy_update_atoms(self,Ct,at,Tt, lr_Cs,OT_loss='square_loss', epsilon=10**(-15)):
        """
        Stochastic Gradient step on {C_s} with vanilla SGD
        """
        
        grad_Cs = np.zeros_like(self.Cs)
        sum_as_Cs = gwu.np_sum_scaled_mat(self.Cs, at)            
        for pos in range(self.number_Cs):
            grad_Cs[pos] += 2*at[pos]*(sum_as_Cs*self.hhs - (Ct.dot(Tt)).T.dot(Tt))            
            self.Cs[pos] -= lr_Cs*grad_Cs[pos]
            self.Cs[pos] = np.maximum(np.zeros((self.shape_Cs,self.shape_Cs)), (self.Cs[pos]+ self.Cs[pos].T )/2)
                
    
            
    
    def numpy_GW_mixture_learning(self,graphs,l2_reg,eps=10**(-5),max_iter= 1000, OT_loss = 'square_loss',seed=0):
        """
        GW mixture learning for a given state of dictionary - on provided sample of graphs 
        """
        np.random.seed(seed)
        K= len(graphs)
        #initialize embeddings w as uniform for each graph
        best_w = np.ones((K,self.number_Cs))/self.number_Cs
        best_loss = np.ones(K)*np.inf
        for k in range(K):
            saved_transport=None
            w= best_w[k].copy()
            curr_loss_w, convergence_criterion = 10**(8),np.inf
            outer_count=0
            #stacked_inner_counts =[]
            while (convergence_criterion >eps) and (outer_count<max_iter):
                outer_count+=1
                prev_loss_w = curr_loss_w
                w,curr_loss_w,saved_transport,inner_count =self.BCD_step(graphs[k], w, T_init=saved_transport, l2_reg=l2_reg,eps=eps,max_iter=max_iter, OT_loss='square_loss')
                #stacked_inner_counts.append(inner_count)
                if prev_loss_w !=0: 
                    convergence_criterion = abs(prev_loss_w - curr_loss_w)/abs(prev_loss_w)
                else:
                    convergence_criterion = abs(prev_loss_w - curr_loss_w)/abs(prev_loss_w+10**(-12))
                
                #print('current loss a: %s / convergence_criterion:%s'%(current_loss_a,convergence_criterion))
                if curr_loss_w < best_loss[k]:
                    best_w[k]= w.copy()
                    best_loss[k] = curr_loss_w
        print('[checkpoint results] number of graphs:%s / cumulated GW reconstruction error : %s'%(K,np.sum(best_loss)))
        return best_w, best_loss
    
    def numpy_GW_mixture_learning_latevalidation(self,val_sampling,val_seed, batchsize_bylabel, selected_labels,l2_reg,eps=10**(-5),max_iter= 1000, OT_loss ='square_loss', seed=0):
        """
        GW unmixing learning to handle the high variability of the number of nodes depending on classes. 
        """
        np.random.seed(seed)
        if val_sampling=='balanced':
            graphs,y=dataloader.data_streamer(self.data_path,batchsize_bylabel, selected_labels,balanced_shapes=True,sampling_seed=val_seed,return_idx = False)
        else:
            raise 'not implemented yet'
        n_track=len(self.tracked_Cs)
        print('len tracked Cs : %s / shape: %s'%(n_track,self.tracked_Cs[0].shape[0]))
        full_path = os.path.abspath('../')+'/%s/%s/'%(self.experiment_repo,self.experiment_name)
        np.save(full_path+'validation_y.npy', y)
        tracked_w = []
        tracked_losses= []
        for pos in range(n_track):
            print('computing unmixing for checkpoint / steps = %s / pos= %s'%( self.checkpoint_iter_[pos],pos) )
            self.Cs = self.tracked_Cs[pos]
            self.number_Cs= self.Cs.shape[0]
            self.shape_Cs = self.Cs.shape[-1]
            hs= np.ones(self.shape_Cs)/self.shape_Cs
            self.hhs = hs[:,None].dot(hs[None,:])
            self.diagh= np.diag(hs)
            K= len(graphs)
            #uniform init for unmixings
            local_best_w = np.ones((K,self.number_Cs))/self.number_Cs
            local_best_loss = np.ones(K)*np.inf
            for k in tqdm(range(K)):
                saved_transport=None
                curr_loss_w, convergence_criterion = 10**(7),np.inf
                w= local_best_w[k].copy()
                outer_count=0
                #stacked_inner_counts =[]
                while (convergence_criterion >eps) and (outer_count<max_iter):
                    outer_count+=1
                    prev_loss_w = curr_loss_w
                    w,curr_loss_w,saved_transport,inner_count =self.BCD_step(graphs[k], w, T_init=saved_transport, l2_reg=l2_reg,eps=eps,max_iter=max_iter, OT_loss='square_loss')
                    #stacked_inner_counts.append(inner_count)
                    if prev_loss_w !=0: 
                        convergence_criterion = abs(prev_loss_w - curr_loss_w)/abs(prev_loss_w)
                    else:
                        convergence_criterion = abs(prev_loss_w - curr_loss_w)/abs(prev_loss_w+10**(-12))
                    if curr_loss_w < local_best_loss[k]:
                        local_best_w[k]= w.copy()
                        local_best_loss[k] = curr_loss_w
            tracked_w.append(local_best_w)
            tracked_losses.append(local_best_w)
            #np.save(full_path+'validation_tracked_losses.npy', tracked_losses)
        return np.array(tracked_w), np.array(tracked_losses)
    def save_elements(self,save_settings=False,local_step=0,checkpoint_w=None,checkpoint_losses=None):
        """
        saving tracked events
        """
        path = os.path.abspath('../')+self.experiment_repo
        #print('path',path)
        if not os.path.exists(path+self.experiment_name):
            os.makedirs(path+self.experiment_name)
            print('made dir', path+self.experiment_name)
        if save_settings:
            pd.DataFrame(self.settings, index=self.settings.keys()).to_csv(path+'%s/settings'%self.experiment_name)
        
        if not self.save_chunks:
            self.tracked_w.append(checkpoint_w)
            self.tracked_losses.append(checkpoint_losses)
            self.tracked_Cs.append(self.Cs)
            # We save the full set of tracked elements
            np.save(path+'%s/tracked_Cs.npy'%self.experiment_name, np.array(self.tracked_Cs))
            np.save(path+'%s/tracked_w.npy'%self.experiment_name, np.array(self.tracked_w))
            np.save(path+'%s/tracked_losses.npy'%self.experiment_name, np.array(self.tracked_losses))
        else:
            np.save(path+'%s/Cs_checkpoint%s.npy'%(self.experiment_name,local_step), self.Cs)
            np.save(path+'%s/w_checkpoint%s.npy'%(self.experiment_name,local_step), checkpoint_w)
            np.save(path+'%s/losses_checkpoint%s.npy'%(self.experiment_name,local_step), checkpoint_losses)
            
        for key in self.log.keys():
            try:
                np.save(path+'%s/%s.npy'%(self.experiment_name,key), np.array(self.log[key]))
            except:
                print('bug for  log component: %s'%key)
                pass
        
    def load_elements(self, pos=[-1],path=None, save_chunks=False):
        """
        loading elements
        """
        if path is None:
            path = os.path.abspath('../')+self.experiment_repo
        else:
            path+='/%s/'%self.experiment_repo
        if not save_chunks:
            assert type(pos)==list
            assert len(pos)==1
            self.tracked_Cs = np.load(path+'%s/tracked_Cs.npy'%self.experiment_name)
            self.Cs  = self.tracked_Cs[pos]
            self.number_Cs= self.Cs.shape[0]
            self.shape_Cs = self.Cs.shape[-1]
            hs= np.ones(self.shape_Cs)/self.shape_Cs
            self.hhs = hs[:,None].dot(hs[None,:])
            self.diagh= np.diag(hs)
        if save_chunks:
            assert type(pos)==list
            self.tracked_Cs=[]
            self.checkpoint_pos=[]
            self.checkpoint_iter_ =pos 
            for elem in pos:
                local_Cs = np.load(path+'%s/Cs_checkpoint%s.npy'%(self.experiment_name,elem))
                self.tracked_Cs.append(local_Cs)
                self.checkpoint_pos.append(pos)
                
                