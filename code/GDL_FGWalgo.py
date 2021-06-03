"""
@author: cvincentcuaz
"""

from tqdm import tqdm

import GDL_utils as gwu
import numpy as np
import dataloader
import os
import pandas as pd
import pylab as pl

#%%


# =============================================================================
# Fused Gromov Wasserstein Graph Dictionary Learning
#   1/  FGW_Unsupervised_DictionaryLearning : Learning GDL over a fixed dataset 
#       of labeled graphs on several epochs with atoms {C_s,A_s}
#       > C_s refers to the structures 
#       > A_s refers to the features
#       As in the supplementary material
#
#   3/  Online_FGW_Unsupervised_DictionaryLearning: 
#    Online Learning of our GDL on the dataset TRIANGLES of labeled graphs
# =============================================================================

class FGW_Unsupervised_DictionaryLearning():
    """
        Graph Dictionary Learning for graph datasets  node features
    """
    def __init__(self,
                 dataset_name:str,
                 mode:str, 
                 number_Cs:int, 
                 shape_Cs:int, 
                 alpha:float,
                 experiment_repo:str, 
                 experiment_name:str,
                 data_path:str='../data/'):
        """
        Parameters
        ----------
        dataset_name : name of the dataset to experiment on. To match our data loaders it is restricted to ['imdb-b','imdb-m','balanced_clustertoy','clustertoy2C']
        mode : representations for input graphs. (e.g) 'ADJ':adjacency / 'SP': shortest path 
        number_Cs : number of atoms in our dictionary
        alpha: alpha parameter of FGW distance
        shape_Cs : number of nodes similar for each atom
        experiment_repo : subrepository to save results of the experiment 
        experiment_name : subrepository to save results of the experiment under the 'experiment repo' repository
        data_path : path where data is. The default is '../data/'.
        """
        assert dataset_name in ['mutag','enzymes', 'ptc','bzr','cox2','protein']
        self.dataset_name= dataset_name
        self.mode = mode
        
        str_to_method = {'ADJ': 'adjacency', 'SP':'shortest_path','LAP':'laplacian'}
        if dataset_name in ['mutag','ptc']:
            X,self.y=dataloader.load_local_data(data_path,dataset_name, one_hot=True)
        elif dataset_name in [ 'enzymes','protein','cox2','bzr']:
            X,self.y=dataloader.load_local_data(data_path,dataset_name, one_hot=False)
            
        self.A_target  = [np.array(X[t].values(),dtype=np.float64) for t in range(X.shape[0])]
            
        if self.mode in str_to_method.keys():
            self.C_target = [np.array(X[t].distance_matrix(method=str_to_method[mode]),dtype=np.float64) for t in range(X.shape[0])]
        else:
            raise 'unknown graph mode'
        self.dataset_size = len(self.C_target)
        self.number_Cs = number_Cs
        self.shape_Cs = shape_Cs
    
        self.shapes= np.array([X.shape[0] for X in self.C_target])
        self.alpha = alpha
        self.d = self.A_target[0].shape[-1]
        self.experiment_repo= experiment_repo
        self.experiment_name = experiment_name
        
    def initialize_atoms(self,
                         init_mode_atoms:int=1,
                         seed:int=0):
        """
        Initialize atoms {C_s,A_s}: 0= randomly following uniform distribution / 1= through sampling in the dataset
        """
        np.random.seed(seed)
        if init_mode_atoms==0:
            # initialize Cs components with uniform distribution
            init_Cs = np.random.uniform(low= 10**(-15), high=1, size= (self.number_Cs,self.shape_Cs, self.shape_Cs ))
            self.Cs = 0.5* (init_Cs + init_Cs.transpose((0,2,1))) 
            self.As = np.random.uniform(low= 10**(-15),high=1, size=(self.number_Cs,self.shape_Cs, self.d))
        
        elif init_mode_atoms ==1:
            # Sample from the dataset
            shape_idx = np.argwhere(self.shapes==self.shape_Cs)[:,0]
            if shape_idx.shape[0]>0:
                print('find samples with good shape')
                if shape_idx.shape[0]>self.number_Cs:
                    print('find enough samples with good shapes')
                    warmstart_idx = np.random.choice(shape_idx,size=self.number_Cs,replace=False)
                    print('SELECTED IDX FOR INITIALIZATION : ', warmstart_idx)
                    self.Cs= np.stack([self.C_target[idx] for idx in warmstart_idx])
                    #print('SELECTED INITIAL ATOMS: ', self.Cs)
                    self.As= np.stack([self.A_target[idx] for idx in warmstart_idx])
                    #print('SELECTED INITIAL FEATURES:', self.As)
                else:
                    warmstart_idx = np.random.choice(shape_idx, size=self.number_Cs, replace=True)
                    #add noise to existing ones to reach the required number of atoms
                    self.Cs= np.stack([self.graphs[idx] for idx in warmstart_idx])
                    mean_ = np.mean(self.Cs)
                    std_ = np.std(self.Cs)
                    noise = np.random.normal(loc= mean_, scale = std_, size= (self.number_Cs,self.shape_Cs, self.shape_Cs))
                    noise = 0.5* (noise+ noise.transpose((0,2,1))) #add symmetric noise
                    self.Cs+=noise
                    self.As= np.stack([self.A_target[idx] for idx in warmstart_idx])
                    noise = np.random.uniform(low= -0.1, high=0.1, size= (self.number_Cs,self.shape_Cs, self.d))
                    self.As+=noise
            else:
                raise 'There is no graph with adequate shape in the dataset - change the value of init_mode_atoms from 1 to 0 for random initialization '
        
        
    def initialize_optimizer(self):
        """
        Initialize Adam Optimizer for Stochastic algorithm
        """
        self.Cs_adam_m = np.zeros((self.number_Cs, self.shape_Cs,self.shape_Cs))#Initialize first  moment vector
        self.Cs_adam_v = np.zeros((self.number_Cs, self.shape_Cs, self.shape_Cs))#Initialize second moment vector
        self.Cs_adam_count = np.zeros((self.number_Cs))
        self.As_adam_m = np.zeros((self.number_Cs, self.shape_Cs,self.d))#Initialize first  moment vector
        self.As_adam_v = np.zeros((self.number_Cs, self.shape_Cs, self.d))#Initialize second moment vector
        self.As_adam_count = np.zeros((self.number_Cs))
        
            
    
    def Stochastic_GDL(self,l2_reg:float,
                       eps:float, 
                       max_iter_outer:int,
                       max_iter_inner:int,
                       lr_Cs:float,
                       lr_As:float,
                       batch_size:int,
                       beta_1:float=0.9,
                       beta_2:float=0.99, 
                       epochs:int=100,
                       algo_seed:int=0,
                       init_mode_atoms:int=1,
                       checkpoint_timestamp:int=None,
                       visualization_frame:int=None,
                       verbose:bool=False):
        """
        Stochastic Algorithm to learn dictionary atoms with FGW loss {C_s,A_s}
        cf supplementary material

        Parameters
        ----------
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter_outer : maximum number of outer loop iterations of our BCD algorithm            
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        lr_Cs : Initial learning rate of Adam optimizer for atoms structure Cs
        lr_As : Initial learning rate of Adam optimizer for atoms features As
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
        # Save settings used for learning the dictionary
        self.settings = {'number_atoms':self.number_Cs, 'shape_atoms': self.shape_Cs, 
                         'eps':eps,'max_iter_outer':max_iter_outer,'max_iter_inner':max_iter_inner,
                         'lr_Cs':lr_Cs, 'lr_As':lr_As,'epochs':epochs,'init_mode_atoms':init_mode_atoms,
                         'batch_size':batch_size,'algo_seed':algo_seed,
                         'beta1':beta_1, 'beta2':beta_2,'alpha':self.alpha, 'l2_reg':l2_reg}
        self.initialize_atoms(init_mode_atoms, algo_seed)
        self.initialize_optimizer()
        self.log ={}
        self.log['loss']=[]
        
        if not (checkpoint_timestamp is None):
            self.checkpoint_best_loss = np.inf
            self.checkpoint_atoms = self.Cs.copy() # save atoms state if the reconstruction over the whole dataset is minimized at current epoch
            self.log['checkpoint_loss']=[]
        hs= np.ones(self.shape_Cs)/self.shape_Cs
        self.hhs = hs[:,None].dot(hs[None,:])
        self.diagh= np.diag(hs)
        self.reshaped_hs = hs[:,None].dot(np.ones((1,self.d)))
        already_saved= False
        iter_by_epoch= int(self.dataset_size/batch_size)+1
        for epoch in tqdm(range(epochs)):
            cumulated_loss_over_epoch = 0
            for _ in range(iter_by_epoch):
                #batch sampling
                batch_idx = np.random.choice(range(self.dataset_size), size=batch_size, replace=False)
                w = np.ones((batch_size,self.number_Cs))/self.number_Cs
                best_w = w.copy()
                best_T = [None]*batch_size
                batch_best_loss= 0
                # BCD algorithm to solve FGW unmixing problem over all graphs of the batch independently
                for k,idx in enumerate(batch_idx):
                    prev_loss_w = 10**(8)
                    current_loss_w = 10**(7)
                    best_loss_w = np.inf
                    convergence_criterion = np.inf
                    outer_count=0
                    saved_transport=None
                    while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                        prev_loss_w = current_loss_w
                        w[k],current_loss_w,saved_transport,_= self.BCD_step(self.C_target[idx], self.A_target[idx],w[k], T_init=saved_transport, l2_reg=l2_reg,max_iter=max_iter_inner,eps=eps)
                        
                        if current_loss_w< best_loss_w:
                            best_w[k] = w[k]
                            best_T[k]= saved_transport
                            best_loss_w= current_loss_w
                        outer_count+=1
                        if prev_loss_w !=0:
                            convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w)
                        else:
                            convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w+10**(-15))
                            
                    batch_best_loss+= best_loss_w
                cumulated_loss_over_epoch+=batch_best_loss
                
                self.log['loss'].append(batch_best_loss)
                Ct = [self.C_target[idx] for idx in batch_idx]
                At = [self.A_target[idx] for idx in batch_idx]
                #Stochastic update of atoms
                self.atoms_stochastic_update(Ct,At,best_w,best_T,beta_1,beta_2, lr_Cs,lr_As, batch_size,verbose=verbose)
            if verbose:
                print('epoch : %s / cumulated_loss_over_epoch : %s '%(epoch, cumulated_loss_over_epoch))
            
            # Visualization learning evolution 
            if not (visualization_frame is None):
                if (epoch%visualization_frame ==0):
                    pl.plot(self.log['loss'])
                    pl.xlabel('updates')
                    pl.ylabel('FGW loss')
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
                        self.checkpoint_AS = self.As.copy()
                        self.save_checkpoint(checkpoint_unmixing,checkpoint_losses)
                        
            if not already_saved:
                print('saved settings:', self.settings)
                self.save_elements(save_settings=True)
                already_saved=True
            else:
                self.save_elements(save_settings=False)
    
    def BCD_step(self,C, A,w, T_init, l2_reg,eps,max_iter):
        """
        One step of the BCD algorithm to solve FGW unmixing problem on a labeled graph
        - refers to unmixing equation 2 and algorithm 1 for Algorithm on unlabeled graphs
        - Detailed algorithm for Fused Gromov-Wasserstein is provided in the supplementary material.
        
        Parameters
        ----------
        C: (np.array) input graph structure
        A: (np.array) input graph features
        wt: (np.array) embeddings of the graph
        T_init: (np.array) matrix to initialize the solver of GW distance problem 
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        """
        sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, w)
        sum_ws_As = gwu.np_sum_scaled_mat(self.As, w)
        # Solve FGW problem to get Tt with fixed wt
        local_FGW_loss,Tt = gwu.numpy_FGW_loss(C,sum_ws_Cs,A,sum_ws_As,p=None,q=None,alpha=self.alpha,T_init=T_init)
        local_FGW_loss-= l2_reg*np.sum(w**2)
        TCT = np.transpose(C.dot(Tt)).dot(Tt)
        TA = (Tt.T).dot(A)
        local_count = 0
        
        prev_criterion = 10**8
        curr_criterion = local_FGW_loss
        convergence_criterion = np.inf
        local_best_loss=np.inf
        #CG algorithm to get optimal wt given Tt
        while (convergence_criterion> eps) and (local_count<max_iter):
            prev_criterion = curr_criterion
            #compute gradients of FGW objective w.r.t wt
            grad_wt = np.zeros(self.number_Cs)
            for s in range(self.number_Cs):
                grad_wt[s] = self.alpha*np.sum(self.Cs[s]*sum_ws_Cs*self.hhs - self.Cs[s]*TCT)
                grad_wt[s] += (1-self.alpha)* np.trace(self.diagh.dot(sum_ws_As).dot(self.As[s].T) - Tt.T.dot(A).dot(self.As[s].T))
                grad_wt[s] -= l2_reg*w[s]
                grad_wt[s]*=2
            
            # CG direction
            x= np.zeros(self.number_Cs)
            sorted_idx = np.argsort(grad_wt)#ascending order
            pos=0
            while (pos<self.number_Cs) and (grad_wt[sorted_idx[pos]] == grad_wt[sorted_idx[0]]) :
                x[sorted_idx[pos]] = 1
                pos+=1
            x/= pos
            #Line search step: find argmin_{\gamma in (0,1)} a*gamma^2+ b*gamma +c
            # Literal expressions are used to gain time. They can be found in supplementary material (equations 52. 53.)
            sum_xs_Cs =  gwu.np_sum_scaled_mat(self.Cs,x)
            sum_xsws_Cs = sum_xs_Cs - sum_ws_Cs
            sum_xs_As = gwu.np_sum_scaled_mat(self.As,x)
            sum_xsws_As  = sum_xs_As - sum_ws_As
            trC_xsws_xs = np.sum((sum_xsws_Cs*sum_xs_Cs)*self.hhs)
            trC_xsws_ws = np.sum((sum_xsws_Cs*sum_ws_Cs)*self.hhs)
            trA_xsws_xs = np.sum((sum_xsws_As*sum_xs_As)*self.reshaped_hs)
            trA_xsws_ws = np.sum((sum_xsws_As*sum_ws_As)*self.reshaped_hs)
            a=self.alpha*(trC_xsws_xs - trC_xsws_ws)+(1-self.alpha)*(trA_xsws_xs - trA_xsws_ws)
            
            b= self.alpha*(trC_xsws_ws - np.sum( sum_xsws_Cs*TCT))
            b+= (1-self.alpha)*(trA_xsws_ws- np.sum(sum_xsws_As*TA))
            b*=2
            
            if l2_reg !=0:
                a -=l2_reg*np.sum((x-w)**2)
                b -= 2*l2_reg* (w.T).dot(x-w)
            #check_a,check_b= self.check_line_coefficients(C,A,w,x,curr_criterion, T_star=Tt,l2_reg=l2_reg)
            #print('a : %s / check_a: %s '%(a,check_a))
            #print('b: %s /check_b: %s '%(b,check_b))
            if a>0:
                gamma = min(1, max(0, -b/(2*a)))
            elif a+b<0:
                gamma=1
            else:
                gamma=0
            w +=gamma*(x-w)
            sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, w)
            sum_ws_As = gwu.np_sum_scaled_mat(self.As, w)
            curr_criterion += a*(gamma**2) + b*gamma
            local_count +=1
            if local_best_loss > curr_criterion:
                local_best_loss=curr_criterion
                local_best_w = w
            if prev_criterion!=0:
                convergence_criterion = abs(prev_criterion - curr_criterion)/abs(prev_criterion)
            else:
                convergence_criterion = abs(prev_criterion - curr_criterion)/abs(prev_criterion+10**(-15))

        
        return  local_best_w,local_best_loss, Tt,local_count
    
         
    def atoms_stochastic_update(self,batch_C,batch_A,batch_w,batch_T,beta_1,beta_2, lr_Cs,lr_As,batch_size,proj='nsym',epsilon=10**(-15),verbose=False):
        """
        
        Stochastic gradient step on atoms {C_s,A_s}

        Parameters
        ----------
        batch_C : (list) batch of graphs
        batch_A: (list) batch of corresponding features
        batch_w : (np.array) batch of corresponding embeddings
        batch_T : (np.array) batch of corresponding optimal transport plans
        beta_1 : (float) Adam parameter
        beta_2 : (float) Adam parameter
        lr : (float) learning rate
        batch_size : (int) batch size
        proj : (str) Projection of atoms - in {'nsym','sym'}. The default is 'nsym'.
        epsilon : (float) value to avoid division by 0. The default is 10**(-15).
        """
        
        grad_Cs = np.zeros_like(self.Cs)
        grad_As = np.zeros_like(self.As)
        for k in range(batch_size):
            sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, batch_w[k])
            sum_ws_As = gwu.np_sum_scaled_mat(self.As,batch_w[k])
            GW_generic_terms = self.alpha*sum_ws_Cs*self.hhs - (batch_C[k].dot(batch_T[k])).T.dot(batch_T[k])
            W_generic_terms = (1-self.alpha)*(self.diagh.dot(sum_ws_As) - batch_T[k].T.dot(batch_A[k]))
            for pos in range(self.number_Cs):
                grad_Cs[pos] += batch_w[k,pos]*GW_generic_terms
                grad_As[pos] += batch_w[k,pos]*W_generic_terms
                
        for pos in range(self.number_Cs):
            
            grad_Cs[pos]*=2/batch_size
            grad_As[pos]*=2/batch_size
            
            if verbose:
                print('gradients Cs (norm):', np.linalg.norm(grad_Cs[pos], ord=2))
                print('gradients As (norm):', np.linalg.norm(grad_As[pos],ord=2))
        
            C_t=self.Cs_adam_count[pos]+1
            C_m_k = beta_1*self.Cs_adam_m[pos] + (1-beta_1)*grad_Cs[pos]
            C_v_k = beta_2*self.Cs_adam_v[pos] + (1-beta_2)*(grad_Cs[pos]**2)
            unbiased_m_k = C_m_k/ (1-beta_1**C_t)
            unbiased_v_k = C_v_k/(1-beta_2**C_t)
            self.Cs[pos] -= lr_Cs*unbiased_m_k/ (np.sqrt(unbiased_v_k)+epsilon)
            if proj=='nsym':
                self.Cs[pos] = np.maximum(np.zeros((self.shape_Cs,self.shape_Cs)), (self.Cs[pos]+ self.Cs[pos].T )/2)
            elif proj=='sym':
                self.Cs[pos] = (self.Cs[pos]+ self.Cs[pos].T )/2
            else:
                raise 'unknown projection of atoms'
            A_t=self.As_adam_count[pos]+1
            A_m_k = beta_1*self.As_adam_m[pos] + (1-beta_1)*grad_As[pos]
            A_v_k = beta_2*self.As_adam_v[pos] + (1-beta_2)*(grad_As[pos]**2)
            unbiased_m_k = A_m_k/ (1-beta_1**A_t)
            unbiased_v_k = A_v_k/(1-beta_2**A_t)
            self.As[pos] -= lr_As*unbiased_m_k/ (np.sqrt(unbiased_v_k)+epsilon)
            
            #update Adam
            self.Cs_adam_count[pos]= C_t
            self.Cs_adam_m[pos] = C_m_k
            self.Cs_adam_v[pos] = C_v_k
            self.As_adam_count[pos] = A_t
            self.As_adam_m[pos] = A_m_k
            self.As_adam_v[pos]= A_v_k
            
    def GDL_unmixing(self,l2_reg:float,eps:float,max_iter_outer:int,max_iter_inner:int,verbose:bool=False):
        """
        For each labeled graph - solve independently the FGW unmixing problem with BCD algorithm
        
        Parameters
        ----------
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter_outer : maximum number of outer loop iterations of our BCD algorithm
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        verbose : Check the good evolution of the loss. The default is False.
        """
        
        w = np.ones((self.dataset_size,self.number_Cs))/self.number_Cs
        best_w = w.copy()
        best_losses= np.zeros(self.dataset_size)
        for t in range(self.dataset_size):
            saved_transport=None
            
            prev_loss_w = 10**(8)
            current_loss_w = 10**(7)
            convergence_criterion = np.inf
            best_loss_w = np.inf
            outer_count=0
            while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                outer_count+=1
                prev_loss_w = current_loss_w
                w[t],current_loss_w,saved_transport,inner_count = self.BCD_step(self.C_target[t], self.A_target[t], w[t], saved_transport, l2_reg,max_iter=max_iter_inner,eps=eps)
                if prev_loss_w!=0:
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w)
                else:
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w+10**(-12))
                if current_loss_w < best_loss_w:
                    best_w[t]= w[t]
                    best_loss_w = current_loss_w
            best_losses[t]=best_loss_w
        return best_w, best_losses
    
    
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
        np.save(path+'/checkpoint_As.npy', self.checkpoint_As)
        
        
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
        np.save(path+'%s/As.npy'%self.experiment_name, self.As)

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
            self.As = np.load(path+'/As.npy')
            self.number_atoms= self.Cs.shape[0]
            self.shape_atoms = self.Cs.shape[-1]
            self.d = self.As.shape[-1]
            hs= np.ones(self.shape_atoms)/self.shape_atoms
            self.hhs = hs[:,None].dot(hs[None,:])
            self.diagh= np.diag(hs)
            
            self.reshaped_hs = hs[:,None].dot(np.ones((1,self.d)))
        else:
            self.Cs= np.load(path+'/checkpoint_Cs.npy')
            self.As= np.load(path+'/checkpoint_As.npy')
            self.number_atoms= self.Cs.shape[0]
            self.shape_atoms = self.Cs.shape[-1]
            self.d = self.As.shape[-1]
            hs= np.ones(self.shape_atoms)/self.shape_atoms
            self.hhs = hs[:,None].dot(hs[None,:])
            self.diagh= np.diag(hs)            
            self.reshaped_hs = hs[:,None].dot(np.ones((1,self.d)))
#%% ONLINE Learning - cf section 4.3 of the paper with TRIANGLES dataset

class Online_FGW_Unsupervised_DictionaryLearning():
     
    """
    Online Learning of our GDL on the dataset TRIANGLES- with features (FGW distance). 
    """
    
    def __init__(self,
                 dataset_name:str,
                 mode:str, 
                 number_Cs:int, 
                 shape_Cs:int, 
                 alpha:float,
                 experiment_repo:str, 
                 experiment_name:str,
                 data_path:str='../data/'):
        """
        Parameters
        ----------
        dataset_name : name of the dataset to experiment on. To match our data loaders it is restricted to ['imdb-b','imdb-m','balanced_clustertoy','clustertoy2C']
        mode : representations for input graphs. (e.g) 'ADJ':adjacency / 'SP': shortest path 
        number_Cs : number of atoms in our dictionary
        alpha: alpha parameter of FGW distance
        shape_Cs : number of nodes similar for each atom
        experiment_repo : subrepository to save results of the experiment 
        experiment_name : subrepository to save results of the experiment under the 'experiment repo' repository
        data_path : path where data is. The default is '../data/'.
        """  
        assert dataset_name in ['triangles'] # Sampler has to be adapted for new datasets
        self.dataset_name= dataset_name
        self.mode = mode
        self.number_Cs = number_Cs
        self.shape_Cs = shape_Cs
        self.alpha=alpha
        str_to_method = {'ADJ': 'adjency', 'SP':'shortest_path','LAP':'laplacian'}
    
        #Stock All the dataset in memory
        np.random.seed(0)
        full_X,full_y = dataloader.load_local_data(data_path,'triangles')
        self.dataset_size = len(full_X)
        self.C_target = [np.array(full_X[idx].distance_matrix(method=str_to_method[self.mode]), dtype = np.float64) for idx in range(self.dataset_size) ]
        self.y = np.array(full_y, dtype= np.float64)
        self.shapes = np.array([X.shape[0] for X in self.C_target])
        self.A_target = [np.array(full_X[idx].values(), dtype=np.float64)[:,None] for idx in range(self.dataset_size)]
        self.idx_by_label = [np.argwhere(self.y==i)[:,0] for i in range(1,11)]
        
        self.d = self.A_target[0].shape[-1]
        self.experiment_repo= experiment_repo
        self.experiment_name = experiment_name
    
    def initialize_atoms(self,algo_seed:int=0):
        """
        Initialize atoms {C_s,A_s}: Fixed at random for these experriments i.e = 0
        """
        np.random.seed(algo_seed)
        
        # initialize Cs components with uniform distribution
        init_Cs = np.random.uniform(low= 10**(-15), high=1, size= (self.number_Cs,self.shape_Cs, self.shape_Cs ))
        self.Cs = 0.5* (init_Cs + init_Cs.transpose((0,2,1))) 
        self.As = np.random.uniform(low= 10**(-15),high=1, size=(self.number_Cs,self.shape_Cs, self.d))

    def triangles_streaming_mode(self, streaming_mode:int=0):
        """
        Monitoring events based on labels as depicted in the paper.
        Group A = labels {7,6,5,4}
        Group B= labels {8,9,10}
        Group C = labels {3,2,1}
        """
        if streaming_mode ==0:
            
            self.labels_by_events =[[7,6,5,4],[8,9,10],[3,2,1]]
        else:
            raise 'unknown streaming mode'
    
    def Online_Learning_triangles(self,
                                  l2_reg:float,
                                  eps:float = 10**(-5),
                                  sampler_batchsize:int=100,
                                  checkpoint_size:int=2000, 
                                  max_iter_outer:int=10,
                                  max_iter_inner:int=100,
                                  lr_Cs:float= 0.01,
                                  lr_As:float=0.01,
                                  steps:int = 1000,
                                  algo_seed:int=0,
                                  event_steps:list=[10000], 
                                  streaming_mode:int=0,
                                  checkpoint_steps:int= 100,
                                  save_chunks:bool=False,
                                  verbose=False):
        """
        Online Learning of our Fused-GW GDL by streams of the full dataset TRIANGLES
        
        Stochastic Algorithm to learn dictionary atoms with the FGW loss on the fly
        
        - refers to Equation 4 in the paper and algorithm 2. + Experiments ran in section 4.3

        Parameters
        ----------
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        sampler_batchsize: number of graphs stored in memory. refreshed after everyone contributed to SGD update one by one.
        checkpoint_size: number of graphs randomly sampled to evaluate unmixing reconstruction
        max_iter_outer : maximum number of outer loop iterations of our BCD algorithm            
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        lr_Cs : learning rate for atoms Cs
        lr_As: learning rate for atoms As
        batch_size : batch size 
        steps : number of updates of dictionaries. 
        algo_seed : initialization random seed
        event_steps: list of iterations index to make a change occur in the stream (here change of class)
        streaming_mode: used to control which class is streamed first
        verbose : Check the good evolution of the loss. The default is False.
        """
        self.settings = {'number_Cs':self.number_Cs, 'shape_Cs': self.shape_Cs, 'eps':eps,'max_iter_outer':max_iter_outer,'max_iter_inner':max_iter_inner,'lr_Cs':lr_Cs, 
                         'steps':steps,'algo_seed':algo_seed, 'alpha':self.alpha, 
                         'lr_As':lr_As,'l2_reg':l2_reg,'checkpoint_steps':checkpoint_steps }
        self.initialize_atoms( algo_seed)
        self.save_chunks=save_chunks
        self.event_steps = event_steps
        print('STEPS= %s / EVENT STEPS: %s'%(steps,event_steps))
        self.triangles_streaming_mode(streaming_mode)
        self.log ={}
        self.log['loss']=[]
        self.log['steps']=[]
        if not self.save_chunks:
            self.tracked_a=[]            
            self.tracked_losses = []
            self.tracked_Cs =[]
            self.tracked_As= []
            
        hs= np.ones(self.shape_Cs)/self.shape_Cs
        self.hhs = hs[:,None].dot(hs[None,:])
        self.diagh= np.diag(hs)
        self.reshaped_hs = hs[:,None].dot(np.ones((1,self.d)))
        
        already_saved= False
        self.current_event = 0
        tracking_new_event = self.event_steps[self.current_event]
        for i in tqdm(range(steps)):
            if i > tracking_new_event:
                if verbose:
                    print('initial current event = %s / tracking_new_event = %s'%(self.current_event, tracking_new_event))
                self.current_event +=1
                if len(self.event_steps)== self.current_event:
                    tracking_new_event = np.inf
                else:
                    tracking_new_event = self.event_steps[self.current_event]
                if verbose:
                    print('updated current event = %s / tracking_new_event = %s'%(self.current_event, tracking_new_event))
                
            label_t = np.random.choice(self.labels_by_events[self.current_event])
            t = np.random.choice(self.idx_by_label[label_t-1])
            if verbose:
                print('sampled label = %s / t = %s'%(label_t,t))
            w = np.ones(self.number_Cs)/self.number_Cs
            best_w = w.copy()
            best_T = None
            best_loss= np.inf
            total_steps = 0
            
            prev_loss_w = 10**(8)
            current_loss_w = 10**(7)
            convergence_criterion = np.inf
            outer_count=0
            saved_transport=None
            Ct = self.C_target[t]
            At = self.A_target[t]
            while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                prev_loss_w = current_loss_w 
                w,current_loss_w,saved_transport,inner_count= self.BCD_step(Ct,At,w, T_init=saved_transport, l2_reg=l2_reg,max_iter=max_iter_inner,eps=eps)
                total_steps+=inner_count              
                if current_loss_w< best_loss:
                    best_w = w
                    best_T= saved_transport
                    best_loss= current_loss_w
                outer_count+=1
                if prev_loss_w !=0:
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w)
                else:
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w+10**(-15))
                
            self.log['loss'].append(best_loss)
            self.log['steps'].append(total_steps)
            self.SGD_update_atoms(Ct,At,best_w,best_T,lr_Cs,lr_As)
                    
            if (((i%checkpoint_steps)==0) or (i==(steps-1))) :
                
                checkpoint_a,checkpoint_losses=self.numpy_FGW_FW_mixture_learning_l2reg_v2(l2_reg,eps,max_iter_outer,max_iter_inner)
                self.save_elements(save_settings=(not already_saved),local_step=i,checkpoint_a=checkpoint_a,checkpoint_losses=checkpoint_losses)
                already_saved=True
                
                     
    def BCD_step(self,
                 C:np.array, 
                 A:np.array,
                 w:np.array, 
                 T_init:np.array, 
                 l2_reg:float,
                 eps:float,
                 max_iter:int):
        """
        One step of the BCD algorithm to solve FGW unmixing problem on a labeled graph
        - refers to unmixing equation 2 and algorithm 1 for Algorithm on unlabeled graphs
        - Detailed algorithm for Fused Gromov-Wasserstein is provided in the supplementary material.
        
        Parameters
        ----------
        C: (np.array) input graph structure
        A: (np.array) input graph features
        wt: (np.array) embeddings of the graph
        T_init: (np.array) matrix to initialize the solver of GW distance problem 
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        """
        sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, w)
        sum_ws_As = gwu.np_sum_scaled_mat(self.As, w)
        # Solve FGW problem to get Tt with fixed wt
        local_FGW_loss,Tt = gwu.numpy_FGW_loss(C,sum_ws_Cs,A,sum_ws_As,p=None,q=None,alpha=self.alpha,T_init=T_init)
        local_FGW_loss-= l2_reg*np.sum(w**2)
        TCT = np.transpose(C.dot(Tt)).dot(Tt)
        TA = (Tt.T).dot(A)
        local_count = 0
        
        prev_criterion = 10**8
        curr_criterion = local_FGW_loss
        convergence_criterion = np.inf
        local_best_loss=np.inf
        #CG algorithm to get optimal wt given Tt
        while (convergence_criterion> eps) and (local_count<max_iter):
            prev_criterion = curr_criterion
            #compute gradients of FGW objective w.r.t wt
            grad_wt = np.zeros(self.number_Cs)
            for s in range(self.number_Cs):
                grad_wt[s] = self.alpha*np.sum(self.Cs[s]*sum_ws_Cs*self.hhs - self.Cs[s]*TCT)
                grad_wt[s] += (1-self.alpha)* np.trace(self.diagh.dot(sum_ws_As).dot(self.As[s].T) - Tt.T.dot(A).dot(self.As[s].T))
                grad_wt[s] -= l2_reg*w[s]
                grad_wt[s]*=2
            
            # CG direction
            x= np.zeros(self.number_Cs)
            sorted_idx = np.argsort(grad_wt)#ascending order
            pos=0
            while (pos<self.number_Cs) and (grad_wt[sorted_idx[pos]] == grad_wt[sorted_idx[0]]) :
                x[sorted_idx[pos]] = 1
                pos+=1
            x/= pos
            #Line search step: find argmin_{\gamma in (0,1)} a*gamma^2+ b*gamma +c
            # Literal expressions are used to gain time. They can be found in supplementary material (equations 52. 53.)
            sum_xs_Cs =  gwu.np_sum_scaled_mat(self.Cs,x)
            sum_xsws_Cs = sum_xs_Cs - sum_ws_Cs
            sum_xs_As = gwu.np_sum_scaled_mat(self.As,x)
            sum_xsws_As  = sum_xs_As - sum_ws_As
            trC_xsws_xs = np.sum((sum_xsws_Cs*sum_xs_Cs)*self.hhs)
            trC_xsws_ws = np.sum((sum_xsws_Cs*sum_ws_Cs)*self.hhs)
            trA_xsws_xs = np.sum((sum_xsws_As*sum_xs_As)*self.reshaped_hs)
            trA_xsws_ws = np.sum((sum_xsws_As*sum_ws_As)*self.reshaped_hs)
            a=self.alpha*(trC_xsws_xs - trC_xsws_ws)+(1-self.alpha)*(trA_xsws_xs - trA_xsws_ws)
            
            b= self.alpha*(trC_xsws_ws - np.sum( sum_xsws_Cs*TCT))
            b+= (1-self.alpha)*(trA_xsws_ws- np.sum(sum_xsws_As*TA))
            b*=2
            
            if l2_reg !=0:
                a -=l2_reg*np.sum((x-w)**2)
                b -= 2*l2_reg* (w.T).dot(x-w)
            #check_a,check_b= self.check_line_coefficients(C,A,w,x,curr_criterion, T_star=Tt,l2_reg=l2_reg)
            #print('a : %s / check_a: %s '%(a,check_a))
            #print('b: %s /check_b: %s '%(b,check_b))
            if a>0:
                gamma = min(1, max(0, -b/(2*a)))
            elif a+b<0:
                gamma=1
            else:
                gamma=0
            w +=gamma*(x-w)
            sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, w)
            sum_ws_As = gwu.np_sum_scaled_mat(self.As, w)
            curr_criterion += a*(gamma**2) + b*gamma
            local_count +=1
            if local_best_loss > curr_criterion:
                local_best_loss=curr_criterion
                local_best_w = w
            if prev_criterion!=0:
                convergence_criterion = abs(prev_criterion - curr_criterion)/abs(prev_criterion)
            else:
                convergence_criterion = abs(prev_criterion - curr_criterion)/abs(prev_criterion+10**(-15))

        return  local_best_w,local_best_loss, Tt,local_count
    
    def SGD_update_atoms(self,Ct,At,wt,Tt,lr_Cs,lr_As,epsilon=10**(-15)):
        """
        Stochastic gradient step on atoms {C_s, A_s} for streamed graph (Ct,At) with embedding wt and Tt with associated OT
        
        Parameters
        ----------
        Ct: (np.array) input graph structure at time t
        At: (np.array) input graph features at time t
        wt: (np.array) embeddings of the graph at time t
        Tt: (np.array) corresponding OT matrix
        lr_Cs : learning rate for atoms Cs
        lr_As: learning rate for atoms As
        
        """
        grad_Cs = np.zeros_like(self.Cs)
        grad_As = np.zeros_like(self.As)
        sum_ws_Cs = gwu.np_sum_scaled_mat(self.Cs, wt)
        sum_ws_As = gwu.np_sum_scaled_mat(self.As,wt)
        shared_term_Cs = self.alpha* (sum_ws_Cs*self.hhs - (Ct.dot(Tt)).T.dot(Tt)) 
        shared_term_As = (1-self.alpha)*(self.diagh.dot(sum_ws_As) - Tt.T.dot(At))
        for pos in range(self.number_Cs):
            grad_Cs[pos] = 2*wt[pos]*shared_term_Cs
            grad_As[pos] = 2*wt[pos]*shared_term_As
    
            self.Cs[pos] -= lr_Cs*grad_Cs[pos]
            self.Cs[pos] = np.maximum(np.zeros((self.shape_Cs,self.shape_Cs)), (self.Cs[pos]+ self.Cs[pos].T )/2)
            self.As[pos] -=lr_As*grad_As[pos]
                
    def GDL_unmixing(self,l2_reg:float,eps:float,max_iter_outer:int,max_iter_inner:int,verbose:bool=False):
        """
        For each labeled graph - solve independently the FGW unmixing problem with BCD algorithm
        
        Parameters
        ----------
        l2_reg : regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        max_iter_outer : maximum number of outer loop iterations of our BCD algorithm
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        verbose : Check the good evolution of the loss. The default is False.
        """
        
        w = np.ones((self.dataset_size,self.number_Cs))/self.number_Cs
        best_w = w.copy()
        best_losses= np.zeros(self.dataset_size)
        for t in tqdm(range(self.dataset_size)):
            saved_transport=None
            
            prev_loss_w = 10**(8)
            current_loss_w = 10**(7)
            convergence_criterion = np.inf
            best_loss_w = np.inf
            outer_count=0
            while (convergence_criterion >eps) and (outer_count<max_iter_outer):
                outer_count+=1
                prev_loss_w = current_loss_w
                w[t],current_loss_w,saved_transport,inner_count = self.BCD_step(self.C_target[t], self.A_target[t], w[t], saved_transport, l2_reg,max_iter=max_iter_inner,eps=eps)
                if prev_loss_w!=0:
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w)
                else:
                    convergence_criterion = abs(prev_loss_w - current_loss_w)/abs(prev_loss_w+10**(-12))
                if current_loss_w < best_loss_w:
                    best_w[t]= w[t]
                    best_loss_w = current_loss_w
            best_losses[t]=best_loss_w
        return best_w, best_losses
    
    
    def save_elements(self,save_settings=False,local_step=0,checkpoint_a=None,checkpoint_losses=None):
        """
        Save dictionary states and tracked reconstruction for visualization
        """
        path = os.path.abspath('../')+self.experiment_repo
        #print('path',path)
        if not os.path.exists(path+self.experiment_name):
            os.makedirs(path+self.experiment_name)
            print('made dir', path+self.experiment_name)
        if save_settings:
            pd.DataFrame(self.settings, index=self.settings.keys()).to_csv(path+'%s/settings'%self.experiment_name)
        
        if not self.save_chunks:
            self.tracked_a.append(checkpoint_a)
            self.tracked_losses.append(checkpoint_losses)
            self.tracked_Cs.append(self.Cs)
            self.tracked_As.append(self.As)
            # We save the full set of tracked elements
            np.save(path+'%s/tracked_Cs.npy'%self.experiment_name, np.array(self.tracked_Cs))
            np.save(path+'%s/tracked_As.npy'%self.experiment_name, np.array(self.tracked_As))
            np.save(path+'%s/tracked_a.npy'%self.experiment_name, np.array(self.tracked_a))
            np.save(path+'%s/tracked_losses.npy'%self.experiment_name, np.array(self.tracked_losses))
        else:
            np.save(path+'%s/Cs_checkpoint%s.npy'%(self.experiment_name,local_step), self.Cs)
            np.save(path+'%s/As_checkpoint%s.npy'%(self.experiment_name,local_step), self.As)
            np.save(path+'%s/a_checkpoint%s.npy'%(self.experiment_name,local_step), checkpoint_a)
            np.save(path+'%s/losses_checkpoint%s.npy'%(self.experiment_name,local_step), checkpoint_losses)
            
        for key in self.log.keys():
            try:
                np.save(path+'%s/%s.npy'%(self.experiment_name,key), np.array(self.log[key]))
            except:
                print('bug for  log component: %s'%key)
                pass
        
    def load_elements(self, path=None, pos=-1):
        """
        load elements
        """
        if path is None:
            path = os.path.abspath('../')+self.experiment_repo
        else:
            path+='/%s/'%self.experiment_repo
        
        self.tracked_Cs = np.load(path+'%s/tracked_Cs.npy'%self.experiment_name)
        self.tracked_As = np.load(path+'%s/tracked_As.npy'%self.experiment_name)
        self.Cs  = self.tracked_Cs[pos]
        self.As = self.tracked_As[pos]
        self.number_Cs= self.Cs.shape[0]
        self.shape_Cs = self.Cs.shape[-1]
        self.d = self.As.shape[-1]
        
        hs= np.ones(self.shape_Cs)/self.shape_Cs
        self.hhs = hs[:,None].dot(hs[None,:])
        self.diagh= np.diag(hs)
        self.reshaped_hs = hs[:,None].dot(np.ones((1,self.d)))
        