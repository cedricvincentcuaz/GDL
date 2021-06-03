"""
@author: cvincentcuaz
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as sklearn_split
import GDL_utils as gwu
#%%

        
def load_DL_experiments(save_folder:str, 
                        experiment_repo:str='/toy_experiments/'):
    """
    load all .npy files present in experiment_repo+save_folder
    and store them in a dictionary with filenames as keys.
    """
    path = os.path.abspath('../')+experiment_repo
    #print('full path',path)
    elements=[]
    for file in os.listdir(path+save_folder):
        #print(file)
        if file.endswith(".npy"):
            local_path =os.path.join(path+save_folder, file)
            #print('local path', local_path)
            last_piece = local_path.split('/')[-1]
            #print('last piece',last_piece)
            #elements.append(last_piece.split('.')[0])
            pos = last_piece[::-1].find('.')
            elements.append(last_piece[:-pos-1])
    loaded_res={}
    for elem in elements:
        try:
            loaded_res[elem] = np.load(path+save_folder+elem+'.npy')
        except:
            continue #some experiments have pickled data - to handle later
    return loaded_res


def read_experiment_settings(experiment_repo:str,
                             name:str, 
                             parameters:list):
    """
    Load settings file present for a given experiment
    - parameters can be used to select some parameters
    """
    df = pd.read_csv('../%s%s/settings'%(experiment_repo,name), sep=',', engine='python')
    settings={}
    for param in parameters:
        try:
            settings[param] = df[param][0]
        except:
            print('failed loading param : %s'%param)
            continue
    return settings


def Kernel_Matrix_precomputed(D:np.array,
                              gamma:float):
    
    return np.exp(-gamma*D)

def FGW_matrix(exp,settings,a,atoms_name):
    """
        Compute pairwise FGW matrix between embeddings stored in exp
        which designes the kind of dictionaries provided by the function load_DL_experiments
    """
    n= exp[a].shape[0]
    alpha= settings['alpha']
    if atoms_name =='Cs':
        learnt_mat = [(gwu.np_sum_scaled_mat(exp['Cs'], exp[a][t]),gwu.np_sum_scaled_mat(exp['As'], exp[a][t])) for t in range(exp[a].shape[0]) ]
    elif atoms_name =='checkpoint_Cs':
        learnt_mat = [(gwu.np_sum_scaled_mat(exp['checkpoint_Cs'], exp[a][t]),gwu.np_sum_scaled_mat(exp['checkpoint_As'], exp[a][t])) for t in range(exp[a].shape[0]) ]
    
    D = np.zeros((n,n), dtype=np.float64)
   
    for i in tqdm(range(n-1)):
        for j in range (i+1, n):
            
            dist,T= gwu.numpy_FGW_loss(learnt_mat[i][0], learnt_mat[j][0], learnt_mat[i][1], learnt_mat[j][1], alpha=alpha)
            D[i,j]= dist
            D[j,i]= dist
    return D



def GW_matrix(exp,a,atoms_name):
    """
        Compute pairwise GW matrix between embeddings stored in exp
        which designes the kind of dictionaries provided by the function load_DL_experiments
    """
    n= exp[a].shape[0]
    if atoms_name =='Cs':
        learnt_mat = [gwu.np_sum_scaled_mat(exp['Cs'], exp[a][t]) for t in range(exp[a].shape[0]) ]
    else:
        learnt_mat = [gwu.np_sum_scaled_mat(exp['checkpoint_Cs'], exp[a][t]) for t in range(exp[a].shape[0]) ]

    D = np.zeros((n,n), dtype=np.float64)

    for i in tqdm(range(n-1)):
        for j in range (i+1, n):
            
            dist,T= gwu.np_GW2(learnt_mat[i],learnt_mat[j])
            D[i,j]= dist
            D[j,i]= dist
    return D

def extendedGW_matrix(exp,ac,ah,atoms_name):
    """
        Compute pairwise GW matrix between embeddings stored in exp
        for the extended model proposed in equation 6,
        Where atoms are structures and masses {C_s, h_s}.
    """
    T= exp[ac].shape[0]
    embeddings = []
    if atoms_name =='Cs':
        for t in range(T):
            sum_as_Cs = gwu.np_sum_scaled_mat(exp['Cs'], exp[ac][t,:] )
            sum_as_hs = gwu.np_sum_scaled_mat(exp['hs'], exp[ah][t,:] )
            embeddings.append((sum_as_Cs,sum_as_hs))
    else:
        for t in range(T):
            sum_as_Cs = gwu.np_sum_scaled_mat(exp['checkpoint_Cs'], exp[ac][t,:] )
            sum_as_hs = gwu.np_sum_scaled_mat(exp['checkpoint_hs'], exp[ah][t,:] )
            embeddings.append((sum_as_Cs,sum_as_hs))

    D = np.zeros((T,T), dtype=np.float64)
    
    for i in tqdm(range(T-1)):
        for j in range (i+1, T):
            
            dist,_= gwu.np_GW2(embeddings[i][0],embeddings[j][0],p=embeddings[i][1],q=embeddings[j][1])
            D[i,j]= dist
            D[j,i]= dist
    return D

def nested_classifications_GWDL_SVC_precomputed_finale(y,
                                                       experiment_repo:str, 
                                                       experiment_name:str,
                                                       unmixing_name:str,
                                                       atoms_name:str,
                                                       kernel:str='Mahalanobis',
                                                       n_folds:int=10,
                                                       n_iter:int=10, 
                                                       verbose:bool=False):
    """
    Evaluation of the embeddings stored in an experiment reposity from GDL

    Parameters
    ----------
    y : true_labels (list or np.array)
    unmixing_name : unmixing to use for classification - stored in experiment subfolder.
                    Can designed either the unmixing computed on last update or the one saved for the best dictionary state while using checkpoints
    kernel : type of kernel matrix to use.
    n_folds : 
    n_iter : parameters for running 10 folds CV over 10 iterations. Default as specified in supplementary
    """
    
    if kernel in ['FGW','GW']:
        nested_classifications_GWDL_SVCgraphs_precomputed_finale(y, experiment_repo, experiment_name,unmixing_name,atoms_name,kernel,n_folds, n_iter, verbose)
    elif kernel in ['Mahalanobis']:
        nested_classifications_GWDL_SVCweights_precomputed_mahalanobis_finale(y, experiment_repo, experiment_name,unmixing_name,atoms_name,n_folds, n_iter,verbose)
    elif kernel in ['MahalanobisFGW']:
        nested_classifications_GWDL_SVCweights_precomputed_mahalanobisFGW_finale(y, experiment_repo, experiment_name,unmixing_name,atoms_name,n_folds, n_iter,verbose)
    else:
        raise 'UNKNOWN KERNEL'
        
def nested_classifications_GWDL_SVCgraphs_precomputed_finale(y, experiment_repo, experiment_name,unmixing_name,atoms_name='Cs', kernel='GW',n_folds=10, n_iter=10,verbose=False):
    """
        Compute GW or FGW on embedded (labeled) graphs
        set kernel to:
            -   'GW' to run GW kernel on embedded graphs without attributes
            -   'FGW' to run FGW kernel on embedded labeled graphs. 
                The trade-off parameters alpha will be the one used for the dictionary learning
    """
    exp= load_DL_experiments(experiment_name, experiment_repo)
    path=  os.path.abspath('../')
    print('SVC on graphs with %s distance - experiment_name = %s'%(kernel,experiment_name))
    settings = read_experiment_settings(experiment_repo,experiment_name, ['l2_reg','alpha'])
    print('settings:',settings)
    
    key=unmixing_name
    unmixing_path = path+'%s%sres_svc%s_nestedcv_%s.csv'%(experiment_repo,experiment_name,kernel,key)
    assert key in exp.keys()
    if os.path.exists(unmixing_path):
        print('existing results')
    else:    
        
        res_best_svc={'C':[], 'gamma':[],'val_mean_acc':[],'test_acc':[]}
        print('start computing pairwise matrix')
        if kernel=='GW':
            D=GW_matrix(exp,key,atoms_name)
            D[D<=10**(-15)]=0
        elif kernel=='FGW':
            D=FGW_matrix(exp,settings,key,atoms_name)
            D[D<=10**(-15)]=0
        size= D.shape[0]
        for i in tqdm(range(n_iter)): # do the nested CV
            if verbose:
                print('n_iter:',i)
            k_fold=StratifiedKFold(n_splits=n_folds,random_state=i,shuffle=True)
            idx_train,idx_test,y_train,y_test=sklearn_split(np.arange(size),y, test_size=0.1, stratify=y, random_state=i)
       
            res_SVC = {}
            for C in [10**x for x in range(-7,8)]:
                for gamma in [2**k for k in np.linspace(-10,10)]:
                    local_mean_train = []
                    local_mean_val = []
                    for k,(idx_subtrain, idx_valid) in enumerate(k_fold.split(idx_train,y_train)):
                        if verbose:
                            print('fold:',k)
                        true_idx_subtrain=[idx_train[i] for i in idx_subtrain]
                        true_idx_valid=[idx_train[i] for i in idx_valid]
            
                        y_subtrain = np.array([y[i] for i in true_idx_subtrain])
                        y_val=np.array([y[i] for i in true_idx_valid])
                        
                        if kernel in ['GW','FGW']:
                            clf= SVC(C=C, kernel="precomputed",max_iter=5*10**6,random_state=0)
                        else:
                            raise 'Kernel not handled'
                        G_subtrain = Kernel_Matrix_precomputed(D[true_idx_subtrain,:][:,true_idx_subtrain],gamma=gamma)
                        if verbose:
                            print('check G_subtrain: sum/ nan / inf', G_subtrain.sum(), np.isnan(G_subtrain).sum(), (G_subtrain ==np.inf).sum())
                        clf.fit(G_subtrain,y_subtrain)
                        #print('n_iter_:', clf.n_iter_)
                        train_score= clf.score(G_subtrain,y_subtrain)
                        G_val = Kernel_Matrix_precomputed(D[true_idx_valid, :][:,true_idx_subtrain],gamma=gamma)
                        if verbose: 
                            print('check G_val: sum/ nan / inf', G_val.sum(), np.isnan(G_val).sum(), (G_val ==np.inf).sum())
                        val_score = clf.score(G_val,y_val)
                        local_mean_train.append(train_score)
                        local_mean_val.append(val_score)
                        if verbose:
                            print('SVC/ kernel:%s/ C:%s /gamma:%s /train: %s / val :%s'%(kernel,C,gamma,train_score,val_score))
                    if verbose:
                        print('C:%s / gamma:%s / train: %s / val : %s'%(C,gamma,np.mean(local_mean_train), np.mean(local_mean_val)))
                    res_SVC[(C,gamma)]=np.mean(local_mean_val)
        
            best_idx = np.argmax(list(res_SVC.values()))
            best_key = list(res_SVC.keys())[best_idx]
            res_best_svc['C'].append(best_key[0])
            res_best_svc['gamma'].append(best_key[1])
            res_best_svc['val_mean_acc'].append(res_SVC[best_key])
            
            clf= SVC(C=best_key[0], kernel="precomputed",random_state=0)
            G_train =Kernel_Matrix_precomputed(D[idx_train,:][:,idx_train],gamma=best_key[1])
            if verbose: 
                print('check G_full_train: sum/ nan / inf', G_train.sum(), np.isnan(G_train).sum(), (G_train ==np.inf).sum())
                            
            clf.fit(G_train, y_train)
            G_test = Kernel_Matrix_precomputed(D[idx_test,:][:,idx_train],gamma=best_key[1])
            if verbose: 
                print('check G_test: sum/ nan / inf', G_test.sum(), np.isnan(G_test).sum(), (G_test ==np.inf).sum())
            res_best_svc['test_acc'].append(clf.score(G_test,y_test))
        
        pd.DataFrame(res_best_svc).to_csv(unmixing_path,index=False)
        print('done compute SVM with %s distances'%kernel)
    



def Mahalanobis_Distance_Matrix(X1,X2,M):
    n1= X1.shape[0]
    n2= X2.shape[0]
    D=np.zeros((n1,n2))
    if n1==n2:
        #assuming X1 and X2 are identical regarding our experiments
        for i in tqdm(range(n1)):
            for j in range(i+1,n2):
                z=X1[i]-X2[j]
                D[i,j]=D[j,i] = np.sqrt((z.T).dot(M).dot(z))
    else:
        for i,x1 in enumerate(X1):
            for j,x2 in enumerate(X2):
                z=x1-x2
                D[i,j]= np.sqrt((z.T).dot(M).dot(z))
    return D

def nested_classifications_GWDL_SVCweights_precomputed_mahalanobisFGW_finale(y, experiment_repo, experiment_name,unmixing_name,atoms_name,n_folds=10, n_iter=10,verbose=False):
    exp= load_DL_experiments(experiment_name, experiment_repo)
    path=  os.path.abspath('../')
    settings = read_experiment_settings(experiment_repo,experiment_name, ['alpha'])
    assert unmixing_name in exp.keys()
    key = unmixing_name
    res_path=path+'%s%sres_svcMahalanobisFGW_nestedcv_%s.csv'%(experiment_repo,experiment_name,key)
    alpha = settings['alpha']

    if os.path.exists(res_path):
        print('existing results')
    else:

        res_best_svc={'C':[], 'gamma':[],'val_mean_acc':[],'test_acc':[]}
            
        X = exp[key]
        assert X.shape[0]==y.shape[0]
        if (os.path.exists(path+'%s%sMahalanobisFGW%s_distances_%s.csv'%(experiment_repo,experiment_name,alpha,key))):
            D= np.load(path+'%s%sMahalanobisFGW%s_distances_%s.csv'%(experiment_repo,experiment_name,alpha,key))
            D[D<=10**(-15)]=0
        else:
            if atoms_name=='checkpoint_Cs' :
                M = gwu.compute_mahalanobisFGW_matrix(exp['checkpoint_Cs'],exp['checkpoint_As'],alpha)
            else:
                
                M = gwu.compute_mahalanobisFGW_matrix(exp['Cs'],exp['As'],alpha)
            D= Mahalanobis_Distance_Matrix(X, X, M)
            D[D<=10**(-15)]=0
            np.save(path+'%s%sMahalanobisFGW%s_distances_%s.csv'%(experiment_repo,experiment_name,alpha,key), D)
        size = D.shape[0]
        for i in tqdm(range(n_iter)): # do the nested CV
            if verbose:
                print('n_iter:',i)
            k_fold=StratifiedKFold(n_splits=n_folds,random_state=i,shuffle=True)
            idx_train,idx_test,y_train,y_test=sklearn_split(np.arange(size),y, test_size=0.1, stratify=y, random_state=i)
            if verbose:
                print('split computed')
        
            res_SVC = {}
            for C in [10**x for x in range(-7,8)]:
                for gamma in list([2**k for k in np.linspace(-10,10)]):
                    local_mean_train = []
                    local_mean_val = []
                    for k,(idx_subtrain, idx_valid) in enumerate(k_fold.split(idx_train,y_train)):
                        if verbose:
                            print('fold:',k)
                        true_idx_subtrain=[idx_train[i] for i in idx_subtrain]
                        true_idx_valid=[idx_train[i] for i in idx_valid]
            
                        y_subtrain = np.array([y[i] for i in true_idx_subtrain])
                        y_val=np.array([y[i] for i in true_idx_valid])
                        
                        clf= SVC(C=C, kernel="precomputed",max_iter=5*10**6,random_state=0)
                        G_subtrain = Kernel_Matrix_precomputed(D[true_idx_subtrain,:][:,true_idx_subtrain],gamma=gamma)
                        if verbose:
                            print('check G_subtrain: sum/ nan / inf', G_subtrain.sum(), np.isnan(G_subtrain).sum(), (G_subtrain ==np.inf).sum())
                        clf.fit(G_subtrain,y_subtrain)
                        #print('n_iter_:', clf.n_iter_)
                        train_score= clf.score(G_subtrain,y_subtrain)
                        G_val = Kernel_Matrix_precomputed(D[true_idx_valid, :][:,true_idx_subtrain],gamma=gamma)
                        if verbose: 
                            print('check G_val: sum/ nan / inf', G_val.sum(), np.isnan(G_val).sum(), (G_val ==np.inf).sum())
                        val_score = clf.score(G_val,y_val)
                        local_mean_train.append(train_score)
                        local_mean_val.append(val_score)
                        if verbose:
                            print('SVC/ kernel: Mahalanobis/ C:%s /gamma:%s /train: %s / val :%s'%(C,gamma,train_score,val_score))
                    if verbose:
                        print('C:%s / gamma:%s / train: %s / val : %s'%(C,gamma,np.mean(local_mean_train), np.mean(local_mean_val)))
                    res_SVC[(C,gamma)]=np.mean(local_mean_val)
        
            best_idx = np.argmax(list(res_SVC.values()))
            best_key = list(res_SVC.keys())[best_idx]
            res_best_svc['C'].append(best_key[0])
            res_best_svc['gamma'].append(best_key[1])
            res_best_svc['val_mean_acc'].append(res_SVC[best_key])
            
            clf= SVC(C=best_key[0], kernel="precomputed",random_state=0)
            G_train =Kernel_Matrix_precomputed(D[idx_train,:][:,idx_train],gamma=best_key[1])
            if verbose: 
                print('check G_full_train: sum/ nan / inf', G_train.sum(), np.isnan(G_train).sum(), (G_train ==np.inf).sum())
                            
            clf.fit(G_train, y_train)
            G_test = Kernel_Matrix_precomputed(D[idx_test,:][:,idx_train],gamma=best_key[1])
            if verbose: 
                print('check G_test: sum/ nan / inf', G_test.sum(), np.isnan(G_test).sum(), (G_test ==np.inf).sum())
            res_best_svc['test_acc'].append(clf.score(G_test,y_test))
        
        pd.DataFrame(res_best_svc).to_csv(res_path,index=False)
        
def nested_classifications_GWDL_SVCweights_precomputed_mahalanobis_finale(y, experiment_repo, experiment_name,unmixing_name,atoms_name, n_folds=10, n_iter=10, verbose=False):
    
    exp= load_DL_experiments(experiment_name, experiment_repo)
    path=  os.path.abspath('../')
    print('SVC on embeddings wk - experiment name= ', experiment_name) 
    if atoms_name=='checkpoint_Cs' :
        M = gwu.compute_mahalanobis_matrix(exp['checkpoint_Cs'])
    else:
        M = gwu.compute_mahalanobis_matrix(exp['Cs'])
    
    print('M:',M)
    key=unmixing_name
    assert key in exp.keys()
    if (os.path.exists(path+'%s%sres_svcMahalanobis_nestedcv_%s.csv'%(experiment_repo,experiment_name,key))):
        print('existing results')
        
    else:
        
        res_best_svc={'C':[], 'gamma':[],'val_mean_acc':[],'test_acc':[]}
        X = exp[key]
        assert X.shape[0]==y.shape[0]
        if (os.path.exists(path+'%s%sMahalanobis_distances_%s.csv'%(experiment_repo,experiment_name,key))):
            D= np.load(path+'%s%sMahalanobis_distances_%s.csv'%(experiment_repo,experiment_name,key))
            D[D<=10**(-15)]=0

        else:
            print('computing pairwise distances matrix')
            D= Mahalanobis_Distance_Matrix(X, X, M)
            D[D<=10**(-15)]=0
            np.save(path+'%s%sMahalanobis_distances_%s.csv'%(experiment_repo,experiment_name,key), D)
        size = D.shape[0]
        for i in tqdm(range(n_iter)): # Nested cross validation repeated n_iter times.
            if verbose:
                print('n_iter:',i)
            k_fold=StratifiedKFold(n_splits=n_folds,random_state=i,shuffle=True)
            idx_train,idx_test,y_train,y_test=sklearn_split(np.arange(size),y, test_size=0.1, stratify=y, random_state=i)
            res_SVC = {}
            #running nested cross validation for a seed
            for C in [10**x for x in range(-7,8)]:
                for gamma in [2**k for k in np.linspace(-10,10)]:
                    local_mean_train = []
                    local_mean_val = []
                    for k,(idx_subtrain, idx_valid) in enumerate(k_fold.split(idx_train,y_train)):
                        if verbose:
                            print('fold:',k)
                        true_idx_subtrain=[idx_train[i] for i in idx_subtrain]
                        true_idx_valid=[idx_train[i] for i in idx_valid]
            
                        y_subtrain = np.array([y[i] for i in true_idx_subtrain])
                        y_val=np.array([y[i] for i in true_idx_valid])
                        
                        clf= SVC(C=C, kernel="precomputed",max_iter=5*10**6,random_state=0)
                        G_subtrain = Kernel_Matrix_precomputed(D[true_idx_subtrain,:][:,true_idx_subtrain],gamma=gamma)
                        if verbose:
                            print('check G_subtrain: sum/ nan / inf', G_subtrain.sum(), np.isnan(G_subtrain).sum(), (G_subtrain ==np.inf).sum())
                        clf.fit(G_subtrain,y_subtrain)
                        #print('n_iter_:', clf.n_iter_)
                        train_score= clf.score(G_subtrain,y_subtrain)
                        G_val = Kernel_Matrix_precomputed(D[true_idx_valid, :][:,true_idx_subtrain],gamma=gamma)
                        if verbose: 
                            print('check G_val: sum/ nan / inf', G_val.sum(), np.isnan(G_val).sum(), (G_val ==np.inf).sum())
                        val_score = clf.score(G_val,y_val)
                        local_mean_train.append(train_score)
                        local_mean_val.append(val_score)
                        if verbose:
                            print('SVC/ kernel: Mahalanobis/ C:%s /gamma:%s /train: %s / val :%s'%(C,gamma,train_score,val_score))
                    if verbose:
                        print('C:%s / gamma:%s / train: %s / val : %s'%(C,gamma,np.mean(local_mean_train), np.mean(local_mean_val)))
                    res_SVC[(C,gamma)]=np.mean(local_mean_val)
        
            best_idx = np.argmax(list(res_SVC.values()))
            best_key = list(res_SVC.keys())[best_idx]
            res_best_svc['C'].append(best_key[0])
            res_best_svc['gamma'].append(best_key[1])
            res_best_svc['val_mean_acc'].append(res_SVC[best_key])
            
            clf= SVC(C=best_key[0], kernel="precomputed",random_state=0)
            G_train =Kernel_Matrix_precomputed(D[idx_train,:][:,idx_train],gamma=best_key[1])
            if verbose: 
                print('check G_full_train: sum/ nan / inf', G_train.sum(), np.isnan(G_train).sum(), (G_train ==np.inf).sum())
                            
            clf.fit(G_train, y_train)
            G_test = Kernel_Matrix_precomputed(D[idx_test,:][:,idx_train],gamma=best_key[1])
            if verbose: 
                print('check G_test: sum/ nan / inf', G_test.sum(), np.isnan(G_test).sum(), (G_test ==np.inf).sum())
            res_best_svc['test_acc'].append(clf.score(G_test,y_test))
        pd.DataFrame(res_best_svc).to_csv(path+'%s%sres_svcMahalanobis_nestedcv_%s.csv'%(experiment_repo,experiment_name,key),index=False)
        print('Done computing nested cross validation')
