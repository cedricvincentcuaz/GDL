import os
n_jobs=2
os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ['OPENBLAS_NUM_THREADS']= str(n_jobs)
os.environ['MKL_NUM_THREADS'] = str(n_jobs)
os.environ['VECLIB_MAXIMUM_THREADS']=str(n_jobs)
os.environ['NUMEXPR_NUM_THREADS']=str(n_jobs)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from scipy.linalg import sqrtm

import GDL_GWalgo as algos_sto
import GDL_utils as gwu
import result_manager
import pandas as pd
#%% 
# =============================================================================
# Experiments on sbm datasets with 3 clusters
# - 1/ Learn the dictionary atoms {C_s}
# - 2/ Solve GW unmixing problems by projecting input graphs Ck on 
#       the linear subspace defined by atoms {C_s}. Providing embeddings wk
# - 3/ Evaluate embeddings wk:
#       a) Clustering with kmeans using the Mahalanobis distance from Proposition 1
#       
# =============================================================================
# python run_GW_GDL.py "imdb-m" [6,12,18,24] 10 [0,0.1,0.01,0.001] 3000 16 "ADJ" 0.00001 10 100 0 0.1 1
# python run_GW_GDL.py "imdb-m" [4,8,12,16] 10 [0,0.1,0.01,0.001] 2000 16 "ADJ" 0.00001 10 100 0 0.1 1


dataset_name='balanced_clustertoy'
list_number_Cs=[3]

list_shape_Cs=[6]

list_l2reg= [0.,0.001]
epochs= 20
batch_size =16
mode = 'ADJ'
eps = 10**(-5)
max_iter_outer = 10
max_iter_inner = 200
list_seed = [0]
lr = 0.01
lr_str = str(lr).replace('.','')
init_mode_atoms= 0 #random init
beta1= 0.9 #parameters of adam optimizer (grad) 
beta2=0.99 #parameters of adam optimizer (grad**2)
checkpoint_timestamp  = None 
# visualize_frame = 2 would plot loss evolution every 2 epochs
visualization_frame=2

#%%

experiment_repo = "/%s_experiments"%dataset_name

for number_Cs in list_number_Cs:
    for shape_Cs in list_shape_Cs:
        for seed in list_seed:
            for l2_reg in list_l2reg:
                
                experiment_name= "/%s/"%mode+dataset_name+"_GWGDL_randomC_S%sN%s_l2%s_lr%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(shape_Cs,number_Cs,l2_reg,lr_str,max_iter_outer,max_iter_inner,eps,epochs,seed)
                local_path = os.path.abspath('../')
                full_path = local_path +experiment_repo+experiment_name
                print('full path:',full_path)
                DL = algos_sto.GW_Unsupervised_DictionaryLearning(dataset_name,mode, number_Cs, shape_Cs, experiment_repo, experiment_name,data_path='../data/')
                
                #### 1/ Learn the dictionary atoms {C_s}
                print('Starting to learn atoms')
                _=DL.Stochastic_GDL(l2_reg,eps = eps, max_iter_outer = max_iter_outer, max_iter_inner=max_iter_inner,
                                    lr=lr,batch_size=batch_size,beta_1=beta1, beta_2=beta2, epochs=epochs, algo_seed=seed,
                                    init_mode_atoms=init_mode_atoms,checkpoint_timestamp=checkpoint_timestamp, visualization_frame=visualization_frame)
                print('done learning')
                #### 2/ Solve GW unmixing problems wk for all graph Gk
                w_str= 'unmixing_l2reg%s_unif_maxiterout%s_maxiterin%s_eps%s'%(l2_reg, max_iter_outer,max_iter_inner, eps)
                unmixing_path=local_path+experiment_repo+experiment_name+'/%s.npy'%w_str
                reconstruction_path=local_path+experiment_repo+experiment_name+'/reconstruction_%s.npy'%w_str
                
                print('COMPUTING UNMIXING')
        
                unmixing,reconstruction_errors=DL.GDL_unmixing(l2_reg,eps,max_iter_outer,max_iter_inner)
                np.save(unmixing_path, unmixing)
                np.save(reconstruction_path,reconstruction_errors)
                #load all files of experiments                
                exp = result_manager.load_DL_experiments(experiment_name, experiment_repo)
                y_true=DL.labels
                exp['shapes']= DL.shapes
                exp['dataset']=DL.graphs
                exp['labels']= y_true
                Cs_str = 'Cs'
                # 3/ Evaluate embeddings {wk}
                X= exp[w_str]
                M = gwu.compute_mahalanobis_matrix(exp[Cs_str])
                
                n_clusters=3
                #### 3/ a)Clustering with kmeans using the Mahalanobis distance from Proposition 1
                #compute M^{1/2}. 
                rootM = sqrtm(M)
                #Transform features so they follow an euclidean geometry
                new_X = np.zeros_like(X)
                for i in range(new_X.shape[0]):
                    new_X[i] = rootM.dot(X[i])
                local_res = {'nC':[]}
                key_to_func = {"RI": rand_score}
                for metric in key_to_func.keys():
                    local_res[metric]=[]
                #Perform clustering over 10 "good" initializations
                print('computing clustering on embeddings {wk} with Mahalanobis distance')
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, random_state=0, tol=10**(-9), max_iter = 10**5)
                y_pred=kmeans.fit_predict(new_X)
                local_res['nC'].append(n_clusters)
                for key in key_to_func.keys():
                    local_res[key].append(key_to_func[key](y_true,y_pred))
                pd.DataFrame(local_res).to_csv(full_path+'/res_mahalanobisKmeans.csv')
                