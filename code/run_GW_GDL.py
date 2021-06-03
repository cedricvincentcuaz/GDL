"""
@author: cvincentcuaz
"""

import os
#n_jobs=2
#os.environ["OMP_NUM_THREADS"] = str(n_jobs)
#os.environ['OPENBLAS_NUM_THREADS']= str(n_jobs)
#os.environ['MKL_NUM_THREADS'] = str(n_jobs)
#os.environ['VECLIB_MAXIMUM_THREADS']=str(n_jobs)
#os.environ['NUMEXPR_NUM_THREADS']=str(n_jobs)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from scipy.linalg import sqrtm
import GDL_utils as gwu
import GDL_GWalgo as algos_sto
import result_manager
import pandas as pd
#from sys import argv
import argparse
#%% 
# =============================================================================
# Experiments on real datasets without node features:
# - 1/ Learn the dictionary atoms {Cbar_s}
# - 2/ Solve GW unmixing problems by projecting input graphs Ck on 
#       the linear subspace defined by atoms {Cbar_s}. Providing embeddings wk
# - 3/ Evaluate embeddings wk:
#       a) Clustering with kmeans using the Mahalanobis distance from Proposition 1
#       b) Classification with SVM endowed with Mahalanobis distance
# - 4/ Evaluate embedded graphs sum_s wk_s Cbar_s :
#       - Classification with SVM endowed with GW distance
#       - Gromov-Wasserstein kmeans on the embedded graphs 
# =============================================================================

#python run_GW_GDL.py -ds "imdb-m" -natoms [4,8] -satoms [10] -reg [0.01] -ep 2 -b 16 -lr 0.01 -mode "ADJ"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='learning vanilla GDL for unlabeled graphs')
    parser.add_argument('-ds','--dataset_name', type=str,help='the name of the dataset',choices=['imdb-b','imdb-m','balanced_clustertoy','clustertoy2C'],required=True)
    parser.add_argument('-natoms','--list_number_atoms', type=str,help='validated list of number of atoms',required=True)
    parser.add_argument('-satoms','--list_shape_atoms', type=str,help='validated list of atoms shapes',required=True)
    parser.add_argument('-reg','--list_l2reg', type=str,default=str([0.0]),help='validated list of negative quadratic regularization parameters',required=True)
    parser.add_argument('-ep','--epochs', type=int,default=50,help='epochs to run for stochastic algorithm',required=True)
    parser.add_argument('-b','--batch_size', type=int,default=16,help='batch size for stochastic updates',required=True)
    parser.add_argument('-lr','--learning_rate', type=float,default=0.01,help='learning rate for SGD updates - Adam is included with default values',required=True)
    parser.add_argument('-mode','--mode', type=str,default='ADJ',help='graph representation for graphs',choices=['ADJ','SP','LAP'])
    parser.add_argument('-eps','--eps', type=float,default=10**(-5),help='precision to assert unmixings convergence')
    parser.add_argument('-maxout','--max_iter_outer', type=int,default=20,help='maximum number of iterations for BCD algorithm')
    parser.add_argument('-maxin','--max_iter_inner', type=int,default=100,help='maximum number of iterations for FW algorithm for w_k')
    parser.add_argument('-s','--list_seed', type=str,default=str([0]),help='seed to initialize stochastic algorithm and ensure reproductibility')
    parser.add_argument('-init','--init_mode_atoms', type=int,default=1,help='initialization mode for atoms. 0 = random / 1 = sampling')
    
    args = parser.parse_args()
    
    beta_1= 0.9 #parameters of adam optimizer (grad) 
    beta_2=0.99 #parameters of adam optimizer (grad**2)
    # can be used to save best state dictionary by computing 
    # unmixing over the whole dataset at indicate time stamp
    checkpoint_timestamp  = None 
    # visualize_frame = 2 would plot loss evolution every 2 epochs
    visualization_frame=None
    # Used for supervised experiments provided in supplementary
    run_supervised_embedding_evaluation=False
    used_checkpoint=False
    experiment_repo = "/%s_experiments"%args.dataset_name

    list_number_atoms = [int(x) for x in args.list_number_atoms[1:-1].split(',')]
    list_shape_atoms = [int(x) for x in args.list_shape_atoms[1:-1].split(',')]
    list_seed = [int(x) for x in args.list_seed[1:-1].split(',')]
    list_l2reg = [float(x) for x in args.list_l2reg[1:-1].split(',')]
    
    for number_atoms in list_number_atoms:
        for shape_atoms in list_shape_atoms:
            for seed in list_seed:
                for l2_reg in list_l2reg:
                    
                    if args.init_mode_atoms==0:
                        experiment_name= "/%s/"%args.mode+args.dataset_name+"_GWGDL_randomC_atomshape%snum%s_l2reg%s_lr%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(shape_atoms,number_atoms,l2_reg,args.learning_rate,args.max_iter_outer,args.max_iter_inner,args.eps,args.epochs,seed)
                    elif args.init_mode_atoms==1:
                        experiment_name= "/%s/"%args.mode+args.dataset_name+"_GWGDL_sampledC_atomshape%snum%s_l2%s_lr%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(shape_atoms,number_atoms,l2_reg,args.learning_rate,args.max_iter_outer,args.max_iter_inner,args.eps,args.epochs,seed)
                    else:
                        raise 'unknown initialization mode for atoms - has to be in {0,1}'
                    local_path = os.path.abspath('../')
                    full_path = local_path +experiment_repo+experiment_name
                    print('full path:',full_path)
                    DL = algos_sto.GW_Unsupervised_DictionaryLearning(args.dataset_name,args.mode, number_atoms, shape_atoms, experiment_repo, experiment_name,data_path='../data/')
                    if not os.path.exists(full_path):
                        #### 1/ Learn the dictionary atoms {C_s}
                        print('Starting to learn atoms')
                        _=DL.Stochastic_GDL(l2_reg,eps = args.eps, max_iter_outer = args.max_iter_outer, max_iter_inner=args.max_iter_inner,
                                            lr=args.learning_rate,batch_size=args.batch_size,beta_1=beta_1, beta_2=beta_2, 
                                            epochs= args.epochs, algo_seed=seed,init_mode_atoms=args.init_mode_atoms,
                                            checkpoint_timestamp =checkpoint_timestamp, visualization_frame = visualization_frame, verbose=False)
                        print('done learning')
                    else:
    
                        DL.load_elements(use_checkpoint = (not checkpoint_timestamp is None))
                    #### 2/ Solve GW unmixing problems wk for all graph Gk
                    w_str= 'unmixing_l2reg%s_maxiterout%s_maxiterin%s_eps%s'%(l2_reg, args.max_iter_outer,args.max_iter_inner, args.eps)
                    unmixing_path=local_path+experiment_repo+experiment_name+'/%s.npy'%w_str
                    reconstruction_path=local_path+experiment_repo+experiment_name+'/reconstruction_%s.npy'%w_str
                    
                    print('COMPUTING UNMIXING')
            
                    unmixing,reconstruction_errors=DL.GDL_unmixing(l2_reg,args.eps,args.max_iter_outer,args.max_iter_inner)
                    np.save(unmixing_path, unmixing)
                    np.save(reconstruction_path,reconstruction_errors)
                    #load all files of experiments                
                    exp = result_manager.load_DL_experiments(experiment_name, experiment_repo)
                    
                    exp['shapes']= DL.shapes
                    exp['dataset']=DL.graphs
                    exp['labels']= DL.labels
                    # 3/ Evaluate embeddings {wk}
                    Cs_str = 'Cs'
                    if not (checkpoint_timestamp is None):
                        # we ran checkpoint hence the best unmixings are stored in 'checkpoint_unmixings.npy'
                        used_checkpoint=True
                        w_str = 'checkpoint_unmixings'
                        Cs_str  ='checkpoint_Cs'
                    X= exp[w_str]
                    M = gwu.compute_mahalanobis_matrix(exp[Cs_str])
                    if args.dataset_name in ['imdb-b']:
                        n_clusters=2
                    elif args.dataset_name in ['imdb-m','balanced_clustertoy']:
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
                        
                    print('computing clustering on ')
                
                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, random_state=seed, tol=10**(-9), max_iter = 10**5)
                    y_pred=kmeans.fit_predict(new_X)
                    local_res['nC'].append(n_clusters)
                    for key in key_to_func.keys():
                        local_res[key].append(key_to_func[key](exp['labels'],y_pred))
                    if used_checkpoint:
                        #clustering from dictionary state saved at best checkpoint
                        pd.DataFrame(local_res).to_csv(full_path+'/res_mahalanobisKmeans_checkpoint.csv')
                    else:
                        #clustering from finale dictionary state otherwise
                        pd.DataFrame(local_res).to_csv(full_path+'/res_mahalanobisKmeans.csv')
                    #### 3/ b) SVM classification with kernels derived from the Mahalanobis distance over {wk}
                    if run_supervised_embedding_evaluation:
                        n_folds= 10
                        n_iter= 10
                        kernel='Mahalanobis'
                        result_manager.nested_classifications_GWDL_SVC_precomputed_finale(DL.labels, DL.experiment_repo, experiment_name,unmixing_name=w_str,atoms_name =Cs_str,kernel=kernel,n_folds=n_folds,n_iter=n_iter)
                            
                        #### 4/ SVM classification with GW kernels on embedded graphs
                        kernel='GW'
                        result_manager.nested_classifications_GWDL_SVC_precomputed_finale(DL.labels, DL.experiment_repo, experiment_name,unmixing_name=w_str,atoms_name =Cs_str,kernel=kernel,n_folds=n_folds,n_iter=n_iter)
