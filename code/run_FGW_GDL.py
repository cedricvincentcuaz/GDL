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

from GDL_FGWalgo import FGW_Unsupervised_DictionaryLearning
import GDL_utils as gwu
import result_manager
import pandas as pd
import argparse
#%% 
# =============================================================================
# Experiments on real datasets without node features:
# - 1/ Learn the dictionary atoms {C_s}
# - 2/ Solve GW unmixing problems by projecting input graphs Ck on 
#       the linear subspace defined by atoms {C_s}. Providing embeddings wk
# - 3/ Evaluate embeddings wk:
#       a) Clustering with kmeans using the Mahalanobis distance from Proposition 1
#       b) Classification with SVM endowed with Mahalanobis distance
# - 4/ Evaluate embedded graphs sum_s wk_s C_s :
#       Classification with SVM endowed with GW distance
# =============================================================================

#python run_FGW_GDL.py -ds "imdb-m" -natoms [4,8] -satoms [10] -reg [0.01] -alpha [0.25,0.5] -ep 2 -b 16 -lrC 0.01 -lrA 0.01 -mode "ADJ"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='learning vanilla GDL for unlabeled graphs')
    parser.add_argument('-ds','--dataset_name', type=str,help='the name of the dataset',choices=['mutag','ptc','bzr','cox2','enzymes','protein'],required=True)
    parser.add_argument('-natoms','--list_number_atoms', type=str,help='validated list of number of atoms',required=True)
    parser.add_argument('-satoms','--list_shape_atoms', type=str,help='validated list of atoms shapes',required=True)
    parser.add_argument('-reg','--list_l2reg', type=str,default=str([0.0]),help='validated list of negative quadratic regularization parameters',required=True)
    parser.add_argument('-alpha','--list_alpha', type=str,default=str([0.5]),help='validated list of alpha parameter for FGW distance',required=True)
    parser.add_argument('-ep','--epochs', type=int,default=50,help='epochs to run for stochastic algorithm',required=True)
    parser.add_argument('-b','--batch_size', type=int,default=16,help='batch size for stochastic updates',required=True)
    parser.add_argument('-lrC','--learning_rate_Cs', type=float,default=0.01,help='learning rate for SGD updates of atoms Cs - Adam is included with default values',required=True)
    parser.add_argument('-lrA','--learning_rate_As', type=float,default=0.01,help='learning rate for SGD updates of atoms As - Adam is included with default values',required=True)
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
    list_alpha = [float(x) for x in args.list_alpha[1:-1].split(',')]
    # used for experiments in general is:
    # [0.000001,0.0003,0.005,0.1,0.25,0.5,0.75,0.9,0.995,0.9997,0.999999]
    
    for number_Cs in list_number_atoms:
        for shape_Cs in list_shape_atoms:
            for alpha in list_alpha:
                for seed in list_seed:
                    for l2_reg in list_l2reg:
                                        
                        if args.dataset_name in ['mutag','ptc']:#one-hot features
                            experiment_name= "/%s/"%args.mode+args.dataset_name+"_FGWGDL_ONEHOT_sampledCA_S%sN%s_a%s_l2%s_batch%s_lrC%s_lrA%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(shape_Cs,number_Cs,alpha,l2_reg,args.batch_size,args.learning_rate_Cs,args.learning_rate_As,args.max_iter_outer,args.max_iter_inner,args.eps,args.epochs,seed)                        
                        else:
                            experiment_name= "/%s/"%args.mode+args.dataset_name+"_FGWGDL_sampledCA_S%sN%s_a%s_l2%s_batch%s_lrC%s_lrA%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(shape_Cs,number_Cs,alpha,l2_reg,args.batch_size,args.learning_rate_Cs,args.learning_rate_As,args.max_iter_outer,args.max_iter_inner,args.eps,args.epochs,seed)
                        
                        local_path = os.path.abspath('../')
                        full_path = local_path +experiment_repo+experiment_name
                        print(full_path)
                        DL = FGW_Unsupervised_DictionaryLearning(args.dataset_name, args.mode, number_Cs, shape_Cs, alpha,experiment_repo,experiment_name,data_path='../data/')
                        print('Starting to learn atoms')
                        if not os.path.exists(full_path):
                            print('Starting to learn atoms')
                            #### 1/ Learn the dictionary atoms {C_s,A_s}
                            _=DL.Stochastic_GDL(l2_reg,eps= args.eps, max_iter_outer = args.max_iter_outer,max_iter_inner=args.max_iter_inner,
                                                lr_Cs=args.learning_rate_Cs,lr_As=args.learning_rate_As,batch_size=args.batch_size,
                                                beta_1=beta_1, beta_2=beta_2, epochs=args.epochs, algo_seed=seed,init_mode_atoms=args.init_mode_atoms,
                                                checkpoint_timestamp=checkpoint_timestamp, visualization_frame=visualization_frame,verbose=False)
                            print('done learning')
                        else:
                            DL.load_elements(use_checkpoint = (not checkpoint_timestamp is None))
                        #### 2/ Solve GW unmixing problems wk for all graph Gk
                        print('COMPUTE UNMIXINGS')
                        unmixing_name = 'unmixing_l2reg%s_maxiterout%s_maxiterin%s_eps%s'%(l2_reg, args.max_iter_outer,args.max_iter_inner, args.eps)
                        unmixing_path = local_path+experiment_repo+experiment_name+'/%s.npy'%unmixing_name
                        reconstruction_path = local_path+experiment_repo+experiment_name+'/reconstruction_%s.npy'%unmixing_name
                        unmixing, reconstruction_errors=DL.GDL_unmixing(l2_reg=l2_reg,eps=args.eps,max_iter_outer=args.max_iter_outer,max_iter_inner=args.max_iter_inner)
                        np.save(unmixing_path, unmixing)
                        np.save(reconstruction_path,reconstruction_errors)
                        # 3/ Evaluate embeddings {wk}
                        exp = result_manager.load_DL_experiments(experiment_name, experiment_repo)
                        y_true=DL.y
                        exp['Ct']=DL.C_target
                        exp['At']=DL.A_target
                        
                        exp['labels']= y_true
                        Cs_str = 'Cs'
                        As_str ='As'
                        if not (checkpoint_timestamp is None):
                            # we ran checkpoint hence the best unmixings are stored in 'checkpoint_unmixings.npy'
                            used_checkpoint=True
                            w_str = 'checkpoint_unmixings'
                            Cs_str  ='checkpoint_Cs'
                            As_str = 'checkpoint_As'
                        X= exp[unmixing_name]
                        M = gwu.compute_mahalanobisFGW_matrix(exp[Cs_str],exp[As_str],alpha)
                        def dist_func(x,y):
                            z = x-y
                            return np.sqrt((z.T).dot(M).dot(z))
            
                        if args.dataset_name in ['mutag','ptc','bzr','cox2','protein']:
                            n_clusters=2
                        elif args.dataset_name in ['enzymes']:
                            n_clusters=6
                        else:
                            raise 'UNKNOWN DATASET NAME'
                        #### 3/ a)Clustering with kmeans using the Mahalanobis distance from Proposition 1
                        print('computing Kmeans -Mahalanobis')
                        
                        #compute M^{1/2}. 
                        rootM = sqrtm(M)
                        #Transform features so they follow an euclidean geometry
                    
                        new_X = np.zeros_like(X)
                        for i in range(new_X.shape[0]):
                            new_X[i] = rootM.dot(X[i])
                        local_res = {'nC':[] }
                        key_to_func = {"RI": rand_score}
                        for metric in key_to_func.keys():
                            local_res[metric]=[]
                        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, random_state=0, tol=10**(-9), max_iter = 10**5)
                        y_pred=kmeans.fit_predict(new_X)
                        local_res['nC'].append(n_clusters)
                        for key in key_to_func.keys():
                            local_res[key].append(key_to_func[key](y_true,y_pred))
                        if used_checkpoint:
                            pd.DataFrame(local_res).to_csv(full_path+'/res_FGWmahalanobisKmeans_checkpoint.csv')
                        else:
                            pd.DataFrame(local_res).to_csv(full_path+'/res_FGWmahalanobisKmeans.csv')
                        
                    
                        if args.dataset_name in ['ptc','mutag']:
                            n_folds= 10
                            n_iter= 50
                        else:
                            n_folds= 10
                            n_iter= 10
                        #### 3/ b) SVM classification with kernels derived from the Mahalanobis distance over {wk}
                        if run_supervised_embedding_evaluation:
                            
                            kernel='MahalanobisFGW'
                            result_manager.nested_classifications_GWDL_SVC_precomputed_finale(DL.y, DL.experiment_repo, experiment_name,unmixing_name=unmixing_name,atoms_name=Cs_str,kernel=kernel,n_folds=n_folds,n_iter=n_iter)
                            
                            #### 4/ SVM classification with GW kernels on embedded graphs
                            kernel='FGW'
                            result_manager.nested_classifications_GWDL_SVC_precomputed_finale(DL.y, DL.experiment_repo, experiment_name,unmixing_name=unmixing_name,atoms_name=Cs_str,kernel=kernel,n_folds=n_folds,n_iter=n_iter)
