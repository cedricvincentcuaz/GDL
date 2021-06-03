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
import GDL_GWalgo as algos_sto
import argparse
#%% 
# =============================================================================
# Experiments on real datasets without node features:
# Extended GDL where we simultaneously learning {Cbar_s, hbar_s}
# Cf proposition 2 and equation 6 in the main paper.
#
# - 1/ Learn the dictionary atoms {Cbar_s,hbar_s}
# - 2/ Solve GW unmixing problems by projecting input graphs Ck on 
#       the linear subspace defined by atoms {Cbar_s,hbar_s}. Providing embeddings {wc_k,wh_k}
# - 3/ Evaluate embeddings:
#       GW kmeans on the embeddings cf supplementary.
# =============================================================================

#python run_extendedGWBCDtoy_numpy.py "unbalanced_clustertoy" "default" 3 6 0 1000 4 "ADJ" 0.00001 10 1000 0 0.01 0.0001 0.99 0 1 1 3
#python run_extendedGWBCDtoy_numpy.py "unbalanced_clustertoy" "default" 4 12 0 1000 8 "ADJ" 0.00001 10 1000 0 0.001 0.001 0.99 0 1 1 3
#python run_GW_extendedDL.py -ds "imdb-m" -natoms [4,8] -satoms [10] -regC [0.0] -ep 2 -b 16 -lrC 0.01 -lrh 0.001 -mode "ADJ"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='learning vanilla GDL for unlabeled graphs')
    parser.add_argument('-ds','--dataset_name', type=str,help='the name of the dataset',choices=['imdb-b','imdb-m','balanced_clustertoy','clustertoy2C'],required=True)
    parser.add_argument('-natoms','--list_number_atoms', type=str,help='validated list of number of atoms',required=True)
    parser.add_argument('-satoms','--list_shape_atoms', type=str,help='validated list of atoms shapes',required=True)
    parser.add_argument('-regC','--list_l2reg_Cs', type=str,default=str([0.0]),help='validated list of negative quadratic regularization parameters on atoms Cs',required=True)
    parser.add_argument('-ep','--epochs', type=int,default=100,help='epochs to run for stochastic algorithm',required=True)
    parser.add_argument('-b','--batch_size', type=int,default=16,help='batch size for stochastic updates',required=True)
    parser.add_argument('-lrC','--learning_rate_Cs', type=float,default=0.001,help='learning rate for SGD updates of atoms Cs- Adam is included with default values',required=True)
    parser.add_argument('-lrh','--learning_rate_hs', type=float,default=0.0001,help='learning rate for SGD updates of atoms hs- Adam is included with default values',required=True)
    parser.add_argument('-mode','--mode', type=str,default='ADJ',help='graph representation for graphs',choices=['ADJ','SP','LAP'])
    parser.add_argument('-eps','--eps', type=float,default=10**(-5),help='precision to assert unmixings convergence')
    parser.add_argument('-maxout','--max_iter_outer', type=int,default=20,help='maximum number of iterations for BCD algorithm')
    parser.add_argument('-maxin','--max_iter_inner', type=int,default=100,help='maximum number of iterations for FW algorithm for w_k')
    parser.add_argument('-s','--list_seed', type=str,default=str([0]),help='seed to initialize stochastic algorithm and ensure reproductibility')
    parser.add_argument('-initC','--init_mode_Cs', type=int,default=1,help='initialization mode for atoms Cs. 0 = random / 1 = sampling')
    parser.add_argument('-inith','--init_mode_hs', type=int,default=1,help='initialization mode for atoms hs. 0 = uniform / 1 = random')
    parser.add_argument('-opt','--use_optimizer', type=bool,default=True,help='True for using Adam - False for vanilla SGD')
    
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
    centering= True #centering potentials from proposition 2 of the paper for numerical stability
    
    list_number_atoms = [int(x) for x in args.list_number_atoms[1:-1].split(',')]
    list_shape_atoms = [int(x) for x in args.list_shape_atoms[1:-1].split(',')]
    list_seed = [int(x) for x in args.list_seed[1:-1].split(',')]
    list_l2reg_Cs = [float(x) for x in args.list_l2reg_Cs[1:-1].split(',')]
    
    experiment_repo = "/%s_experiments"%args.dataset_name
    for number_Cs in list_number_atoms:
        for shape_Cs in list_shape_atoms:
            for seed in list_seed:
                for l2_reg_c in list_l2reg_Cs:
                    if args.init_mode_hs==1:
                        if args.use_optimizer:
                            experiment_name= "/extended_%s/"%args.mode+"%s_GDLextended_Adambatch%s_sampledC_randomH_S%sN%s_l2C%s_lrC%s_lrH%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(args.dataset_name,args.batch_size,shape_Cs,number_Cs,l2_reg_c,args.learning_rate_Cs,args.learning_rate_hs,args.max_iter_outer,args.max_iter_inner,args.eps,args.epochs,seed)
                        else:
                            experiment_name= "/extended_%s/"%args.mode+"%s_GDLextended_SGDbatch%s_sampledC_randomH_S%sN%s_l2C%s_lrC%s_lrH%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(args.dataset_name,args.batch_size,shape_Cs,number_Cs,l2_reg_c,args.learning_rate_Cs,args.learning_rate_hs,args.max_iter_outer,args.max_iter_inner,args.eps,args.epochs,seed)
        
                    
                    elif args.init_mode_hs==0:
                        if args.use_optimizer:
                            experiment_name= "/extended_%s/"%args.mode+"%s_GDLextended_Adambatch%s_sampledC_unifH_S%sN%s_l2C%s_lrC%s_lrH%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(args.dataset_name,args.batch_size,shape_Cs,number_Cs,l2_reg_c,args.learning_rate_Cs,args.learning_rate_hs,args.max_iter_outer,args.max_iter_inner,args.eps,args.epochs,seed)
                        else:
                            experiment_name= "/extended_%s/"%args.mode+"%s_GDLextended_SGDbatch%s_sampledC_unifH_S%sN%s_l2C%s_lrC%s_lrH%s_maxiterout%s_maxiterin%s_eps%s_epochs%s_seed%s/"%(args.dataset_name,args.batch_size,shape_Cs,number_Cs,l2_reg_c,args.learning_rate_Cs,args.learning_rate_hs,args.max_iter_outer,args.max_iter_inner,args.eps,args.epochs,seed)
        
        
                    print('experiment_name:',experiment_name)
                    local_path = os.path.abspath('../')
                    
                    full_path = local_path +experiment_repo+experiment_name
                    print(full_path)
                    
                    DL = algos_sto.extended_GW_Unsupervised_DictionaryLearning(args.dataset_name,args.mode, number_atoms=number_Cs, shape_atoms=shape_Cs, experiment_repo=experiment_repo, experiment_name=experiment_name,data_path='../data/')
                
                    _= DL.Stochastic_algorithm(l2_reg_c, eps = args.eps,max_iter_outer=args.max_iter_outer,max_iter_inner = args.max_iter_inner,
                                               lr_Cs= args.learning_rate_Cs,lr_hs=args.learning_rate_hs,batch_size=args.batch_size,
                                               beta_1=0.9, beta_2=0.99,epochs= args.epochs,algo_seed=seed,
                                               init_mode_atoms=args.init_mode_Cs, init_hs_mode=args.init_mode_hs,verbose =False,
                                               centering=centering,use_optimizer=args.use_optimizer)
           
                    print('done learning')
                    
                    
                    print('LEARNING PROCESS HAS BEEN COMPLETED CORRECTLY')
                    
                
                    ac_str= 'unmixing_v2_l2C%s_unif_maxiterout%s_maxiterin%s_eps%s'%(l2_reg_c,args.max_iter_outer,args.max_iter_inner,args.eps)
                    ah_str= 'unmixing_v2_l2C%s_unif_maxiterout%s_maxiterin%s_eps%s'%(l2_reg_c,args.max_iter_outer,args.max_iter_inner,args.eps)
                    if not os.path.exists(full_path+'/%s.npy'%ac_str):
                        ac,ah,losses = DL.extendedGDL_unmixing(l2_reg_c,args.eps,args.max_iter_outer,args.max_iter_inner,seed=0,verbose=False,centering=centering)
                
                        np.save(full_path+'/%s.npy'%ac_str, ac)
                        np.save(full_path+'/%s.npy'%ah_str,ah)
                        np.save(full_path+'/reconstruction.npy',losses)
                    else:
                        print(' existing mixture file')
                    