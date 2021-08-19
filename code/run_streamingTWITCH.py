#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import GDL_GWalgo as algos_sto
#%% settings
"""
WARNING:
    - The dataset twitch_egos available here: https://snap.stanford.edu/data/twitch_ego_nets.html
    was split by labels such that:
        - the data repository ../data/twitch_egos/ contains 
        subfolder label0/ and label1/
        - each subfolder label{0,1} contains the graph of index k in the file graphk.npy
    parsed according to the original file twitch_egos_graph_labels.txt

    The result of this preprocessing is stored in ./data/twitch_egos.tar.gz
"""


dataset_name = 'twitch_egos'
dataset_mode = 'full'
number_Cs= 2
shape_Cs = 14
l2_reg=0
streaming_mode = 1
steps = 127000 #number of updates for streaming experiment (approximately the size of the full dataset)
event_steps = [63500] # switch class at the middle 
checkpoint_steps = 3000 # each time steps%checkpoint steps we compute unmixings over the whole dataset for tracking
sampler_batchsize=1000 # number of graphs stored in memory
checkpoint_size= 2000 #number of sampled graphs used to track unmixing. Changing at each checkpoint
eps = 10**(-5)
max_iter_outer = 20
max_iter_inner = 100
lr = 0.01
graph_mode='ADJ'
seed=0
#%%

experiment_repo = "/%s_experiments"%dataset_name
experiment_name= "/%s/"%graph_mode+dataset_name+dataset_mode+"_stream%s_randomC_S%sN%s_l2%s_lrC%s_maxiterout%s_maxiterin%s_eps%s_steps%s_seed%s/"%(streaming_mode,shape_Cs,number_Cs,l2_reg,lr,max_iter_outer,max_iter_inner,eps,steps,seed)
                   
local_path = os.path.abspath('../')

full_path = local_path +experiment_repo+experiment_name
print(full_path)
DL = algos_sto.Online_GW_Twitch_streaming(dataset_name=dataset_name,
                                          dataset_mode=dataset_mode,
                                          graph_mode=graph_mode, 
                                          number_Cs=number_Cs, 
                                          shape_Cs=shape_Cs, 
                                          experiment_repo=experiment_repo,
                                          experiment_name=experiment_name)
save_chunks = True 
#save unmixings computed at each checkpoint
#for later visualization as illustrated in the paper.
    
_=DL.Online_Learning_fulldataset(l2_reg,
                                 eps = eps,
                                 max_iter_outer=max_iter_outer,
                                 max_iter_inner=max_iter_inner,
                                 sampler_batchsize=sampler_batchsize,
                                 checkpoint_size=checkpoint_size, 
                                 lr_Cs= lr,                                 
                                 steps = steps, 
                                 algo_seed=seed,
                                 event_steps=event_steps, 
                                 streaming_mode=streaming_mode, 
                                 checkpoint_steps=checkpoint_steps,
                                 save_chunks =save_chunks,
                                 verbose=False)
    