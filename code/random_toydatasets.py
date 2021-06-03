import numpy as np
import networkx as nx

from networkx.generators.community import stochastic_block_model as sbm

#%%


def toy_sbm3( max_blocs, graphs_by_group,graph_sizes, intra_p, inter_p , seed):
    """From Optimal Transport for structured data with application on graphs :
        - 4 groups of community graphs 
        - groups are defined wrt the number of communities inside each graph (here without labels)
        - 10 graphs by groups
        - the number of nodes of each graph is drawn randomly from {20...50 / 10}
    --> Version inspired from them / definitely not the same one
    """
    dataset = {}
    np.random.seed(seed)
    for bloc_qt in range(1,max_blocs+1):
        dataset[bloc_qt]=[]
        
        for _ in range(graphs_by_group):
            #number of nodes in the graph
            n_nodes=np.random.choice(graph_sizes)
            if n_nodes%bloc_qt ==0:
                sizes = [n_nodes//bloc_qt for _ in range(bloc_qt)]
            else:
                residuals = (n_nodes%bloc_qt)
                sizes =[n_nodes//bloc_qt for _ in range(bloc_qt)]
                for i in range(residuals):
                    #pos= np.random.choice(len(sizes))
                    #we delete this feature - boring for supervised analysis
                    sizes[i]+=1
            probs = inter_p*np.ones((bloc_qt, bloc_qt))
            np.fill_diagonal(probs, intra_p)
            local_seed= np.random.choice(range(100))
            G=sbm(sizes,probs,seed=int(local_seed))
            dataset[bloc_qt].append(nx.to_numpy_array(G))
            
    return dataset


def toy_sbm2clusters_1Dinterpolation( graph_qt,graph_sizes, cluster_perturbation, intra_p, inter_p , seed):
    """
    Generate SBM with 2 clusters of different number of nodes
    with varying cluster size
    """
    dataset = []
    np.random.seed(seed)
    
    def perturbate_size_vector(cluster_perturbation, sizes_vector, n_nodes):
        #We sample a cluster - as GW invariant with perturbation we keep with first cluster
        #Apply the random size perturbation based on cluster_perturbation parameter
        #Propagate the rest to keep the proper number of nodes n_nodes
        rest = n_nodes
        n = len(sizes_vector)
        size_rate= 1 - cluster_perturbation
        #make sure that a cluster keeps a size >= 2
        assert sizes_vector[0]>2
        max_perturbation = max(1, int(sizes_vector[0]*size_rate))
        
        perturbation0= np.random.choice(range(1,max_perturbation))
        sizes_vector[0]-= perturbation0
        rest-= sizes_vector[0]
        for i in range(1, n-1):
            max_perturbation = max(1, int(sizes_vector[i]*size_rate))
            assert sizes_vector[i]>2
        
            perturbation = np.random.choice(np.random.choice(range(1,max_perturbation)))
            sizes_vector[i]-=perturbation
            rest-=sizes_vector[i]
        sizes_vector[-1] = rest
        return sizes_vector
    
    bloc_qt=2
    stacked_rates= []
    for k in range(graph_qt):
        #number of nodes in the graph
        n_nodes=np.random.choice(graph_sizes)
        #Here if we have more than one cluster we had the perturbation
        #on cluster size depending on size_perturbation rate
        
        if n_nodes%bloc_qt ==0:
            
            sizes = [n_nodes//bloc_qt for _ in range(bloc_qt)]
        else:
            residuals = (n_nodes%bloc_qt)
            sizes =[n_nodes//bloc_qt for _ in range(bloc_qt)]
            for i in range(residuals):
                #pos= np.random.choice(len(sizes))
                #we delete this feature - boring for supervised analysis
                sizes[i]+=1
        
        probs = inter_p*np.ones((bloc_qt, bloc_qt))
        np.fill_diagonal(probs, intra_p)
        local_seed= np.random.choice(range(100))
        sizes = perturbate_size_vector(cluster_perturbation,sizes, n_nodes)
        local_rate = sizes[0]/n_nodes
        stacked_rates.append(local_rate)
        print('Graph %s - perturbated_size:%s / rate size C1: %s'%(k,sizes,local_rate))
        G=sbm(sizes,probs,seed=int(local_seed))
        dataset.append(nx.to_numpy_array(G))
        
    return dataset,stacked_rates


def toy_sbm3_discretefeatures( max_blocs, graphs_by_group,graph_sizes, intra_p, inter_p , seed):
    """From Optimal Transport for structured data with application on graphs :
        - 4 groups of community graphs 
        - groups are defined wrt the number of communities inside each graph (here without labels)
        - 10 graphs by groups
        - the number of nodes of each graph is drawn randomly from {20...50 / 10}
    --> Version inspired from them / definitely not the same one
    """
    dataset = {}
    np.random.seed(seed)
    for bloc_qt in range(1,max_blocs+1):
        dataset[bloc_qt]=[]
        
        for _ in range(graphs_by_group):
            #number of nodes in the graph
            n_nodes=np.random.choice(graph_sizes)
            if n_nodes%bloc_qt ==0:
                sizes = [n_nodes//bloc_qt for _ in range(bloc_qt)]
            else:
                residuals = (n_nodes%bloc_qt)
                sizes =[n_nodes//bloc_qt for _ in range(bloc_qt)]
                for i in range(residuals):
                    #pos= np.random.choice(len(sizes))
                    #we delete this feature - boring for supervised analysis
                    sizes[i]+=1
            probs = inter_p*np.ones((bloc_qt, bloc_qt))
            np.fill_diagonal(probs, intra_p)
            local_seed= np.random.choice(range(100))
            G=sbm(sizes,probs,seed=int(local_seed))
            matG=nx.to_numpy_array(G)
            #pick a set of features  
            matA = np.zeros((matG.shape[0],1))
            feature = np.random.choice([1,2])
            matA+=feature
            label = 2*(bloc_qt-1) +feature
            dataset[bloc_qt].append((matG,matA,label))
            
    return dataset


def toy_sbm3_realfeatures( max_blocs, graphs_by_group,graph_sizes, intra_p, inter_p , y0_mean,y1_mean, y_std, seed):
    """From Optimal Transport for structured data with application on graphs :
        - 4 groups of community graphs 
        - groups are defined wrt the number of communities inside each graph (here without labels)
        - 10 graphs by groups
        - the number of nodes of each graph is drawn randomly from {20...50 / 10}
    --> Version inspired from them / definitely not the same one
    """
    features_values = [y0_mean,y1_mean]
    dataset = {}
    np.random.seed(seed)
    for bloc_qt in range(1,max_blocs+1):
        dataset[bloc_qt]=[]
        
        for _ in range(graphs_by_group):
            #number of nodes in the graph
            n_nodes=np.random.choice(graph_sizes)
            if n_nodes%bloc_qt ==0:
                sizes = [n_nodes//bloc_qt for _ in range(bloc_qt)]
            else:
                residuals = (n_nodes%bloc_qt)
                sizes =[n_nodes//bloc_qt for _ in range(bloc_qt)]
                for i in range(residuals):
                    #pos= np.random.choice(len(sizes))
                    #we delete this feature - boring for supervised analysis
                    sizes[i]+=1
            probs = inter_p*np.ones((bloc_qt, bloc_qt))
            np.fill_diagonal(probs, intra_p)
            local_seed= np.random.choice(range(100))
            G=sbm(sizes,probs,seed=int(local_seed))
            matG=nx.to_numpy_array(G)
            #pick a set of features  
            #by picking mean 
            choice = np.random.choice([0,1])
            matA = np.random.normal(loc = features_values[choice], scale = y_std, size = (matG.shape[0],1))
            
            label = 2*(bloc_qt-1) +choice
            dataset[bloc_qt].append((matG,matA,label))
            
    return dataset

def toy_sbm3_unbalanced( max_blocs, graphs_by_group,graph_sizes, cluster_perturbation,intra_p, inter_p , seed):
    """From Optimal Transport for structured data with application on graphs :
        - 4 groups of community graphs 
        - groups are defined wrt the number of communities inside each graph (here without labels)
        - 10 graphs by groups
        - the number of nodes of each graph is drawn randomly from {20...50 / 10}
    --> Version inspired from them / definitely not the same one
    """
    dataset = {}
    np.random.seed(seed)
    
    def perturbate_size_vector(cluster_perturbation, sizes_vector, n_nodes):
        #We sample a cluster - as GW invariant with perturbation we keep with first cluster
        #Apply the random size perturbation based on cluster_perturbation parameter
        #Propagate the rest to keep the proper number of nodes n_nodes
        rest = n_nodes
        n = len(sizes_vector)
        size_rate= 1 - cluster_perturbation
        #make sure that a cluster keeps a size >= 2
        assert sizes_vector[0]>2
        max_perturbation = max(1, int(sizes_vector[0]*size_rate))
        
        perturbation0= np.random.choice(range(1,max_perturbation))
        sizes_vector[0]-= perturbation0
        rest-= sizes_vector[0]
        for i in range(1, n-1):
            max_perturbation = max(1, int(sizes_vector[i]*size_rate))
            assert sizes_vector[i]>2
        
            perturbation = np.random.choice(np.random.choice(range(1,max_perturbation)))
            sizes_vector[i]-=perturbation
            rest-=sizes_vector[i]
        sizes_vector[-1] = rest
        return sizes_vector
    
    for bloc_qt in range(1,max_blocs+1):
        dataset[bloc_qt]=[]
        
        for _ in range(graphs_by_group):
            #number of nodes in the graph
            n_nodes=np.random.choice(graph_sizes)
            #Here if we have more than one cluster we had the perturbation
            #on cluster size depending on size_perturbation rate
            
            if n_nodes%bloc_qt ==0:
                
                sizes = [n_nodes//bloc_qt for _ in range(bloc_qt)]
            else:
                residuals = (n_nodes%bloc_qt)
                sizes =[n_nodes//bloc_qt for _ in range(bloc_qt)]
                for i in range(residuals):
                    #pos= np.random.choice(len(sizes))
                    #we delete this feature - boring for supervised analysis
                    sizes[i]+=1
            
            probs = inter_p*np.ones((bloc_qt, bloc_qt))
            np.fill_diagonal(probs, intra_p)
            local_seed= np.random.choice(range(100))
            if bloc_qt>1:
                #print('initial_sizes:', sizes)
                
                sizes = perturbate_size_vector(cluster_perturbation,sizes, n_nodes)
                #print('perturbated sizes:',sizes)
            G=sbm(sizes,probs,seed=int(local_seed))
            dataset[bloc_qt].append(nx.to_numpy_array(G))
            
    return dataset



   
def same_graph_generated(dataset):
    similar_graph=False
    last_bloc = list(dataset.keys())[-1]
    bloc= list(dataset.keys())[0]
    while (not similar_graph) and (bloc<last_bloc+1):
        n_graph = len(dataset[bloc])
        i=0; j=1
        while (not similar_graph) and (i<(n_graph-1)):
        
            if np.all(dataset[bloc][i]== dataset[bloc][j]):
                similar_graph=True
            elif j< (n_graph-1):
                j+=1
            else:
                i+=1
                j=i+1
        bloc+=1
    return similar_graph
    

def same_graph_generated_count(dataset):
    
    last_bloc = list(dataset.keys())[-1]
    bloc= list(dataset.keys())[0]
    similar_graph_counts = 0
    while (bloc<last_bloc+1):
        n_graph = len(dataset[bloc])
        i=0; j=1
        while (i<(n_graph-1)):
        
            if np.all(dataset[bloc][i]== dataset[bloc][j]):
                
                similar_graph_counts+=1
            elif j< (n_graph-1):
                j+=1
            else:
                i+=1
                j=i+1
        bloc+=1
    return similar_graph_counts
