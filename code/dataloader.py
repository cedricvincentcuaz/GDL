from graph_class import Graph,wl_labeling
import networkx as nx
#from utils import per_section,indices_to_one_hot
from collections import defaultdict
import numpy as np
import math
import os
from tqdm import tqdm
import pickle
#%%
def indices_to_one_hot(number, nb_classes,label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""
    
    if number==label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]


def per_section(it, is_delimiter=lambda x: x.isspace()):
    ret = []
    for line in it:
        if is_delimiter(line):
            if ret:
                yield ret  # OR  ''.join(ret)
                ret = []
        else:
            ret.append(line.rstrip())  # OR  ret.append(line)
    if ret:
        yield ret
        
def data_streamer(data_path,batchsize_bylabel, selected_labels,balanced_shapes=False,sampling_seed=None,return_idx = False):
    batch_graphs, batch_labels = [],[]
    if not (sampling_seed is None):
        np.random.seed(sampling_seed)
    if return_idx:
        batch_idx=[]
    if not balanced_shapes:
        for label in selected_labels:
            files = os.listdir(data_path+'/label%s/'%label)
            
            file_idx = np.random.choice(range(len(files)), size=batchsize_bylabel,replace=False)
            for idx in file_idx:
                    
                batch_graphs.append(np.load(data_path+'/label%s/'%label+files[idx]))
                batch_labels.append(label)
                if return_idx:
                    ls = file_idx[idx].split('.')
                    batch_idx.append(int(ls[0][:5]))
        if return_idx:
            return batch_graphs,batch_labels,batch_idx
        else:
            return batch_graphs,batch_labels
    else:
        shapes={}
        graphidx_shapes={}
        for label in selected_labels:
            files = os.listdir(data_path+'/label%s/'%label)
            shapes[label]=[]
            graphidx_shapes[label]=[]
            print('label = ', label)
            for filename in tqdm(files):
                local_idx = int(filename.split('.')[0][5:])
                graphidx_shapes[label].append(local_idx)
                shapes[label].append(np.load(data_path+'/label%s/'%label+filename).shape[0])
            unique_shapes= np.unique(shapes[label])
            sizebylabel = batchsize_bylabel//len(unique_shapes)
            for local_shape in unique_shapes:
                local_idx_list = np.argwhere(shapes[label]==local_shape)[:,0]
                sampled_idx = np.random.choice(local_idx_list, size=sizebylabel, replace=False)
                for idx in sampled_idx:
                    graphidx = graphidx_shapes[label][idx]
                    batch_graphs.append(np.load(data_path+'/label%s/graph%s.npy'%(label,graphidx)))
                    batch_labels.append(label)
                    
        return batch_graphs,batch_labels
    
    
def load_local_data(data_path,name,one_hot=False,attributes=True,use_node_deg=False):
    """ Load local datasets - modified version
    Parameters
    ----------
    data_path : string
                Path to the data. Must link to a folder where all datasets are saved in separate folders
    name : string
           Name of the dataset to load. 
           Choices=['mutag','ptc','nci1','imdb-b','imdb-m','enzymes','protein','protein_notfull','bzr','cox2','synthetic','aids','cuneiform'] 
    one_hot : integer
              If discrete attributes must be one hotted it must be the number of unique values.
    attributes :  bool, optional
                  For dataset with both continuous and discrete attributes. 
                  If True it uses the continuous attributes (corresponding to "Node Attr." in [5])
    use_node_deg : bool, optional
                   Wether to use the node degree instead of original labels. 
    Returns
    -------
    X : array
        array of Graph objects created from the dataset
    y : array
        classes of each graph    
    References
    ----------    
    [5] Kristian Kersting and Nils M. Kriege and Christopher Morris and Petra Mutzel and Marion Neumann 
        "Benchmark Data Sets for Graph Kernels"
    """
    name_to_path_discretefeatures={'mutag':data_path+'/MUTAG_2/',
                                   'ptc':data_path+'/PTC_MR/',
                                   'triangles':data_path+'/TRIANGLES/'}
    name_to_path_realfeatures={'enzymes':data_path+'/ENZYMES_2/',
                               'protein':data_path+'/PROTEINS_full/',
                               'protein_notfull':data_path+'/PROTEINS/',
                               'bzr':data_path+'/BZR/',
                               'cox2':data_path+'/COX2/'}
    name_to_rawnames={'mutag':'MUTAG', 'ptc':'PTC_MR','triangles':'TRIANGLES',
                      'enzymes':'ENZYMES','protein':'PROTEINS_full','protein_notfull':'PROTEINS',
                      'bzr':'BZR','cox2':'COX2',
                      'imdb-b':'IMDB-BINARY', 'imdb-m':'IMDB-MULTI','reddit':'REDDIT-BINARY','collab':'COLLAB'}
    if name in ['mutag','ptc','triangles']:
        dataset = build_dataset_discretefeatures(name_to_rawnames[name],
                                                 name_to_path_discretefeatures[name],
                                                 one_hot=one_hot)
    elif name in ['enzymes','protein', 'protein_notfull','bzr','cox2']:
        dataset = build_dataset_realfeatures(name_to_rawnames[name], name_to_path_realfeatures[name],
                                             type_attr='real',use_node_deg=use_node_deg)
    elif name in ['imdb-b','imdb-m','reddit', 'collab']:
        rawname  = name_to_rawnames[name]
        dataset = build_dataset_withoutfeatures(rawname, data_path+'/%s/'%rawname,use_node_deg= use_node_deg)
    else:
        raise 'unknown dataset'
    X,y=zip(*dataset)
    return np.array(X),np.array(y)
    
#%%

def label_wl_dataset(X,h):
    X2=[]
    for x in X:
        x2=Graph()
        x2.nx_graph=wl_labeling(x.nx_graph,h=2)
        X2.append(x2)
    return X2

#%%

def histog(X,bins=10):
    node_length=[]
    for graph in X:
        node_length.append(len(graph.nodes()))
    return np.array(node_length),{'histo':np.histogram(np.array(node_length),bins=bins),'med':np.median(np.array(node_length))
            ,'max':np.max(np.array(node_length)),'min':np.min(np.array(node_length))}

def node_labels_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=int(elt)
            k=k+1
    return node_dic

def node_attr_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=[float(x) for x in elt.split(',')]
            k=k+1
    return node_dic

def graph_label_list(path,name,real=False):
    graphs=[]
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            if real:
                graphs.append((k,float(elt)))
            else:
                graphs.append((k,int(elt)))
            k=k+1
    return graphs
    
def graph_indicator(path,name):
    data_dict = defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            data_dict[int(elt)].append(k)
            k=k+1
    return data_dict

def compute_adjency(path,name):
    adjency= defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        for elt in sections[0]:
            adjency[int(elt.split(',')[0])].append(int(elt.split(',')[1]))
    return adjency


def all_connected(X):
    a=[]
    for graph in X:
        a.append(nx.is_connected(graph.nx_graph))
    return np.all(a)

#%% TO FACTORIZE !!!!!!!!!!!


def build_dataset_discretefeatures(dataset_name,path,one_hot=False):
    assert dataset_name in ['MUTAG','PTC_MR','TRIANGLES']
    name_to_ncategories={'MUTAG':7, 'PTC_MR':18}
    n_categories = name_to_ncategories[dataset_name]
    graphs=graph_label_list(path,'%s_graph_labels.txt'%dataset_name)
    adjency=compute_adjency(path,'%s_A.txt'%dataset_name)
    data_dict=graph_indicator(path,'%s_graph_indicator.txt'%dataset_name)
    node_dic=node_labels_dic(path,'%s_node_labels.txt'%dataset_name) 
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],n_categories)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data



def build_dataset_realfeatures(dataset_name,path,type_attr='label',use_node_deg=False):
    assert dataset_name in ['PROTEINS_full','PROTEINS','ENZYMES','BZR','COX2']
    if type_attr=='label':
        node_dic=node_labels_dic(path,'%s_node_labels.txt'%dataset_name)
    if type_attr=='real':
        node_dic=node_attr_dic(path,'%s_node_attributes.txt'%dataset_name)
    graphs=graph_label_list(path,'%s_graph_labels.txt'%dataset_name)
    adjency=compute_adjency(path,'%s_A.txt'%dataset_name)
    data_dict=graph_indicator(path,'%s_graph_indicator.txt'%dataset_name)
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data


def build_dataset_withoutfeatures(dataset_name, path, use_node_deg=False):
    assert dataset_name in ['IMDB-MULTI','IMDB-BINARY','REDDIT-BINARY','COLLAB']
    graphs=graph_label_list(path,'%s_graph_labels.txt'%dataset_name)
    adjency=compute_adjency(path,'%s_A.txt'%dataset_name)
    data_dict=graph_indicator(path,'%s_graph_indicator.txt'%dataset_name)
    data=[]
    for i in tqdm(graphs,desc='loading graphs'):
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            #g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

    