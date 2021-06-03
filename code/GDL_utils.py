"""
@author: cvincentcuaz
"""
import numpy as np
import ot
import local_gromov_optimizer
from tqdm import tqdm


#%% Compute mahalanobis for GDL's models
# Cf proposition 1 of the paper: upper-bound of the GW distance 
# in our embeddings space with a mahalanobis distance
# + see supplementary for the extension to the Fused Gromov-Wasserstein case
# for labeled graphs.

def compute_mahalanobis_matrix(atoms:np.array,
                               h=None):
    """ Compute Mahalanobis matrix for GDL's embeddings on graphs without node attributes
    """
    number_atoms = atoms.shape[0]
    shape_atoms = atoms.shape[-1]
    if h is None:
        h = np.ones(shape_atoms)/shape_atoms
    M = np.zeros((number_atoms,number_atoms))
    Dh = np.diag(h)
    for i in range(number_atoms):
        for j in range(i,number_atoms):
            
            M[i,j] = M[j,i] = np.sum((Dh.dot(atoms[i]))*(atoms[j].dot(Dh)))
    return M

def compute_mahalanobisFGW_matrix(Cs:np.array,
                                  As:np.array,
                                  alpha:float=0.5,
                                  h:np.array=None):
    """Compute Mahalanobis matrix for GDL's embeddings on labeled graphs
        Cs: refers to the structure part of the atoms {Cbar_s}
        As: refers to the features part of the atoms {Abar_s}
    """
    number_atoms=Cs.shape[0]
    shape_atoms=Cs.shape[-1]
    if h is None:
        h = np.ones(shape_atoms)/shape_atoms
    M=np.zeros((number_atoms,number_atoms))
    Dh = np.diag(h)
    for i in range(number_atoms):
        for j in range(i,number_atoms):
            structure_value =   np.sum((Dh.dot(Cs[i]))*(Cs[j].dot(Dh)))
            features_value = np.sum( As[i].dot(As[j].T)*Dh)
            M[i,j]=M[j,i] = alpha*structure_value + (1-alpha)*features_value
    return M




#%% GW GDL utils

def np_init_matrix_GW2(C1, C2, p, q):
    
    f1_=C1**2
    
    f2_=C2**2
    constC1 = f1_.dot(p[:,None]).dot(np.ones((1, q.shape[0])))
    constC2 = np.ones((p.shape[0], 1)).dot(q[None,:].dot(f2_))
    constC=constC1+constC2
    hC1=C1
    hC2=2*C2
    return constC, hC1, hC2


def np_GW2(C1,C2, p=None,q=None,T_init=None,T_star =None):
    if p is None:
        p=np.ones(C1.shape[0])/C1.shape[0]
    if q is None:
        q=np.ones(C2.shape[0])/C2.shape[0]
    constC, hC1, hC2 = np_init_matrix_GW2(C1, C2, p, q)
    if T_star is None:
        if T_init is None:
            T_star=ot.gromov.gromov_wasserstein(C1,C2,p,q, 'square_loss')
        else:
            T_star=local_gromov_optimizer.gromov_wasserstein(C1,C2,p,q, 'square_loss',G0=T_init)
    
        A = - hC1.dot(T_star).dot(hC2.transpose())
        tens = constC + A    
    else:
        A = - hC1.dot(T_star).dot(hC2.transpose())
        tens = constC + A    
    return np.sum(tens * T_star), T_star

        
def np_GW2_extendedDL(C1,C2, p=None,q=None,T_init=None,T_star =None,OT_loss ='square_loss',verbose=False,centering=True):
    if p is None:
        p=np.ones(C1.shape[0])/C1.shape[0]
    if q is None:
        q=np.ones(C2.shape[0])/C2.shape[0]
    #print('shapes C1: %s  / h1: %s'%(C1.shape, p.shape))
    
    #print('shapes C2: %s  / h2: %s'%(C2.shape, q.shape))
    constC, hC1, hC2 = np_init_matrix_GW2(C1, C2, p, q)
    if T_star is None:
        if T_init is None:
            T_star,log=ot.gromov.gromov_wasserstein(C1,C2,p,q, 'square_loss',log=True) #log =True here to output the distance directly
            
        else:
            T_star,log=local_gromov_optimizer.gromov_wasserstein(C1,C2,p,q, 'square_loss',G0=T_init, log=True)
        GW_dist = log['gw_dist']
        
        #print('GW_dist from GW solver:', GW_dist)
        #Compute gradients of GW over T 
        G=ot.gromov.gwggrad(constC, hC1, hC2, T_star)
        #Compute Emd to get potentials (cf Wasserstein strong duality)
        T_star2,log2=ot.emd(p,q,G,log=True)
        
        #print('potentials from EMD:', log2['u'], log2['v'])
        if not centering:
            if verbose:
                print('GW dist from C:', GW_dist)
                local_loss = (log2['u'].T).dot(p)+ (log2['v'].T).dot(q)
                print('emd loss before centering:', local_loss/2)
                
            return GW_dist,T_star, log2['u'], log2['v']
        if centering:
            centered_u = log2['u'] - log2['u'].mean()
            centered_v = log2['v'] - log2['v'].mean()
            if verbose:
                print('GW dist from C:', GW_dist)
                local_loss = (log2['u'].T).dot(p)+ (log2['v'].T).dot(q)
                print('emd loss before centering:', local_loss/2)
                local_loss=(centered_u.T).dot(p)+(centered_v.T).dot(q)
                print('emd loss after centering:',local_loss/2 )
            #print('centered potentials from EMD:', centered_u, centered_v)
            #print('distance between both OT :', np.linalg.norm(T_star - T_star2, ord =2))
            return GW_dist,T_star, centered_u, centered_v
    else:
        #Need to check if this evaluation process as to be changed
        A = - hC1.dot(T_star).dot(hC2.transpose())
        tens = constC + A    
        GW_dist = np.sum(tens * T_star)
        if verbose:
            print('GW dist from C:', GW_dist)
            
        return GW_dist,T_star

def np_simplex_projection(a):
    descending_idx = np.argsort(a)[::-1]
    u = a[descending_idx]
    rho= 0.
    lambda_= 1.
    for i in range(u.shape[0]):
        value = u[i] + (1- np.sum(u[:(i+1)]))/(i+1)
        if value>0:
            rho+=1
            lambda_-=u[i]
        else:
            break
    return np.maximum(a+lambda_/rho, np.zeros_like(a))

def np_sum_scaled_mat(Cs, a):
    """ handle only linear case """
    
    return np.sum(np.stack([Cs[k]*a[k] for k in range(a.shape[0])]), axis= 0)

#%% FGW utils



def numpy_init_matrix(C1, C2, p, q,dtype=np.float64):
    
    f1_=C1**2
    
    f2_=C2**2
    q_ones = np.ones(q.shape[0])
    p_ones= np.ones(p.shape[0])
    constC1 = f1_.dot(p[:,None]).dot(q_ones[None,:])
    constC2 = p_ones[:,None].dot(f2_.dot(q[:,None]).T)
    constC=constC1+constC2
    hC1=C1
    hC2=2*C2
    return constC, hC1, hC2

        
def numpy_FGW_loss(C1,C2,A1,A2,p,q, alpha, M=None, OT_loss='square_loss',features_dist='l2',T_init=None,T_star=None,dtype=np.float64):
    if p is None:
        p=np.ones(C1.shape[0])/C1.shape[0]
    if q is None:
        q=np.ones(C2.shape[0])/C2.shape[0]
    constC, hC1, hC2 = np_init_matrix_GW2(C1, C2, p, q)
    if M is None:
        #Compute the pairwise distance between features of both graphs
        FS2 = (A1**2).dot(np.ones((A1.shape[1], A2.shape[0])))
        FT2 = (np.ones((A1.shape[0], A1.shape[1]))).dot((A2**2).T)
        M= FS2+FT2 - 2*A1.dot(A2.T)
    
    if T_star is None:
        if T_init is None:
            T_star = ot.gromov.fused_gromov_wasserstein(M=M, C1=C1, C2=C2, p=p,q=q,loss_fun=OT_loss, alpha=alpha)
        else:
            T_star = local_gromov_optimizer.fused_gromov_wasserstein(M=M, C1=C1, C2=C2, p=p,q=q,loss_fun=OT_loss, G0 = T_init,alpha=alpha)
    
    GW_A = - hC1.dot(T_star).dot(hC2.T)
    GW_tens = constC+GW_A
    FGW_loss = (1-alpha)*np.sum(T_star*M)+alpha*np.sum(T_star*GW_tens)
    return FGW_loss,T_star

#%% graph representations

def compute_diffusive_distance(X):
    #negative entries
    n=X.shape[0]
    D =np.sum(X,axis=1)
    new_X=np.zeros_like(X)
    for i in range(n):
        for j in range(i,n):
            new_X[i,j] = X[i,i]/(D[i]**2)+X[j,j]/(D[j]**2)-2*X[i,j]/(D[i]*D[j])
            new_X[j,i] = X[i,i]/(D[i]**2)+X[j,j]/(D[j]**2)-2*X[i,j]/(D[i]*D[j])
    return new_X


def compute_frozen_distance(X):
    n=X.shape[0]
    D =np.sum(X,axis=1)
    new_X=np.zeros_like(X)
    for i in range(n-1):
        for j in range(i,n):
            if i==j:
                new_X[i,j]= (1/D[i]) -1
            else:
                    
                new_X[j,i] = new_X[i,j] = 1/D[i] + 1/D[j]
    return new_X

def compute_commute_distance(X):
    #squared euclidean distance
    degrees= np.sum(X,axis=0)
    assert np.all(degrees>0) #ensure there is no isolated node
    D = np.diag(degrees)
    E_inf = degrees.dot(degrees.T)
    B=np.linalg.inv(D-X+E_inf)
    n = X.shape[0]
    Dist = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            Dist[i,j]=Dist[j,i]= B[i,i]+B[j,j]-2*B[i,j]
    return Dist

def compute_sif_distance(X):
    commute_dist = compute_commute_distance(X)
    diffusive_dist = compute_diffusive_distance(X)
    frozen_dist = compute_frozen_distance(X)
    return commute_dist - diffusive_dist - frozen_dist



#%% MCMC hit and run for initializations of GW conditional gradient algorithm
#https://github.com/trneedham/Spectral-Gromov-Wasserstein


def gw_equality_constraints(p,q):
    # Inputs: probability row vectors
    # Output: matrices A and b defining equality constraints

    m = len(p)
    n = len(q)

    A_p_type = np.zeros([m,m*n])
    b_p_type = p.reshape(m,1)

    for i in range(m):
        row = i*n*[0] + n*[1] + (n*m-(i*n+n))*[0]
        row = np.array(row)
        A_p_type[i,:] = row

    A_q_type = np.zeros([n,m*n])
    b_q_type = q.reshape(n,1)

    for j in range(n):
        row_pattern = np.zeros([1,n])
        row_pattern[0,j] = 1
        row = np.tile(row_pattern,m)
        A_q_type[j,:] = row

    A = np.concatenate((A_p_type,A_q_type), axis = 0)
    b = np.concatenate((b_p_type,b_q_type), axis = 0)

    return A, b

def project_mu(mu,A,b,P,product_mu):

    # Input: coupling-shaped matrix mu; equality constraints A,b from gw_equality_constraints
    #        function; product coupling of some probability measures p and q.
    #        P is a projection matrix onto row space of A.
    # Output: Orthogonal projection of mu onto the affine subspace determined by A,b.

    m = product_mu.shape[0]
    n = product_mu.shape[1]

    # Create the vector to actually project and reshape
    vec_to_project = mu - product_mu
    vec_to_project = vec_to_project.reshape(m*n,)

    # Project it
    vec_to_project = vec_to_project - np.matmul(P,vec_to_project)

    projected_mu = product_mu.reshape(m*n,) + vec_to_project

    projected_mu = projected_mu.reshape(m,n)

    return projected_mu

def markov_hit_and_run_step(A,b,P,p,q,mu_initial):
    # Input: equality constraints A,b from gw_equality_constraints; pair of
    #       probability vectors p, q; initialization
    #        P is a projection matrix onto row space of A.
    # Output: new coupling measure after a hit-and-run step.

    m = p.shape[0]
    n = q.shape[0]

    product_mu = p[:,None]*q[None,:]

    
    mu_initial = project_mu(mu_initial,A,b,P,product_mu)
    # Project to the affine subspace
    # We assume mu_initial already lives there, but this will help with accumulation of numerical error

    mu_initial = mu_initial.reshape(m*n,)

    # Choose a random direction
    direction = np.random.normal(size = m*n)

    # Project to subspace of admissible directions

    direction = direction - np.matmul(P,direction)

    # Renormalize

    direction = direction/np.linalg.norm(direction)

    # Determine how far to move while staying in the polytope - These are inequality bounds,
    # so we just need the entries to stay positive

    pos = direction > 1e-6
    neg = direction < -1e-6

    direction_pos = direction[pos]
    direction_neg = direction[neg]
    mu_initial_pos = mu_initial[pos]
    mu_initial_neg = mu_initial[neg]

    lower = np.max(-mu_initial_pos/direction_pos)
    upper = np.min(-mu_initial_neg/direction_neg)

    # Choose a random distance to move
    r = (upper - lower)*np.random.uniform() + lower

    mu_new = mu_initial + r*direction
    mu_new = mu_new.reshape(m,n)

    return mu_new

def coupling_ensemble(A,b,p,q,num_samples,num_skips,seed=0):
    # Inputs: equality constraints A,b; probability vectors p,q; number of steps
    #         to take in the Markov chain; initialization
    # Output: Ensemble of couplings from the probability simplex.
    np.random.seed(seed)
    mu_initial = p[:,None]*q[None,:]

    # Find orthonormal basis for row space of A
    Q = linalg.orth(A.T)
    # Create projector onto the row space of A
    P = np.matmul(Q,Q.T)

    num_steps = num_samples*num_skips

    Markov_steps = []

    for j in range(num_steps):
        mu_new = markov_hit_and_run_step(A,b,P,p,q,mu_initial)
        mu_initial = mu_new
        if j%num_skips == 0:
            Markov_steps.append(mu_new)

    return Markov_steps


def coupling_ensemble_timed(A,b,p,q,num_samples,num_skips,timer=False):
    # Inputs: equality constraints A,b; probability vectors p,q; number of steps
    #         to take in the Markov chain; initialization
    # Output: Ensemble of couplings from the probability simplex.

    mu_initial = p[:,None]*q[None,:]

    # Find orthonormal basis for row space of A
    Q = linalg.orth(A.T)
    # Create projector onto the row space of A
    P = np.matmul(Q,Q.T)

    num_steps = num_samples*num_skips

    Markov_steps = []
    if timer:
        for j in tqdm(range(num_steps)):
            mu_new = markov_hit_and_run_step(A,b,P,p,q,mu_initial)
            mu_initial = mu_new
            if j%num_skips == 0:
                Markov_steps.append(mu_new)
    else:
        for j in range(num_steps):
            mu_new = markov_hit_and_run_step(A,b,P,p,q,mu_initial)
            mu_initial = mu_new
            if j%num_skips == 0:
                Markov_steps.append(mu_new)
    return Markov_steps
