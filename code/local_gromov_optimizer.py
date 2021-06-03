
import numpy as np

from ot.utils import dist
from ot.optim import cg

#%%

# =============================================================================
# Got Back the code from GW and FGW distances of Python Optimal Transport Library
# To be able to give different initializations to Conditional Gradient algorithms
# https://pythonot.github.io/
# =============================================================================

def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    """Return loss matrices and tensors for Gromov-Wasserstein fast computation

    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in [12]

    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces

    The square-loss function L(a,b)=|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)
            * f2(b)=(b^2)
            * h1(a)=a
            * h2(b)=2*b

    The kl-loss function L(a,b)=a*log(a/b)-a+b is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=a*log(a)-a
            * f2(b)=b
            * h1(a)=a
            * h2(b)=log(b)

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
        Coupling between source and target spaces
    p : ndarray, shape (ns,)

    Returns
    -------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):
    """Return the tensor for Gromov-Wasserstein fast computation

    The tensor is computed as described in Proposition 1 Eq. (6) in [12].

    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)

    Returns
    -------
    tens : ndarray, shape (ns, nt)
        \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    A = -np.dot(hC1, T).dot(hC2.T)
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T):
    """Return the Loss for Gromov-Wasserstein

    The loss is computed as described in Proposition 1 Eq. (6) in [12].

    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
        Current value of transport matrix T

    Returns
    -------
    loss : float
        Gromov Wasserstein loss

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    tens = tensor_product(constC, hC1, hC2, T)

    return np.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):
    """Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in [12].

    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
        Current value of transport matrix T

    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    return 2 * tensor_product(constC, hC1, hC2,
                              T)  # [12] Prop. 2 misses a 2 factor


def update_square_loss(p, lambdas, T, Cs):
    """
    Updates C according to the L2 Loss kernel with the S Ts couplings
    calculated at each iteration

    Parameters
    ----------
    p : ndarray, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the S spaces' weights.
    T : list of S np.ndarray of shape (ns,N)
        The S Ts couplings calculated at each iteration.
    Cs : list of S ndarray, shape(ns,ns)
        Metric cost matrices.

    Returns
    ----------
    C : ndarray, shape (nt, nt)
        Updated C matrix.
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_kl_loss(p, lambdas, T, Cs):
    """
    Updates C according to the KL Loss kernel with the S Ts couplings calculated at each iteration


    Parameters
    ----------
    p  : ndarray, shape (N,)
        Weights in the targeted barycenter.
    lambdas : list of the S spaces' weights
    T : list of S np.ndarray of shape (ns,N)
        The S Ts couplings calculated at each iteration.
    Cs : list of S ndarray, shape(ns,ns)
        Metric cost matrices.

    Returns
    ----------
    C : ndarray, shape (ns,ns)
        updated C matrix
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.exp(np.divide(tmpsum, ppt))


def gromov_wasserstein(C1, C2, p, q, loss_fun, G0, log=False, armijo=False, **kwargs):
    """
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        GW = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}

    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices
    - H  : entropy

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'

    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T : ndarray, shape (ns, nt)
        Doupling between the two spaces that minimizes:
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    log : dict
        Convergence information and loss.

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.

    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    if G0 is None:
        G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    if log:
        res, log = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
        log['gw_dist'] = gwloss(constC, hC1, hC2, res)
        return res, log
    else:
        return cg(p, q, 0, 1, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)


def fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun='square_loss', G0= None,alpha=0.5, armijo=False, log=False, **kwargs):
    """
    Computes the FGW transport between two graphs see [24]

    .. math::
        \gamma = arg\min_\gamma (1-\\alpha)*<\gamma,M>_F + \\alpha* \sum_{i,j,k,l}
        L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}

        s.t. \gamma 1 = p
             \gamma^T 1= q
             \gamma\geq 0

    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)
    - L is a loss function to account for the misfit between the similarity matrices

    The algorithm used for solving the problem is conditional gradient as discussed in  [24]_

    Parameters
    ----------
    M : ndarray, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    loss_fun : str, optional
        Loss function used for the solver
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    gamma : ndarray, shape (ns, nt)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.

    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.
    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    if G0 is None:
        G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    if log:
        res, log = cg(p, q, M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)
        log['fgw_dist'] = log['loss'][::-1][0]
        return res, log
    else:
        return cg(p, q, M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)


def fused_gromov_wasserstein2(M, C1, C2, p, q, loss_fun='square_loss', alpha=0.5, armijo=False, log=False, **kwargs):
    """
    Computes the FGW distance between two graphs see [24]

    .. math::
        \min_\gamma (1-\\alpha)*<\gamma,M>_F + \\alpha* \sum_{i,j,k,l}
        L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}


        s.t. \gamma 1 = p
             \gamma^T 1= q
             \gamma\geq 0

    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)
    - L is a loss function to account for the misfit between the similarity matrices
    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_

    Parameters
    ----------
    M : ndarray, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix respresentative of the structure in the source space.
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix espresentative of the structure in the target space.
    p :  ndarray, shape (ns,)
        Distribution in the source space.
    q :  ndarray, shape (nt,)
        Distribution in the target space.
    loss_fun : str, optional
        Loss function used for the solver.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research.
        Else closed form is used. If there is convergence issues use False.
    **kwargs : dict
        Parameters can be directly pased to the ot.optim.cg solver.

    Returns
    -------
    gamma : ndarray, shape (ns, nt)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.

    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    res, log = cg(p, q, M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)
    if log:
        log['fgw_dist'] = log['loss'][::-1][0]
        log['T'] = res
        return log['fgw_dist'], log
    else:
        return log['fgw_dist']


def gromov_wasserstein2(C1, C2, p, q, loss_fun, log=False, armijo=False, **kwargs):
    """
    Returns the gromov-wasserstein discrepancy between (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        GW = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}

    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices
    - H  : entropy

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space.
    q :  ndarray, shape (nt,)
        Distribution in the target space.
    loss_fun :  str
        loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.

    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    log : dict
        convergence information and Coupling marix

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.

    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)
    res, log = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
    log['gw_dist'] = gwloss(constC, hC1, hC2, res)
    log['T'] = res
    if log:
        return log['gw_dist'], log
    else:
        return log['gw_dist']



def gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun,
                       max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None):
    """
    Returns the gromov-wasserstein barycenters of S measured similarity matrices

    (Cs)_{s=1}^{s=S}

    The function solves the following optimization problem with block
    coordinate descent:

    .. math::
        C = argmin_C\in R^NxN \sum_s \lambda_s GW(C,Cs,p,ps)

    Where :

    - Cs : metric cost matrix
    - ps  : distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S np.ndarray of shape (ns, ns)
        Metric cost matrices
    ps : list of S np.ndarray of shape (ns,)
        Sample weights in the S spaces
    p : ndarray, shape (N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the S spaces' weights
    loss_fun :  tensor-matrix multiplication function based on specific loss function
    update : function(p,lambdas,T,Cs) that updates C according to a specific Kernel
             with the S Ts couplings calculated at each iteration
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0).
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | ndarray, shape(N,N)
        Random initial value for the C matrix provided by user.

    Returns
    -------
    C : ndarray, shape (N, N)
        Similarity matrix in the barycenter space (permutated arbitrarily)

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    S = len(Cs)

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        # XXX : should use a random state and not use the global seed
        xalea = np.random.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
    else:
        C = init_C

    cpt = 0
    err = 1

    error = []

    while(err > tol and cpt < max_iter):
        Cprev = C

        T = [gromov_wasserstein(Cs[s], C, ps[s], p, loss_fun,
                                numItermax=max_iter, stopThr=1e-5, verbose=verbose, log=log) for s in range(S)]
        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(C - Cprev)
            error.append(err)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    return C

def update_sructure_matrix(p, lambdas, T, Cs):
    """Updates C according to the L2 Loss kernel with the S Ts couplings.

    It is calculated at each iteration

    Parameters
    ----------
    p : ndarray, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the S spaces' weights.
    T : list of S ndarray of shape (ns, N)
        The S Ts couplings calculated at each iteration.
    Cs : list of S ndarray, shape (ns, ns)
         Metric cost matrices.

    Returns
    -------
    C : ndarray, shape (nt, nt)
        Updated C matrix.
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s]) for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_feature_matrix(lambdas, Ys, Ts, p):
    """Updates the feature with respect to the S Ts couplings.


    See "Solving the barycenter problem with Block Coordinate Descent (BCD)"
    in [24] calculated at each iteration

    Parameters
    ----------
    p : ndarray, shape (N,)
        masses in the targeted barycenter
    lambdas : list of float
        List of the S spaces' weights
    Ts : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Ys : list of S ndarray, shape(d,ns)
        The features.

    Returns
    -------
    X : ndarray, shape (d, N)

    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    p = np.array(1. / p).reshape(-1,)

    tmpsum = sum([lambdas[s] * np.dot(Ys[s], Ts[s].T) * p[None, :] for s in range(len(Ts))])

    return tmpsum
