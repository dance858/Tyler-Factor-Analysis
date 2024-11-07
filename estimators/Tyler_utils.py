import numpy as np
import numpy.linalg as LA
from estimators.estimators_utils import PCFA_via_corr


def Tyler_FA_init(X, k):
    # compute low rank decomposition S ~ F @ F.T + D of sample covariance S
    mu = np.mean(X, axis=1, keepdims=True)
    zero_mean_X = X - mu 
    S = (1 / X.shape[1]) * (zero_mean_X @ zero_mean_X.T)
    
    F, d = PCFA_via_corr(S, k)

    # transform to low rank decomposition inv(S) ~ E - G @ G.T
    e0 = 1 / d
    evals, evecs = LA.eigh(np.eye(k) + (F.T * e0) @ F) 
    G0 = (F.T * e0).T @ (evecs * np.sqrt(1/evals)) @ evecs.T 

    return G0, e0

def grad_term1(G, e, k):
    temp1 = LA.inv(np.eye(k) - (G.T * (1 / e)) @ G)
    temp2 = ((1 / e) * G.T).T # np.diag(1/e) @ G
    grad_term1_e = -np.diag(temp2 @ temp1 @ temp2.T) - 1 / e
    grad_term1_G = 2 * temp2 @ temp1
    return grad_term1_e, grad_term1_G

def grad_term2(G, e, X, X2, n, m):
    GTX = G.T @ X  
    a =  1 / (e @ X2 - np.power(LA.norm(GTX, axis=0), 2))
    grad_p_e = (n / m) * np.sum(a * X2, axis=1)
    grad_p_G = (-2*n/m) * (a * X) @ GTX.T
    return grad_p_e, grad_p_G


def objective(G, e, X, X2, n, m, k):
    GTX = G.T @ X  
    Q = np.eye(k) - (G.T * (1 / e)) @ G
    a =  1 / (e @ X2 - np.power(LA.norm(GTX, axis=0), 2))

    # check if it belongs to domain
    if np.min(a) < 0 or np.min(e) < 0:
        return np.inf 
    
    Q = np.eye(k) - (G.T * (1 / e)) @ G
    try:
        L = LA.cholesky(Q)
    except:
        return np.inf
    
    p_val = -(n / m) * np.sum(np.log(a))
    term1 = -np.sum(np.log(e)) - 2 * np.sum(np.log(np.diag(L)))
    obj = term1 + p_val 
    return obj

# fixed-point iteration for Tyler's estimator (no factor model structure)
def Tyler_FP(X, max_iter):
    n, m = X.shape
    Sigma = np.eye(n)
    
    for _ in range(1, max_iter + 1):
        SigmainvX = LA.solve(Sigma, X)
        a = 1 / np.sum(X * SigmainvX, axis=0) 
        Sigma = (a * X) @ X.T
        Sigma /= np.trace(Sigma)
        
    return Sigma