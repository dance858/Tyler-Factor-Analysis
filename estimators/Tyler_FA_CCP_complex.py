import numpy as np 
from estimators.estimators_utils import PCFA_via_corr_complex
import numpy.linalg as LA 
import cvxpy as cp
import pdb

def Tyler_FA_init_complex(X, k):
    # compute low rank decomposition S ~ F @ F.T + D of sample covariance S
    mu = np.mean(X, axis=1, keepdims=True)
    zero_mean_X = X - mu 
    S = (1 / X.shape[1]) * (zero_mean_X @ zero_mean_X.conj().T)
    
    F, d = PCFA_via_corr_complex(S, k)

    # transform to low rank decomposition inv(S) ~ E - G @ G.T
    e0 = 1 / d
    evals, evecs = LA.eigh(np.eye(k) + (F.conj().T * e0) @ F) 
    G0 = (F.conj().T * e0).conj().T @ (evecs * np.sqrt(1/evals)) @ evecs.conj().T 

    return G0, e0

def grad_term1_complex(G, e, k):
    temp1 = LA.inv(np.eye(k) - (G.conj().T * (1 / e)) @ G)
    temp2 = ((1 / e) * G.conj().T).conj().T 
    grad_term1_e = -np.diag(temp2 @ temp1 @ temp2.conj().T) - 1 / e
    grad_term1_G = 2 * temp2 @ temp1
    return grad_term1_e, grad_term1_G

def grad_term2_complex(G, e, X, X2, n, m):
    GHX = G.conj().T @ X  
    a =  1 / (e @ X2 - np.power(LA.norm(GHX, axis=0), 2))
    #pdb.set_trace()
    grad_p_e = (n / m) * np.sum(a * X2, axis=1)
    grad_p_G = (-2*n/m) * (a * X) @ GHX.conj().T        # should it be conj here or not?
    return grad_p_e, grad_p_G


def objective_complex(G, e, X, X2, n, m, k):
    GTX = G.conj().T @ X  
    Q = np.eye(k) - (G.conj().T * (1 / e)) @ G
    a =  1 / (e @ X2 - np.power(LA.norm(GTX, axis=0), 2))
    a = np.real(a)
    # check if it belongs to domain
    if np.min(a) < 0 or np.min(e) < 0:
        return np.inf 
    
    Q = np.eye(k) - (G.conj().T * (1 / e)) @ G
    try:
        L = LA.cholesky(Q)
    except:
        return np.inf
    
    p_val = -(n / m) * np.sum(np.log(a))
    term1 = -np.sum(np.log(e)) - 2 * np.sum(np.log(np.real(np.diag(L))))
    obj = term1 + p_val 
    return obj

def flattened_grad(params, n, k, grad_p_e_0, grad_p_G_0):
    e = params[:n]
    G = params[n:].reshape((n, k))
    grad_term1_e, grad_term1_G = grad_term1_complex(G, e, k)
    grad_e = grad_term1_e + grad_p_e_0 
    grad_G = grad_term1_G + grad_p_G_0
    return np.concatenate([grad_e.flatten(), grad_G.flatten()])

def Tyler_FA_via_CCP_complex(X, k, max_iter, DCP=True, G0 = None, e0 = None):  
    if G0 is None or e0 is None:
        G0, e0 = Tyler_FA_init_complex(X, k) 
    G, e, all_QN_iter, H = Tyler_FA_via_CCP__complex(X, max_iter, G0, e0, DCP)
    return G, e, all_QN_iter, H


def Tyler_FA_via_CCP__complex(X, max_iter, G0, e0, DCP=True):
    n, m = X.shape
    k = G0.shape[1]
    X2 = np.power(np.abs(X), 2)
    if m <= n:
        raise ValueError("Tyler's estimator requires m > n")
    G, e = G0, e0 
    H = None

    all_QN_iter = []
   
    for ii in range(max_iter):
        # compute gradients of concave term 
        grad_p_e, grad_p_G = grad_term2_complex(G, e, X, X2, n, m)

        # compute gradient of convex term to track the progress
        grad_term1_e, grad_term1_G = grad_term1_complex(G, e, k)
        grad_e = grad_p_e + grad_term1_e
        grad_G = grad_p_G + grad_term1_G

        norm_grad_e = LA.norm(grad_e)
        norm_grad_G = LA.norm(grad_G.flatten())

        if norm_grad_e < 1e-5 and norm_grad_G < 1e-4:
            break

        obj = objective_complex(G, e, X, X2, n, m, k)
        #print("obj / norm(grad_e) / norm(grad_G): ", obj, norm_grad_e, norm_grad_G)

        # compute next iterate 
        G_new, e_new = subprob_via_DCP(grad_p_e, grad_p_G)
        
        line_search = True 
        dG = G_new - G 
        de = e_new - e
        alpha = 1.1
        t = 1.0
        min_obj = obj
        min_t = t
        while line_search:
            G_cand = G + alpha * t * dG 
            e_cand = e + alpha * t * de 
            new_obj = objective_complex(G_cand, e_cand, X, X2, n, m, k)
            if new_obj > obj:
                line_search = False 
                break 
            
            if new_obj < min_obj:
                min_obj = new_obj 
                min_t = t
            
            
            t = alpha * t 
            
        t = min_t
        G = G + t * dG 
        e = e + t * de 
            
    #print("iterations: ", ii)
        
    return G, e, all_QN_iter, H

def subprob_via_DCP(grad_p_e, grad_p_G):
    n, k = grad_p_G.shape
    e = cp.Variable(n)
    G = cp.Variable((n, k), complex=True)
    block_matrix = cp.bmat([[cp.diag(e), G], [G.H, np.eye(k)]]) 
    objective = cp.Minimize(-cp.log_det(block_matrix) + grad_p_e @ e + cp.real(cp.trace(G.H @ grad_p_G)))
    problem = cp.Problem(objective)
    problem.solve(verbose=False, solver = cp.MOSEK)
    return G.value, e.value


