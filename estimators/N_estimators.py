import numpy as np
import numpy.linalg as LA
import sys 
from estimators.estimators_utils import (GFA_params, PCFA_via_corr, TERMINATE_MAX_ITER, 
                              TERMINATE_KKT, TERMINATE_OBJ, Solver_stats, 
                              PCFA_via_corr_complex)

# X is n x m (n is the dimension, m is the number of samples) 
def sample_mean_and_cov(X):
    mu = np.mean(X, axis=1, keepdims=True)
    zero_mean_X = X - mu 
    cov = (1 / X.shape[1]) * (zero_mean_X @ zero_mean_X.T)
    return mu, cov

def sample_mean_and_cov_complex(X):
    mu = np.mean(X, axis=1, keepdims=True)
    zero_mean_X = X - mu 
    cov = (1 / X.shape[1]) * (zero_mean_X @ zero_mean_X.conj().T)
    return mu, cov

# EM-algorithm described in Johansson 2023, Section 8.
# It is identical to the EM-algorithm by Rudin 1982.
# EM-algorithm described in Johansson 2023, Section 8.
# It is identical to the EM-algorithm by Rudin 1982.
def GFA_EM(S, rank, F=None, d=None, params: GFA_params = GFA_params()):
    if params.verbose:
        print("Solving problem with EM ...")
    
    # initialization
    if F is None or d is None:
        F, d = PCFA_via_corr(S, rank)

    KKTs = [KKT_residual_GFA(F, d, S, params.eta)]
    objs = [obj_GFA(F, d, S)]

    for ii in range(1, params.max_iter + 1):
        # E-step
        G = np.linalg.inv((F.T * (1 / d.reshape(1, -1))) @ F + np.eye(rank))                        # denoted by H in your appendxix
        B = G @ F.T * (1 / d.reshape(1, -1))                                                        # H FT inv(D)         A3 transpose
        Cxz = S @ B.T                                                                               # S inv(D) F H
        Czz = B @ Cxz + G                                                                           # H FT inv(D) S inv(D) F H

        # M-step
        F = Cxz @ np.linalg.inv(Czz)
        d = np.maximum(np.diag(S - 2 * Cxz @ F.T + F @ Czz @ F.T), params.eta)

        status, terminate = check_termination(params, ii, F, d, S, objs, KKTs)
        if terminate:
            break

    if params.verbose:
        print("EM finished! KKT residual: ", KKTs[-1])

    stats = Solver_stats(iterations=ii, objs=objs, KKTs=KKTs, status=status,
                         subp_iterations=[])
    return F, d, stats


def obj_GFA(F, d, S):
    L = LA.cholesky(np.eye(F.shape[1]) + F.T @ (F / d.reshape(-1, 1)))
    term1 = np.sum(np.log(d)) + 2*np.sum(np.log(np.diag(L)))
    term2 = np.trace(LA.solve(F @ F.T + np.diag(d), S))
    return term1 + term2

def grad_GFA(F, d, S):
    # gradients of logdet(F @ F.T + D)
    I = np.eye(F.shape[1])
    dInv = 1 / d
    scatter_inv = (np.diag(dInv) - ((dInv)[:, None] * F) @
                   LA.solve(I + (F.T * (dInv.reshape(1, -1))) @  F,
                            F.T * (dInv.reshape(1, -1))))
  
    grad_1_D = scatter_inv
    grad_1_F = 2 * grad_1_D @ F 
    grad_1_d = np.diag(grad_1_D)

    # gradients of trace(inv(F @ F.T + D) @ S)
    grad_2_D = - scatter_inv @ S @ scatter_inv
    grad_2_F = 2 * grad_2_D @ F 
    grad_2_d = np.diag(grad_2_D)

    grad_d = grad_1_d + grad_2_d 
    grad_F = grad_1_F + grad_2_F
    return grad_F, grad_d


def grad_norm_GFA(F, d, S):
    grad_F, grad_d = grad_GFA(F, d, S)
    grad_norm = np.sqrt(LA.norm(grad_d)**2 + LA.norm(grad_F, 'fro')**2)
    return grad_norm

def KKT_residual_GFA(F, d, S, eta):
    grad_F, grad_d = grad_GFA(F, d, S)
    KKT_res = np.sqrt(LA.norm(grad_F, 'fro')**2 + LA.norm(grad_d * (d-eta))**2
                    + LA.norm(np.minimum(grad_d, 0))**2)
    return KKT_res

# only appends the objective value and KKT residual every "params_term_interval" 
# iteration
def check_termination(params: GFA_params, ii, F, d, S, objs, KKTs):
    terminate = False
    status = TERMINATE_MAX_ITER
    if params.track_obj or (ii % params.term_interval == 0):
        objs.append(obj_GFA(F, d, S))
    
    if params.track_KKT or (ii % params.term_interval == 0):
        KKTs.append(KKT_residual_GFA(F, d, S, params.eta))

    if (ii % params.term_interval == 0):
        new_obj = objs[-1]
        old_obj = objs[-2]
        new_KKT = KKTs[-1]
        old_KKT = KKTs[-2]

        if (abs(new_obj - old_obj) / abs(old_obj) < params.eps_rel_obj):
            status = TERMINATE_OBJ
            terminate = True
            
        if (abs(new_KKT - old_KKT) / abs(old_KKT) < params.eps_rel_KKT):
            status = TERMINATE_KKT
            terminate = True
    
    return status, terminate

def Gaussian_FA_via_EM_complex(S, k, iter):

    # initialization
    F, d = PCFA_via_corr(S, k)

    for _ in range(iter):
        # E-step
        G = np.linalg.inv((F.conj().T * (1 / d.reshape(1, -1))) @ F + np.eye(k))
        B = G @ F.conj().T * (1 / d.reshape(1, -1))
        Cxz = S @ B.conj().T
        Czz = B @ Cxz + G

        # M-step
        F = Cxz @ np.linalg.inv(Czz)
        d = np.diag(S - 2 * Cxz @ F.conj().T + F @ Czz @ F.conj().T)
        
    return F, d