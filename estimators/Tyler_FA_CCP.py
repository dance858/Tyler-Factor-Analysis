import numpy as np 
from estimators.Tyler_utils import Tyler_FA_init, grad_term1, grad_term2, objective
from estimators.CCP_QN import subprob_via_BFGS
import numpy.linalg as LA 
import cvxpy as cp
import pdb

def flattened_grad(params, n, k, grad_p_e_0, grad_p_G_0):
    e = params[:n]
    G = params[n:].reshape((n, k))
    grad_term1_e, grad_term1_G = grad_term1(G, e, k)
    grad_e = grad_term1_e + grad_p_e_0 
    grad_G = grad_term1_G + grad_p_G_0
    return np.concatenate([grad_e.flatten(), grad_G.flatten()])

def Tyler_FA_via_CCP(X, k, max_iter, DCP=False, solver=cp.MOSEK):  
    G0, e0 = Tyler_FA_init(X, k) 
    G, e, stats = Tyler_FA_via_CCP_(X, max_iter, G0, e0, DCP=DCP, solver=solver)
    return G, e,  stats

def diag_hess_approx(x, n, k, grad_p_e_0, grad_p_G_0, epsilon=1e-8):
    # Number of parameters
    num_params = len(x)
    H = np.zeros((num_params, ))  # Initialize diagonal of Hessian matrix

    # Define a lambda function that computes the gradient using flattened_grad
    grad_f = lambda p: flattened_grad(p, n, k, grad_p_e_0, grad_p_G_0)
    
    # Compute Hessian using finite differences on the gradient
    for i in range(num_params):
        # Define a function for the i-th partial derivative
        def partial_derivative_i(p):
            return grad_f(p)[i]
        
        x[i] += epsilon
        f_plus = partial_derivative_i(x)
        x[i] -= 2*epsilon 
        f_minus = partial_derivative_i(x)
        x[i] += epsilon
        
        H[i] = (f_plus - f_minus) / (2 * epsilon)
    
    return H

def Tyler_FA_via_CCP_(X, max_iter, G0, e0, DCP=True, solver=cp.MOSEK):
    n, m = X.shape
    k = G0.shape[1]
    X2 = np.power(X, 2)
    if m <= n:
        raise ValueError("Tyler's estimator requires m > n")
    G, e = G0, e0 
    H = None

    all_QN_iter = []
    all_objs = []
   
    for ii in range(1, max_iter + 1):
        # compute gradients of concave term 
        grad_p_e, grad_p_G = grad_term2(G, e, X, X2, n, m)

        # compute gradient of convex term to track the progress
        grad_term1_e, grad_term1_G = grad_term1(G, e, k)
        grad_e = grad_p_e + grad_term1_e
        grad_G = grad_p_G + grad_term1_G

        norm_grad_e = LA.norm(grad_e)
        norm_grad_G = LA.norm(grad_G.flatten())

        if norm_grad_e < 1e-5 and norm_grad_G < 1e-4:
            break

        obj = objective(G, e, X, X2, n, m, k)
        all_objs.append(obj)
        print("obj / norm(grad_e) / norm(grad_G): ", obj, norm_grad_e, norm_grad_G)

        # compute next iterate 
        if DCP:
            G_new, e_new = subprob_via_DCP(grad_p_e, grad_p_G, solver)
        else:
            if ii == 0:
                params = np.concatenate([e.flatten(), G.flatten()])
                H = np.diag(1/diag_hess_approx(params, n, k, grad_p_e, grad_p_G))
            G_new, e_new, iters, H = subprob_via_BFGS(grad_p_e, grad_p_G, e, G, H0=H)
            all_QN_iter.append(iters)
        # line search
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
            new_obj = objective(G_cand, e_cand, X, X2, n, m, k)
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
            
        
    stats = {'objs': all_objs, 'iterations': ii}
    
    return G, e, stats

def subprob_via_DCP(grad_p_e, grad_p_G, solver=cp.MOSEK):
    n, k = grad_p_G.shape

    e = cp.Variable(n)
    G = cp.Variable((n, k))
    block_matrix = cp.bmat([[cp.diag(e), G], [G.T, np.eye(k)]]) 
    objective = cp.Minimize(-cp.log_det(block_matrix) + grad_p_e @ e + 
                             cp.trace(grad_p_G.T @ G))
    problem = cp.Problem(objective)
    problem.solve(verbose=False, solver = solver)

    return G.value, e.value



