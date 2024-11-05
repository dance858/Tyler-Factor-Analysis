import numpy as np 
import numpy.linalg as LA 
from estimators.Tyler_utils import grad_term1

def flattened_grad_subprob(params, n, k, grad_p_e_0, grad_p_G_0):
    e = params[:n]
    G = params[n:].reshape((n, k))
    grad_term1_e, grad_term1_G = grad_term1(G, e, k)
    grad_e = grad_term1_e + grad_p_e_0 
    grad_G = grad_term1_G + grad_p_G_0
    return np.concatenate([grad_e.flatten(), grad_G.flatten()])

def objective(params, n, k, grad_p_e_0, grad_p_G_0):
    e = params[:n]
    G = params[n:].reshape((n, k))
    
    if np.min(e) < 0:
       return np.inf 
    
    Q = np.eye(k) - (G.T * (1 / e)) @ G
    try:
        L = LA.cholesky(Q)
    except:
        return np.inf
    
    obj = (-np.sum(np.log(e)) - 2 * np.sum(np.log(np.diag(L))) + grad_p_e_0 @ e
            + np.trace(grad_p_G_0.T @ G))
    return obj

def line_search(x, dx, n, k, grad_p_e_0, grad_p_G_0, alpha=1, c1=1e-4, c2=0.9):
    """
    Line search to satisfy Wolfe conditions.
    
    Args:
        f: Function to minimize.
        grad_f: Gradient of the function.
        x: Current point.
        p: Search direction.
        alpha: Initial step size.
        c1, c2: Wolfe condition constants.

    Returns:
        alpha: Step size satisfying Wolfe conditions.
    """
    def phi(alpha):
        return objective(x + alpha * dx, n, k, grad_p_e_0, grad_p_G_0)
    
    def phi_prime(alpha):
        return np.dot(flattened_grad_subprob(x + alpha * dx, n, k, grad_p_e_0, grad_p_G_0), dx)

    #print("checkpoint 1")
    while phi(alpha) == np.inf:
        alpha *= 0.5
        assert(alpha > 1e-10)
    
    alpha_lo = 0
    alpha_hi = alpha
    while True:
        
        # Check Wolfe condition 1 (Armijo condition)
        if phi(alpha) > phi(0) + c1 * alpha * phi_prime(0):
            alpha_hi = alpha
        else:
            # Check Wolfe condition 2 (curvature condition)
            if phi_prime(alpha) >= c2 * phi_prime(0):
                return alpha
            if phi_prime(alpha) <= 0:
                alpha_lo = alpha
            else:
                return alpha

        # temporary fix - must look into this 
        

        # Update alpha using bisection method
        alpha = (alpha_lo + alpha_hi) / 2

        if (alpha_hi - alpha_lo < 1e-5):
            #assert(False)
            return alpha

        if (alpha < 1e-10):
            assert(False)
    print("checkpoint 4")


def subprob_via_BFGS(grad_p_e_0, grad_p_G_0, e0, G0, max_iter=500, H0 = None):
    n, k = G0.shape
    dim = n + n * k 
    x = np.concatenate([e0.flatten(), G0.flatten()])
 
    if H0 is None:
        H = np.eye(dim)    
    else:
        H = H0
    g_new = flattened_grad_subprob(x, n, k, grad_p_e_0, grad_p_G_0)

    for i in range(max_iter):
        e = x[:n]  
        G = x[n:].reshape((n, k))
        g = g_new 

        if LA.norm(g) < 1e-6:
            break
        
        dx = - H @ g 
        dir_der = dx @ g
        assert(dir_der < 0)
        
        alpha = line_search(x, dx, n, k, grad_p_e_0, grad_p_G_0) 
        
        x_new = x + alpha * dx 
        g_new = flattened_grad_subprob(x_new, n, k, grad_p_e_0, grad_p_G_0)
        #if i % 10 == 0:
        #     print("i / alpha / ||g_new|| / f(x_new) : ", i, alpha, LA.norm(g), objective(x_new, n, k, grad_p_e_0, grad_p_G_0))
        s = x_new - x 
        y = g_new - g 

        x = x_new 
        rho = 1 / (y @ s)

        # rho < 0 should never happen with a proper line search
        #if rho < 0:
        #    continue
        assert(rho > 0)
        I = np.eye(dim)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        

    e = x[:n]
    G = x[n:].reshape((n, k))
    return G, e, i, H