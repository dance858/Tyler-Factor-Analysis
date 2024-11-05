import numpy as np
import numpy.linalg as LA
from scipy.stats import multivariate_normal, multivariate_t

def generate_samples(N, distribution, params):
    """
    Generate N samples from the specified distribution.
    
    Parameters:
    N (int): Number of samples
    distribution (str): 'N' for Gaussian, 'T' for multivariate T distribution
    params (dict): Dictionary containing relevant distribution parameters.
                   For 'N': {'mu': mean_vector, 'cov': covariance_matrix}
                   For 'T': {'mu': mean_vector, 'cov': covariance_matrix, 'v': degrees_of_freedom}
    
    Returns:
    samples (np.ndarray): N generated samples
    """

    mu = params.get('mu')
    cov = params.get('cov')


    if distribution == 'N':
        if mu is None or cov is None:
            raise ValueError("Parameters 'mu' and 'cov' must be provided for Gaussian distribution.")
        samples = multivariate_normal.rvs(mean=mu, cov=cov, size=N)
        
    elif distribution == 'T':
        v = params.get('v')
        if mu is None or cov is None or v is None:
            raise ValueError("Parameters 'mu', 'cov', and 'v' must be provided for multivariate T distribution.")
        if v <= 2:
            raise ValueError("Only v > 2 is allowed")
        # DC: I think this is how to compute the scatter matrix from the
        # covariance. It seems reasonable empirically. But check wikipedia. 
        samples = multivariate_t.rvs(loc=mu, shape=cov * (v-2)/v, df=v, size=N)
    else:
        raise ValueError("Invalid distribution type. Use 'N' for Gaussian or 'T' for multivariate T.")
    
    return samples.T


def scatter_to_corr(scatters):
    corr_matrices = []
    for scatter in scatters:
        d = np.diag(scatter)
        corr_matrix = (np.sqrt(1.0 / d)[:, None] * scatter) * np.sqrt(1.0 / d)
        corr_matrices.append(corr_matrix)
    return corr_matrices

def errors_corr(scatters, true_corr):
    corr_matrices = scatter_to_corr(scatters)
    errors = []
    for corr in corr_matrices:
        errors.append(LA.norm(corr - true_corr, 'fro')/LA.norm(true_corr, 'fro'))

    return errors