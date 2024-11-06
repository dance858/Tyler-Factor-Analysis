import numpy as np
import numpy.linalg as LA
from sources import FarField1DSourcePlacement
import matplotlib.pyplot as plt
import pdb

def generate_ula_data_uniform(power_source, sig2, d, m, K, N, wavelength, theta_rad):
    s = np.sqrt(power_source / 2) * (np.random.randn(K, N) + 1j * np.random.randn(K, N))
    e = np.sqrt(sig2 / 2) * (np.random.randn(m, N) + 1j * np.random.randn(m, N))
    A = np.exp(-2j * np.pi / wavelength * (d * np.arange(m).reshape(-1, 1) * np.sin(theta_rad)))
    Y = A @ s + e
    true_cov = power_source * (A @ A.conj().T) + sig2 * np.eye(m) 
    return Y, true_cov

def generate_ula_data_nonuniform(power_source, noise_variances, d, m, K, N,
                                 wavelength, theta_rad, df = 3, 
                                 signal_distribution="N"):
    
    s = np.sqrt(power_source / 2) * (np.random.randn(K, N) + 1j * np.random.randn(K, N))  
    e = np.sqrt(1 / 2) * np.sqrt(np.diag(noise_variances)) @ (np.random.randn(m, N) + 1j * np.random.randn(m, N)) 
    A = np.exp(-2j * np.pi / wavelength * (d * np.arange(m).reshape(-1, 1) * np.sin(theta_rad)))
    Y = A @ s + e
    
    if signal_distribution == "T":
        chi2 =  np.random.chisquare(df, N)
        Y *= np.sqrt(df / chi2)

    true_cov = power_source * (A @ A.conj().T) + np.diag(noise_variances)
    return Y, true_cov, A


def ensure_n_resolvable_sources(k, max_k):
    """Checks if the number of expected sources exceeds the maximum resolvable sources."""
    if k > max_k:
        raise ValueError(
            'Too many sources. Maximum number of resolvable sources is {0}'
            .format(max_k)
        )

def get_noise_subspace(R, k):
    """
    Gets the noise eigenvectors.

    Args:
        R: Covariance matrix.
        k: Number of sources.
    """
    _, E = np.linalg.eigh(R)
    # Note: eigenvalues are sorted in ascending order.
    return E[:,:-k]


class RootMUSIC1D:
    """Creates a root-MUSIC estimator for uniform linear arrays.

    Args:
        wavelength (float): Wavelength of the carrier wave.

    References:
        [1] A. Barabell, "Improving the resolution performance of
        eigenstructure-based direction-finding algorithms," ICASSP '83. IEEE
        International Conference on Acoustics, Speech, and Signal Processing,
        Boston, Massachusetts, USA, 1983, pp. 336-339.

        [2] B. D. Rao and K. V. S. Hari, "Performance analysis of Root-Music,"
        IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37,
        no. 12, pp. 1939-1949, Dec. 1989.
    """

    def __init__(self, wavelength):
        self._wavelength = wavelength

    
    
    def estimate(self, R, k, d0=None, unit='rad'):
        """Estimates the direction-of-arrivals of 1D far-field sources.

        Args:
            R (~numpy.ndarray): Covariance matrix input. This covariance matrix
                must be obtained using a uniform linear array.
            k (int): Expected number of sources.
            d0 (float): Inter-element spacing of the uniform linear array used
                to obtain ``R``. If not specified, it will be set to one half
                of the ``wavelength`` used when creating this estimator.
                Default value is ``None``.
            unit (str): Unit of the estimates. Default value is ``'rad'``.
                See :class:`~doatools.model.sources.FarField1DSourcePlacement`
                for more details on valid units.
        
        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): ``True`` only if the rooting algorithm
              successfully finds ``k`` roots inside the unit circle. This flag
              does **not** guarantee that the estimated source locations are
              correct. The estimated source locations may be completely wrong!
              If resolved is False, ``estimates`` will be ``None``.
            * estimates (:class:`~doatools.model.sources.FarField1DSourcePlacement`):
              A :class:`~doatools.model.sources.FarField1DSourcePlacement`
              recording the estimated source locations. Will be ``None`` if
              resolved is ``False``.
        """
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError('R should be a square matrix.')
        m = R.shape[0]
        ensure_n_resolvable_sources(k, m - 1)
        if d0 is None:
            d0 = self._wavelength / 2.0
        En = get_noise_subspace(R, k)
        # Compute the coefficients for the polynomial.
        C = En @ En.T.conj()
        coeff = np.zeros((m - 1,), dtype=np.complex128)
        for i in range(1, m):
            coeff[i - 1] += np.sum(np.diag(C, i))
        coeff = np.hstack((coeff[::-1], np.sum(np.diag(C)), coeff.conj()))
        # Find the roots of the polynomial.
        z = np.roots(coeff)
        # Find k points inside the unit circle that are also closest to the unit
        # circle.
        nz = len(z)
        mask = np.ones((nz,), dtype=np.bool_)
        for i in range(nz):
            absz = abs(z[i])
            if absz > 1.0:
                # Outside the unit circle.
                mask[i] = False
            elif absz == 1.0:
                # On the unit circle. Need to find the closest point and remove
                # it.
                idx = -1
                dist = np.inf
                for j in range(nz):
                    if j != i and mask[j]:
                        cur_dist = abs(z[i] - z[j])
                        if cur_dist < dist:
                            dist = cur_dist
                            idx = j
                if idx < 0:
                    raise RuntimeError('Unpaired point found on the unit circle, which is impossible.')
                mask[idx] = False
        z = z[mask]
        sorted_indices = np.argsort(1.0 - np.abs(z))
        if len(z) < k:
            return False, None
        else:
            z = z[sorted_indices[:k]]
            return True, FarField1DSourcePlacement.from_z(z, self._wavelength, d0, unit)
        

def uniform_Gaussian_CRB(P, theta_rad, sig2, m, d, N):
    # Parameters:
    #    P      <- The covariance matrix of the source signals
    #    N      <- number of snapshots to generate
    #    sig2   <- noise variance
    #    m      <- number of sensors
    #    d      <- sensor spacing in wavelengths
    
    j = 1j  # Imaginary unit in Python
    A = np.exp(-2 * np.pi * j * d * np.arange(m).reshape(-1, 1) * np.sin(theta_rad))
    D = (-2 * np.pi * j * d * np.arange(m).reshape(-1, 1) * np.cos(theta_rad)) * np.exp(-2 * np.pi * j * d * np.arange(m).reshape(-1, 1) * np.sin(theta_rad))

    # Covariance matrix of array output
    R = A @ P @ A.conj().T + sig2 * np.eye(m)

    # Projection matrix for A orthogonal
    proj_A_orth = np.eye(m) - A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T

    # Compute CRB
    CRB = sig2 / (2 * N) * np.linalg.inv(np.real((D.conj().T @ proj_A_orth @ D) * np.transpose(P @ A.conj().T @ np.linalg.inv(R) @ A @ P)))

    return CRB


def non_uniform_Gaussian_CRB(P, theta_rad, noise_variances, m, d, N):
    # Parameters:
    #    P                  <- The covariance matrix of the source signals
    #    N                  <- number of snapshots to generate
    #    noise_variances    <- noise variance
    #    m                  <- number of sensors
    #    d                  <- sensor spacing in wavelengths
    
    j = 1j  # Imaginary unit in Python
    A = np.exp(-2 * np.pi * j * d * np.arange(m).reshape(-1, 1) * np.sin(theta_rad))
    D = (-2 * np.pi * j * d * np.arange(m).reshape(-1, 1) * np.cos(theta_rad)) * np.exp(-2 * np.pi * j * d * np.arange(m).reshape(-1, 1) * np.sin(theta_rad))

    # Covariance matrix of array output
    Q = np.diag(np.squeeze(noise_variances))
    R = A @ P @ A.conj().T + Q

    Q_inv_sqrt = LA.inv(np.sqrt(Q))

    A_tilde = Q_inv_sqrt @ A 
    D_tilde = Q_inv_sqrt @ D
    R_tilde = Q_inv_sqrt @ R @ Q_inv_sqrt

    # Projection matrix for A_tilde orthogonal
    proj_A_tilde = A_tilde @ np.linalg.inv(A_tilde.conj().T @ A_tilde) @ A_tilde.conj().T
    proj_A_tilde_orth = np.eye(m) - proj_A_tilde

    # compute M 
    temp1 = LA.inv(R_tilde) @ A_tilde @ P
    temp2 = D_tilde.conj().T @ proj_A_tilde_orth 
    M = 2 * np.real(temp1.T * temp2)
    
    # compute T 
    temp1 = LA.inv(R_tilde).conj() * LA.inv(R_tilde)
    temp2 = (proj_A_tilde @ LA.inv(R_tilde))
    T = LA.inv(temp1 - temp2.conj() * temp2)

    # Compute CRB
    temp1 = P @ A_tilde.conj().T @ LA.inv(R_tilde) @ A_tilde @ P 
    temp2 = (D_tilde.conj().T @ proj_A_tilde_orth @ LA.inv(R_tilde) @ D_tilde).T 
    CRB = (1 / N) * np.linalg.inv(2*np.real(temp1 * temp2) - M @ T @ M.T)

    return CRB

def steering_vector(theta, M, d):
    return np.exp(-1j * 2 * np.pi * d * np.arange(M) * np.sin(theta))

def plot_spectrum(R_list, labels, num_of_sources, true_angles, theta_scan = np.linspace(-10, 25, 4000)):
    # Eigenvalue decomposition
    
    counter = -1
    for R in R_list:
        counter += 1
        eigenvalues, eigenvectors = LA.eigh(R)
        num_of_sensors = R.shape[0]

        # Sort eigenvalues and get noise subspace
        idx = np.argsort(eigenvalues)[::-1]
        noise_subspace = eigenvectors[:, idx[num_of_sources:]]  # Eigenvectors corresponding to noise

        spectrum = np.zeros(len(theta_scan))
        for i, theta in enumerate(theta_scan):
            sv = steering_vector(np.deg2rad(theta), num_of_sensors, 0.5)
            spectrum[i] = 1 / np.abs(sv.conj().T @ noise_subspace @ noise_subspace.conj().T @ sv)

        #pdb.set_trace()
        plt.plot(theta_scan, spectrum, label=labels[counter])

    for angle in true_angles:
        plt.axvline(x=angle*180/np.pi, color='k', linestyle='--', alpha=0.3)

    plt.xlabel("Angle (degrees)", fontsize=14)
    plt.ylabel("Spectrum (dB)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("resolution.pdf")
    return 10 * np.log10(spectrum / np.max(spectrum))


def get_row_column(i):
    """
    Get the row and column indices corresponding to the index i 
    in the vectorized upper triangular form of a symmetric matrix.
    
    Parameters:
    - i: int, index in the vectorized form
    - n: int, size of the symmetric matrix
    
    Returns:
    - (row, col): tuple of (row index, column index)
    """
    # Calculate the col index
    c = int((-1 + np.sqrt(1 + 8 * i)) // 2)
    
    # Calculate the row index
    r = int(i - (c * (c + 1)) // 2)
    
    return r, c


def P_diff_i(i, k):
    assert(i < k**2)

    P = np.zeros((k, k), dtype=complex)

    # real parameters
    if i < k*(k+1)/2:
        r, c = get_row_column(i)
        P[r, c] = 1.0
        P[c, r] = 1.0
    # imaginary parameters
    else:
        r, c = get_row_column(i-k*(k+1)/2)
        c += 1
        P[r, c] = 1j
        P[c, r] = -1j
    return P 

def nonuniform_MVT_CRB(P, theta_rad, Q, n, m, d, v): 
    #pdb.set_trace()
    # Parameters:
    #    P      <- The covariance matrix of the source signals
    #    n      <- sensors
    #    Q      <- noise variance matrix
    #    m      <- samples
    #    d      <- sensor spacing in wavelengths
    k = P.shape[0]
    j = 1j  
    A = np.exp(-2 * np.pi * j * d * np.arange(n).reshape(-1, 1) * np.sin(theta_rad))
    D = (-2 * np.pi * j * d * np.arange(n).reshape(-1, 1) * np.cos(theta_rad)) * np.exp(-2 * np.pi * j * d * np.arange(n).reshape(-1, 1) * np.sin(theta_rad))
    Sigma = A @ P @ A.conj().T + Q

    # compute Fisher information matrix
    dim = k + n + k**2
    F = np.zeros((dim, dim))
    psi = (1*n+v)/(1*(n+1)+v)

    # just to verify implementation
    F_G = np.zeros((dim, dim))

    for row in range(dim):

        # with respect to theta_i
        if (row < k):
            i = row 
            Sigma_i = (D[:, i].reshape(-1, 1) @ P[i, :].reshape(1, -1) @ A.conj().T +
                        (D[:, i].reshape(-1, 1) @ P[i, :].reshape(1, -1) @ A.conj().T).conj().T)
        # with respect to q_i
        elif (row >= k and row < k + n):
            i = row - k
            e_i = np.zeros((n, ))
            e_i[i] = 1 
            Sigma_i = np.diag(e_i)
        # with respect to p_i
        else:
            i = row - k - n
            #pdb.set_trace()
            Sigma_i = A @ P_diff_i(i, k) @ A.conj().T

        for col in range(row, dim):        
            # with respect to theta_j
            if (col < k):
                j = col 
                Sigma_j = (D[:, j].reshape(-1, 1) @ P[j, :].reshape(1, -1) @ A.conj().T +
                          (D[:, j].reshape(-1, 1) @ P[j, :].reshape(1, -1) @ A.conj().T).conj().T)
            # with respect to q_j
            elif (col >= k and col < k + n):
                j = col - k
                e_j = np.zeros((n, ))
                e_j[j] = 1 
                Sigma_j = np.diag(e_j)
            # with respect to p_i
            else:
                j = col - k - n
                Sigma_j = A @ P_diff_i(j, k) @ A.conj().T


            #if row == col:
            #    pdb.set_trace()
            term1 = (psi - 1) * np.trace(LA.solve(Sigma, Sigma_i)) * np.trace(LA.solve(Sigma, Sigma_j)) 
            term2 = psi * np.trace(LA.solve(Sigma, Sigma_i) @ LA.solve(Sigma, Sigma_j))

            F_G[row, col] = term2 / psi

            F[row, col] = term1 + term2

    
    F = F + F.T - np.diag(np.diag(F))
    F_G = F_G + F_G.T - np.diag(np.diag(F_G))

    # compute CRB matrix
    CRB = (1/m) * LA.inv(F)

    CRB_G = (1/m) * LA.inv(F_G)

    CRB = np.real(CRB[0:k, 0:k])
    CRB_G = np.real(CRB_G[0:k, 0:k])
    return CRB, CRB_G