import numpy as np 
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
robjects.r('library(fitHeavyTail)')

def TFA_ECME(X, rank):
    X_r = numpy2ri.numpy2rpy(X.T)  
    fit_mvt_iterative = robjects.r['fit_mvt'](X_r, nu="iterative",
                                              nu_iterative_method="ECME", 
                                              optimize_mu="FALSE", factors=rank, 
                                              max_iter=200)
    F_T = np.array(fit_mvt_iterative.rx2('B'))
    d_T = np.array(fit_mvt_iterative.rx2('psi'))
    v_estimated = fit_mvt_iterative.rx2('nu')

    return F_T, d_T