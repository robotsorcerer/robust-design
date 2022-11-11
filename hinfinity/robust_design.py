__all__ = ["gamma_lowerbound", "hamiltonian", "get_hinf_norm", "solve_care"]

__date__        = "November 02, 2022"
__comment__     = "H-Infinity Norm Utilities for Robust control."
__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Robust Learning."
__license__ 	= "Microsoft License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Completed"
__credits__ = "Richard Murray and Steve Brunton"

import copy
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

import sys 
sys.path.append("..")
from linearsys import (compute_sigma, smat, kron, svec)

def gamma_lowerbound(A, B1, B2, C, D, K):
    """Compute the starting lower bound for the 
    search for the H infinity norm of a system.
    
    
        See section 4 (Equations 4.3 to 4.4) in the paper below:
    
        @article{BruinsmaSteinbuch,
        title={{A Fast Algorithm to Compute the $H_\infty$-norm of a Transfer Function Matrix}},
        author={Bruinsma, NA and Steinbuch, M},
        journal={Systems \& Control Letters},
        volume={14},
        number={4},
        pages={287--293},
        year={1990},
        publisher={Elsevier}
        };
    """

    # First find omega_p
    poles = sla.eig(A-B1@K)[0]
    if np.any(np.iscomplex(poles)):
        omega_p = max([np.abs(p.imag/p.real)*(1/np.abs(p)) for p in poles])
    else:
        omega_p = min([np.abs(p) for p in poles])

    # G_omega_p = (C - D@K)@la.pinv(1j*omega_p - A + B1@K)@B2
        
    AA = A - B1@K 
    BB = B2
    CC = C-D@K
    Dnew = np.array([[0]])
    sigma_zero = max(compute_sigma(AA, BB, CC, Dnew, 0))
    sigma_omegap = max(compute_sigma(AA, BB, CC, Dnew, omega_p))
    sigma_D = max(la.svd(D, compute_uv = False))

    result = max(sigma_zero, sigma_omegap, sigma_D)

    return result

def hamiltonian(A, B1, B2, C, D, K, gamma, R=None):
    """Compute the Hamiltonian for a two-player LQ zero sum differential game.
        See equation (15) in my IFAC (2022) paper.

        Parameters
        ----------
        System matrices (arrays) A, B1, B2, C, and D.
        gamma: Robustness parameter for H infinity controller 
        R: Input penalization matrix for the LQ problem. 
            If not provided, it is returned as identity.
            
        Returns
        --------
        H(\gamma): (array) The system's Hamiltonian.
    """
    if not R:
        R = np.eye(K.T.shape[-1])
        
    a11 = A - B1@K
    a12 = -(1/gamma)*B2@B2.T
    a21 = -(1/gamma)*(C.T@C + K.T@R@K)
    a22 = (-A - B1@K).T


    top_row = np.hstack((a11, a12))
    bot_row = np.hstack((a21, a22))

    ham = np.vstack((top_row, bot_row))

    return ham
    
def get_hinf_norm(A, B1, B2, C, D, K, step_size=0.1):
    """Compute the H infinity norm using Bruinsma's algorithm.

        See section 3.3 in the paper below:
    
        @article{BruinsmaSteinbuch,
        title={{A Fast Algorithm to Compute the $H_\infty$-norm of a Transfer Function Matrix}},
        author={Bruinsma, NA and Steinbuch, M},
        journal={Systems \& Control Letters},
        volume={14},
        number={4},
        pages={287--293},
        year={1990},
        publisher={Elsevier}
        };

        Parameters
        ----------
        A: (array) State transition matrix
        B1: (array) Control matrix
        B2: (array)  Disturbance matrix
        C: (array) Output transmission matrix
        D: (array) Feedthrough matrix
        step_size: (float) degree by which to move up the signal.
    """
    gamma_lb = gamma_lowerbound(A, B1, B2, C, D, K)

    # set gamma_ub to inf
    gamma_ub = np.inf 
    AA = A - B1@K 
    BB = B2
    CC = C-D@K
    while gamma_ub==np.inf:
        gamma = (1+2*step_size)*gamma_lb 

        # get Hamiltonian for this gamma
        Hgamma = hamiltonian(A, B1, B2, C, D, K, gamma)

        # get eigen values of H(gamma)
        eigs = sla.eig(Hgamma)[0]

        if not np.any(np.iscomplex(eigs)):
            gamma_ub = gamma
            break
        else:
            imags = eigs[np.nonzero(np.iscomplex(eigs))]
            gamma_lb_tmp = [np.nan for i in range(len(imags))]
            for i in range(len(imags)-1):
                m_tmp = 0.5*(imags[i]+imags[i+1])
                sigmas_tmp = compute_sigma(AA, BB, CC, 0, m_tmp)
                gamma_lb[i] = max(sigmas_tmp)
            gamma_lb = max(gamma_lb_tmp)

    return 0.5*(gamma_lb + gamma_ub)

def solve_care(A, B1, B2, Q, R, γ, S=None, E=None, stabilizing=True, method=None, dt=0.0001):
    """
        A Continuous-time (closed-loop) Riccati equation solver for two players 
        in a zero-sum linear quadratic differential game setting.

        Solve the equation 

            :math:`AP +A^T P - P (B_1 R^{-1} B_1^T - \gamma^{-2} B_2 B_2^T) P + Q = 0`

        where A and Q are square matrices of same dimension. In addition, Q is a symmetric 
        positive definite matrix. It returns the solution P, the gain matrices, K and L, as 
        well as the closed-loop eigenvalues of (A - B_1 K + B_2 L), where K and L are the 
        feedback gains of the two players given by 

            :math: `K=R^{-1}B_1^T P,      L = -\gamma^{-2} B_2^T P.`
        
        For details, see the IFAC paper by Lekan Molu and Hosein Hasanbeig.

        Parameters
        ----------
        A, B1, B2, Q : 2D arrays
            Input matrices for the Riccati equation.
        γ : The H infinity risk measure.
        R, S, E : 2D arrays, optional
            Input matrices for generalized Riccati equation.
        method : str, optional
            Set the method used for computing the result.  Current methods are
            'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
            and then 'scipy'.
        dt : float, optional
            Step size of the integration algorithm

        Returns
        -------
        X : 2D array (or matrix)
            Solution to the Ricatti equation
        V : 1D array
            Closed loop eigenvalues
        K : 2D array (or matrix) for minimizing player
        L : 2D array (or matrix) for maximizing player
            Gain matrix

        Notes
        -----
        Author: Lekan Molu
        Date: October 19, 2022
    """

    # assert method is not None, "method must be 'slycot' or 'scipy'"


    # Reshape input arrays
    A = np.array(A, ndmin=2)
    B1 = np.array(B1, ndmin=2)
    B2 = np.array(B2, ndmin=2)
    Q = np.array(Q, ndmin=2)
    R = np.eye(B1.shape[1]) if R is None else np.array(R, ndmin=2)
    if S is not None:
        S = np.array(S, ndmin=2)
    if E is not None:
        E = np.array(E, ndmin=2)

    # Determine main dimensions
    n = A.shape[0]
    m = B1.shape[1]

    Rinv = la.inv(R)

    P = np.zeros((n, n))
    # initialization for stopping condition
    P0 = np.ones((n, n))
    step = 0

    while la.norm(P0-P, ord=2)>1e-8 and step < 1e7:
         step += 1
         P0 = copy.copy(P)

         P += (A.T@P0 + P0@A + Q - P0@(B1@Rinv@B1.T - 1/(γ**2)*B2@B2.T)@P0)*dt

    K = Rinv@B1.T@P
    L = 1/(γ**2)*B2.T@P

    return P, K, L    

def iterative_robust(A, B1, B2, C, K1, gamma, Phi1, Phi2, Psi, PIter=40, QIter=80):
    """Solve the gains in an iDG robust control framework.
    
    """

    n = A.shape[0]
    K = np.zeros((PIter,)+(K1.shape))
    K[0,:] = K1
    Rmat = np.eye(K1.T.shape[-1])
    P = np.zeros((PIter, QIter, n,n))


    for p in range(1, PIter):
        L = np.zeros((QIter, B2.shape[0], A.shape[0]))
        for q in range(QIter):
            t1 = Phi1@Psi 
            t2 = smat(kron(np.eye(n), K[p-1,:].T) + kron( K[p-1,:].T, np.eye(n)) )@Phi2@Psi 
            t3 = smat(kron(np.eye(n), L[q-1, :].T@B2.T) + kron( L[q-1, :]@B2.T, np.eye(n)))

            Upsilon = t1 - t2 + t3

            Qp = C.T@C - K[p-1,:].T@Rmat@K[p-1,:]

            P[p,q] = la.pinv(Upsilon)@svec(Qp - (1/(gamma**2))*L[q-1,:].T@L[q-1,:])
            L[q-1,:] = gamma**(-2)*B2.T@P[p,q]
        K[p, :] = la.pinv(Rmat)*B1.T@P[p, q]

    return K, L, P    