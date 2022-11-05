__all__ = ["gamma_lowerbound", "hamiltonian", "get_hinf_norm"]

__date__        = "November 02, 2022"
__comment__     = "H-Infinity Norm Utilities for Robust control."
__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Robust Learning."
__license__ 	= "Microsoft License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Completed"
__credits__ = "Richard Murray and Steve Brunton"


def gamma_lowerbound(A, B1, B2, C, D, K):
    """Compute the starting lower bound for the
    search for the H infinity norm of a closed-loop system.


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
    if np.any(np.iscomplex(pole)):
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
