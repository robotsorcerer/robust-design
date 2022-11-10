__all__ = ["vec", "vecv", "svec", "svec2", "mat",  "mdot", "specrad", "vec2vecT",
           "smat", "smat2", "acker", "sympart", "is_pos_def", "succ", "precc", "psdpart", 
           "kron", "ctrb", "obsv", "is_symmetric", "place", "check_shape", "is_controllable", 
           "is_observable", "place_varga", "sys_integrator"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Robust Learning."
__license__ 	= "Microsoft License"
__comment__ 	= "Utilities peculiar to linear systems."
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Completed"


import sys 
sys.path.append("..")

import logging
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

import numpy.linalg as la
import scipy.linalg as sla
import numpy.random as npr
from functools import reduce
from utils import (isfield, Bundle)

# Make sure we have access to the right slycot routines
try:
    from slycot import sb03md57
    # wrap without the deprecation warning
    def sb03md(n, C, A, U, dico, job='X',fact='N',trana='N',ldwork=None):
        ret = sb03md57(A, U, C, dico, job, fact, trana, ldwork)
        return ret[2:]
except ImportError:
    try:
        from slycot import sb03md
    except ImportError:
        sb03md = None

try:
    from slycot import sb03od
except ImportError:
    sb03od = None


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def vec(A):
    """
        Return the vectorized matrix A by stacking its columns
        on top of one another.

        Input: Matrix A.
        Output: Vectorized form of A.
    """

    return A.reshape(-1, order="F")

def svec(P):
    """
        Return the symmetric vectorization of P i.e.
        the vectorization of the upper triangular part of matrix P.

        Inputs
        ------
        P:  (array) Symmetric matrix in \mathbb{S}^n

        Returns
        -------
        svec(P) = [p_{11} , p_{12} , · · · , p_{1n} , · · · , p_{nn} ]^T

    Author: Lekan Molux, Nov 10, 2022
    """

    assert is_symmetric(P), "P must be a symmetric matrix."

    return P[np.triu_indices(P.shape[0])]

def svec2(P):
    """Return the half-vectorization of matrix P such that its off diagonal entries are doubled.

    Inputs
    ------
    P:  (array) Symmetric matrix in \mathbb{S}^n

    Returns
    -------
    vecs(P):= [p11, 2p12, · · · , 2p1n, p22 , · · · , pnn ]

    Author: Lekan Molux, Nov 10, 2022
    """

    assert is_symmetric(P), "P must be a symmetric matrix."

    T = np.tril(P, 0) + np.triu(P,1)*2

    return T[np.triu_indices(T.shape[0])]

# def svec2(A):
#     """
#         Return the symmetric vectorization i.e. the vectorization
#         of the upper triangular part of matrix A
#         with off-diagonal entries multiplied by sqrt(2)
#         so that la.norm(A, ord='fro')**2 == np.dot(svec2(A), svec2(A))
#     """
    # assert is_symmetric(A), "A must be a symmetric matrix."
#     B = A + np.triu(A)*(2)

#     return B[np.triu_indices(A.shape[0])]

def mat(v, shape=()):
    """
        Return matricization of vector v i.e. the
        inverse operation of vec of vector v.
    """
    assert isinstance(shape, (tuple, list)), "shape must be an instance of list or tuple"
    m,n = shape
    matrix = kron(vec(np.eye(n)).T, np.eye(m))@kron(np.eye(n), v)

    return matrix

def smat(v):
    """
        Return the symmetric matricization of vector v
        i.e. the  inverse operation of svec of vector v.
    """
    m = v.size
    n = int(((1+m*8)**0.5 - 1)/2)
    idx_upper = np.triu_indices(n)
    idx_lower = np.tril_indices(n, -1)
    A = np.zeros([n, n])
    A[idx_upper] = v
    A[idx_lower] = A.T[idx_lower]

    return A

def smat2(v):
    """
        Return the symmetric matricization of vector v
        i.e. the inverse operation of svec2 of vector v.

        #ToDo: This appears to solve for the case where the 
        off-diag entries are sqrt(V_{mn})
    """
    m = v.size
    n = int(((1+m*8)**0.5 - 1)/2)
    idx_upper = np.triu_indices(n)
    idx_lower = np.tril_indices(n, -1)
    A = np.zeros([n, n])
    A[idx_upper] = v
    A[np.triu_indices(n,1)] /= 2**0.5
    A[idx_lower] = A.T[idx_lower]

    return A

def mdot(*args):
    """Multiple dot product."""

    return reduce(np.dot, args)

def specrad(A):
    """Spectral radius of matrix A."""

    try:
        return np.max(np.abs(la.eig(A)[0]))
    except np.linalg.LinAlgError:
        return np.nan

def vecv(x):
    """Compute the vectorized dot product of x^T and x. Return the """
    xv = kron(x, x)
    ij = np.array(([0]))
    for i in range(1, len(x)+1):
        ij = np.append(ij, np.arange((i-1)*len(x),(i-1)*len(x)+i-1), axis=0)
    ij = ij[1:]
    xv = np.delete(xv, ij, axis=0)

    return xv

def vec2vecT(nr, nc):
    """
        Calculates the transformation matrix from vec(X) to vec(X')

        X has nr row and nc column

        Calling Sig
        -----------
        vec(X') = Tv2v*vec(X)

        Input:
            nr, nc: Numbers of rows and columns respectively.

        Returns:
            T_vt: The transformation matrix.

        Author: Leilei Cui.
    """

    T_vt = np.zeros((nr*nc, nr*nc))

    for i in range(nr):
        for j in range(nc):
            T_vt[i*nc+j,j*nr+i] = 1

    return T_vt


def ctrb(A, B):
    """
        Controllabilty matrix

        Parameters
        ----------
        A, B: array_like
            Dynamics and input matrix of the system

        Returns
        -------
        C: matrix
            Controllability matrix

        Examples
        --------
        >>> C = ctrb(A, B)
    """

    n = np.shape(A)[0]
    C = np.hstack([B] + [la.matrix_power(A, i)@B for i in range(1, n)])

    return C

def is_controllable(A, B):
    """Test if a system is controllable.

        Parameters
        ----------
        A, B: array_like
            Dynamics and input matrix of the system
    """
    ct = ctrb(A, B)

    if la.matrix_rank(ct) != A.shape[0]:
        return False
    else:
        return True


def obsv(C, A):
    """
        Observability matrix

        Parameters
        ----------
        A, C: array_like
            Dynamics and input matrix of the system

        Returns
        -------
        O: matrix
            Controllability matrix

        Examples
        --------
        >>> C = obsv(C, A)
    """

    n = np.shape(A)[0]
    O = np.hstack([C] + [C@la.matrix_power(A, i) for i in range(1, n)])

    return O

def is_observable(A, C):
    """Test if a system is controllable.

        Parameters
        ----------
        A, C: array_like
            State transition and output matrices of the system
    """
    ct = obsv(A, C)

    if la.matrix_rank(ct) != A.shape[0]:
        return False
    else:
        return True

def acker(A, B, poles):
    """
        Pole placement using Ackerman's formula.


        Call:
        K = acker(A, B, poles)

        Parameters
        ----------
        A, B : 2D array_like
            State and input matrix of the system
        poles : 1D array_like
            Desired eigenvalue locations

        Returns
        -------
        K : 2D array (or matrix)
            Gains such that A - B K has given eigenvalues

        Adopted from Richard Murray's code.
    """

    # Make sure the system is controllable
    ct = ctrb(A, B)
    if la.matrix_rank(ct) != A.shape[0]:
        raise ValueError("System not reachable; pole placement invalid")

    # Compute the desired characteristic polynomial
    p = np.real(np.poly(poles))

    # Place the poles using Ackermann's method
    # TODO: compute pmat using Horner's method (O(n) instead of O(n^2))
    n = np.size(p)
    pmat = p[n-1] * la.matrix_power(A, 0)
    for i in np.arange(1, n):
        pmat = pmat + p[n-i-1] * la.matrix_power(A, i)
    K = np.linalg.solve(ct, pmat)

    K = K[-1][:]                # Extract the last row

    return K

# Pole placement
def place(A, B, p):
    """Place closed loop eigenvalues

    K = place(A, B, p)

    Parameters
    ----------
    A : 2D array_like
        Dynamics matrix
    B : 2D array_like
        Input matrix
    p : 1D array_like
        Desired eigenvalue locations

    Returns
    -------
    K : 2D array (or matrix)
        Gain such that A - B K has eigenvalues given in p

    Notes
    -----
    Algorithm
        This is a wrapper function for :func:`scipy.signal.place_poles`, which
        implements the Tits and Yang algorithm [1]_. It will handle SISO,
        MISO, and MIMO systems. If you want more control over the algorithm,
        use :func:`scipy.signal.place_poles` directly.

    Limitations
        The algorithm will not place poles at the same location more
        than rank(B) times.

    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    References
    ----------
    .. [1] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
       pole assignment by state feedback, IEEE Transactions on Automatic
       Control, Vol. 41, pp. 1432-1452, 1996.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place(A, B, [-2, -5])

    See Also
    --------
    place_varga, acker

    Notes
    -----
    Lifted from statefdbk function in Murray's python control.
    """
    from scipy.signal import place_poles

    # Convert the system inputs to NumPy arrays
    if (A.shape[0] != A.shape[1]):
        raise ValueError("A must be a square matrix")

    if (A.shape[0] != B.shape[0]):
        err_str = "The number of rows of A must equal the number of rows in B"
        raise ValueError(err_str)

    # Convert desired poles to numpy array
    placed_eigs = np.atleast_1d(np.squeeze(np.asarray(p)))

    result = place_poles(A, B, placed_eigs, method='YT')
    K = result.gain_matrix
    return K


def place_varga(A, B, p, dtime=False, alpha=None):
    """Place closed loop eigenvalues
    K = place_varga(A, B, p, dtime=False, alpha=None)

    Required Parameters
    ----------
    A : 2D array_like
        Dynamics matrix
    B : 2D array_like
        Input matrix
    p : 1D array_like
        Desired eigenvalue locations

    Optional Parameters
    ---------------
    dtime : bool
        False for continuous time pole placement or True for discrete time.
        The default is dtime=False.

    alpha : double scalar
        If `dtime` is false then place_varga will leave the eigenvalues with
        real part less than alpha untouched.  If `dtime` is true then
        place_varga will leave eigenvalues with modulus less than alpha
        untouched.

        By default (alpha=None), place_varga computes alpha such that all
        poles will be placed.

    Returns
    -------
    K : 2D array (or matrix)
        Gain such that A - B K has eigenvalues given in p.

    Algorithm
    ---------
    This function is a wrapper for the slycot function sb01bd, which
    implements the pole placement algorithm of Varga [1]. In contrast to the
    algorithm used by place(), the Varga algorithm can place multiple poles at
    the same location. The placement, however, may not be as robust.

    [1] Varga A. "A Schur method for pole assignment."  IEEE Trans. Automatic
        Control, Vol. AC-26, pp. 517-519, 1981.

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place_varga(A, B, [-2, -5])

    See Also:
    --------
    place, acker

    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb01bd
    except ImportError:
        raise print("can't find slycot module 'sb01bd'")

    # Convert the system inputs to NumPy arrays
    if (A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]):
        raise ValueError("matrix dimensions are incorrect")

    # Compute the system eigenvalues and convert poles to numpy array
    system_eigs = np.linalg.eig(A)[0]
    placed_eigs = np.atleast_1d(np.squeeze(np.asarray(p)))

    # Need a character parameter for SB01BD
    if dtime:
        DICO = 'D'
    else:
        DICO = 'C'

    if alpha is None:
        # SB01BD ignores eigenvalues with real part less than alpha
        # (if DICO='C') or with modulus less than alpha
        # (if DICO = 'D').
        if dtime:
            # For discrete time, slycot only cares about modulus, so just make
            # alpha the smallest it can be.
            alpha = 0.0
        else:
            # Choosing alpha=min_eig is insufficient and can lead to an
            # error or not having all the eigenvalues placed that we wanted.
            # Evidently, what python thinks are the eigs is not precisely
            # the same as what slicot thinks are the eigs. So we need some
            # numerical breathing room. The following is pretty heuristic,
            # but does the trick
            alpha = -2*abs(min(system_eigs.real))
    elif dtime and alpha < 0.0:
        raise ValueError("Discrete time systems require alpha > 0")

    # Call SLICOT routine to place the eigenvalues
    A_z, w, nfp, nap, nup, F, Z = \
        sb01bd(B.shape[0], B.shape[1], len(placed_eigs), alpha,
               A, B, placed_eigs, DICO)

    # Return the gain matrix, with MATLAB gain convention
    return -F


def sympart(A):
    """
        Return the symmetric part of matrix A.
    """

    return 0.5*(A+A.T)

def is_pos_def(A):
    """Check if matrix A is positive definite."""
    try:
        la.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Utility function to check if a matrix is symmetric
def is_symmetric(M):
    "This from Richard Murray's Python Control Toolbox."
    M = np.atleast_2d(M)
    if isinstance(M[0, 0], np.inexact):
        eps = np.finfo(M.dtype).eps
        return ((M - M.T) < eps).all()
    else:
        return (M == M.T).all()

def succ(A,B):
    """Check the positive definite partial ordering of A > B."""

    return is_pos_def(A-B)

def precc(A,B):
    """Check the negative definite partial ordering of A < B."""

    return not is_pos_def(A-B)

def psdpart(X):
    """Return the positive semidefinite part of a symmetric matrix."""
    X = sympart(X)
    Y = np.zeros_like(X)
    eigvals, eigvecs = la.eig(X)
    for i in range(X.shape[0]):
        if eigvals[i] > 0:
            Y += eigvals[i]*np.outer(eigvecs[:,i],eigvecs[:,i])
    Y = sympart(Y)

    return Y

def kron(*args):
    """Overload and extend the numpy kron function to take a single argument."""
    if len(args)==1:
        return np.kron(args[0], args[0])
    else:
        return np.kron(*args)


# Utility function to check matrix dimensions
def check_shape(name, M, n, m, square=False, symmetric=False):
    "Verify the dims of matrix M."
    if square and M.shape[0] != M.shape[1]:
        raise logger.warn("%s must be a square matrix" % name)

    if symmetric and not is_symmetric(M):
        raise logger.warn("%s must be a symmetric matrix" % name)

    if M.shape[0] != n or M.shape[1] != m:
        raise logger.warn("Incompatible dimensions of %s matrix" % name)

def sys_integrator(sys, X0, K_init, T):
    """Algorithm: to integrate from time 0 to time dt.

        Inputs
        ------
        sys: (Bundle) Object containing A, B, C, D, E matrices.
        X0:  (array) Initial conditions
        K_init:   (array) Initial feedback gain to be used for forced response simulation.
        T : (array), optional for discrete LTI `sys`
            Time steps at which the input is defined; values must be evenly spaced.
    """

    assert isinstance(sys, Bundle), "sys must be of Bundle Type"
    assert isfield(sys, "A"), "Field A is not in sys."
    assert isfield(sys, "B1"), "Field B1 is not in sys."
    assert isfield(sys, "B2"), "Field B2 is not in sys."
    assert isfield(sys, "C"), "Field C is not in sys."
    assert isfield(sys, "D"), "Field D is not in sys."
    assert isfield(sys, "tf"), "Field tf (final time step) is not in sys."
    assert isfield(sys, "dt"), "Field dt (integration time step) is not in sys."

    dt = 1. if sys.dt in [True, None] else sys.dt

    A, B1, B2, C, D = sys.A, sys.B1, sys.B2, sys.C, sys.D

    n_states  = A.shape[0]
    n_inputs  = B1.shape[1]
    n_disturbs  = B2.shape[1]
    n_outputs = C.shape[0]
    n_steps   = T.shape[0]            # number of simulation steps

    input_data = np.zeros((T.shape[0], n_inputs))
    states_data = np.zeros((T.shape[0], n_states))
    states_data[0,:] = X0

    xi   = np.zeros((n_steps, n_inputs)) # exploration input 
    uout = np.zeros((n_steps, n_inputs))

    xout = np.zeros((n_steps, n_states))
    xout[:, 0] = X0

    zout = np.zeros((n_steps, n_outputs))

    for i in range(1, n_steps):
        xi[i,:] -= xi[i-1,:]*dt + np.random.normal(0, 1, (2))*np.sqrt(dt) # encourage exploration noise.
        uout[i-1,:] = -K_init@xout[i-1,:] + 10*xi[i,:]
        xout[i,:] = xout[i-1,:] + (A @ xout[i-1,:] + B1 @ uout[i-1,:])*dt + B2 @ xi[i,:]*np.sqrt(dt)
        zout[i,:] = C @ xout[i,:] + D @ uout[i,:]


    return xout, uout, zout