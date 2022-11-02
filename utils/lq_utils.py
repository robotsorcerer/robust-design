__all__ = ["vec", "svec", "svec2", "mat",  "mdot", "specrad", "vec2vecT",
           "smat", "smat2", "acker", "sympart", "is_pos_def", "succ", 
           "precc", "psdpart", "kron", "ctrb", "obsv", "is_symmetric", 
           "check_shape", "is_controllable", "is_observable"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Robust Learning."
__license__ 	= "Microsoft License"
__comment__ 	= "LQ, H2, and H_\infty Control Theory Utilities."
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Completed"

import logging
import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg as la
import scipy.linalg as sla
import numpy.random as npr
from functools import reduce

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

def svec(A):
    """
        Return the symmetric vectorization of A i.e.
        the vectorization of the upper triangular part of matrix A.
    """

    return A[np.triu_indices(A.shape[0])]

def svec2(A):
    """
        Return the symmetric vectorization i.e. the vectorization
        of the upper triangular part of matrix A
        with off-diagonal entries multiplied by sqrt(2)
        so that la.norm(A, ord='fro')**2 == np.dot(svec2(A), svec2(A))
    """
    B = A + np.triu(A, 1)*(2**0.5 - 1)

    return B[np.triu_indices(A.shape[0])]

def mat(v, shape=None):
    """
        Return matricization of vector v i.e. the
        inverse operation of vec of vector v.
    """
    if shape is None:
        dim = int(np.sqrt(v.size))
        shape = dim, dim
    matrix = v.reshape(shape[1], shape[0]).T

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
    """
        Compute the \langle vec(x), vec^T(x) \rangle vector product between the column vector
        x and its row vector transformation.

        vecv(x) = [x1^2, x1x2,..., x1xn, x2^2, ..., xn^2]

        Input:
            x: n-dimensional vector
        Output:
            x_vecv: n*(n+1)/2 dimensional vector
    """

    x_vecv = np.kron(x,x)

    idx = np.array([])
    for i in range(1,len(x)):
        idx = np.append(idx, np.linspace(i*len(x), i*len(x)+i, num=i, endpoint=False))
    idx = idx.astype(int)
    x_vecv = np.delete(x_vecv, idx)

    return x_vecv

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
