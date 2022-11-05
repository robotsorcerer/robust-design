__all__ = ["frequency_response", "lti_zero", "compute_sigma"]

__date__        = "November 02, 2022"
__comment__     = "Linear Quadratic Utilities for Robust control."
__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Robust Learning."
__license__ 	= "Microsoft License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Completed"
__credits__ = "Richard Murray and Steve Brunton"

import logging
import numpy as np
import scipy as sp

import numpy.linalg as la
import scipy.linalg as sla

import logging
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def frequency_response(A, B, C, D, omega):
    """Evaluate the linear time-invariant system (A, B, C, D)
    at an array of angular frequencies, omega.

    Reports the frequency response of the system,

            G(j*omega) = mag*exp(j*phase)

    for continuous time systems.

    In general the system may be multiple input, multiple output (MIMO),
    where `m = ninputs` number of inputs and `p = noutputs` number
    of outputs.

    Parameters
    ----------
    omega : float or 1D array_like
        A list, tuple, array, or scalar value of frequencies in
        radians/sec at which the system will be evaluated.
    Returns
    -------
    mag : ndarray
        The magnitude (absolute value, not dB or log10) of the system
        frequency response.  If the system is SISO and squeeze is not
        True, the array is 1D, indexed by frequency.  If the system is not
        SISO or squeeze is False, the array is 3D, indexed by the output,
        input, and frequency.  If ``squeeze`` is True then
        single-dimensional axes are removed.
    phase : ndarray
        The wrapped phase in radians of the system frequency response.
    omega : ndarray
        The (sorted) frequencies at which the response was evaluated.
    """
    nstates = A.shape[0]

    omega = np.sort(np.array(omega, ndmin=1))

    s = 1j * omega

    # evaluate the transfer function at a complex frequency using Horner's method.
    s_arr = np.atleast_1d(s).astype(np.complex, copy=False)


    if nstates == 0:
        return D[:, :, np.newaxis] * np.ones_like(s_arr, dtype=complex)
    if nstates == 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            response = C[:, :, np.newaxis] / (s_arr - A[0, 0]) \
                * B[:, :, np.newaxis] \
                + D[:, :, np.newaxis]
        response[np.isnan(response)] = complex(np.inf, np.nan)
        return np.abs(response), np.angle(response), omega


    # Preallocate
    response = np.empty((noutputs, ninputs, len(s_arr)),
                dtype=complex)

    import warnings
    for idx, s_idx in enumerate(s_arr):
        try:
            xr = np.linalg.solve(s_idx * np.eye(nstates) - A, B)
            response[:, :, idx] = C @ xr + D
        except np.linalg.linalg.LinAlgError:
            # Issue a warning messsage, for consistency with xferfcn
            warnings.warn("singular matrix in frequency response")

            # Evaluating at a pole.  Return value depends if there
            # is a zero at the same point or not.
            if s_idx in lti_zero(A, B, C, D):
                response[:, :, idx] = np.complex(np.nan, np.nan)
            else:
                response[:, :, idx] = np.complex(np.inf, np.nan)

    return np.abs(response), np.angle(response), omega


def lti_zero(A, B, C, D):
    """Compute the zeros of a state space system."""
    nstates = A.shape[0]

    if not nstates:
        return np.array([])

    # Use AB08ND from Slycot if it's available, otherwise use
    # scipy.lingalg.eigvals().
    try:
        from slycot import ab08nd

        out = ab08nd(A.shape[0], B.shape[1], C.shape[0],
                        A, B, C, D)
        nu = out[0]
        if nu == 0:
            return np.array([])
        else:
            return sp.linalg.eigvals(out[8][0:nu, 0:nu],
                                        out[9][0:nu, 0:nu])

    except ImportError:  # Slycot unavailable. Fall back to scipy.
        if C.shape[0] != D.shape[1]:
            raise NotImplementedError("StateSpace.zero only supports "
                                        "systems with the same number of "
                                        "inputs as outputs.")

        # This implements the QZ algorithm for finding transmission zeros
        # from
        # https://dspace.mit.edu/bitstream/handle/1721.1/841/P-0802-06587335.pdf.
        # The QZ algorithm solves the generalized eigenvalue problem: given
        # `L = [A, B; C, D]` and `M = [I_nxn 0]`, find all finite lambda
        # for which there exist nontrivial solutions of the equation
        # `Lz - lamba Mz`.
        #
        # The generalized eigenvalue problem is only solvable if its
        # arguments are square matrices.
        L = np.concatenate((np.concatenate((A, B), axis=1),
                            np.concatenate((C, D), axis=1)), axis=0)
        M = np.pad(np.eye(A.shape[0]), ((0, C.shape[0]),
                                        (0, B.shape[1])), "constant")
        return np.array([x for x in sp.linalg.eigvals(L, M,
                                                        overwrite_a=True)
                            if not np.isinf(x)])

def compute_sigma(A, B, C, D, w):
    """Compute the singular value of the state space model at a given frequency point, w.
        This function closely mimics the sigma function in matlab.

        Parameters
        ==========
        A: (array) System transitin matrix.
        B: (array) System input matrix.
        C: (array) System output matrix.
        D: (array) System feedthrough matrix.
        w: (length m) frequencies of the singular values of SS(A, B, C, D).

        Returns
        =======
        sv: (array) Singular values of the state space LTI system.
    """

    mag, phase, _ = frequency_response(A, B, C, D, w)
    svd_jw = (mag * np.exp(1j * phase)).transpose(2,0,1)
    svd    = np.linalg.svd(svd_jw, compute_uv = False)

    return svd


def sys_integrator(sys, X0, U, T):
    """Algorithm: to integrate from time 0 to time dt, with linear
        interpolation between inputs u(0) = u0 and u(dt) = u1, we solve
            xdot = A x + B u,        x(0) = x0
            udot = (u1 - u0) / dt,   u(0) = u0.

        Solution is
            [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
            [ u(dt) ] = exp [  0     0    I ] [  u0   ]
            [u1 - u0]       [  0     0    0 ] [u1 - u0]

        Adopted from R.M. Murray's `timeresp` code.

        Inputs
        ------
        sys: (Bundle) Object containing A, B, C, D, E matrices.
        X0:  (array) Initial conditions
        U:   (array) Control to be used for forced response simulation.
        T : (array), optional for discrete LTI `sys`
            Time steps at which the input is defined; values must be evenly spaced.

            If None, `U` must be given and `len(U)` time steps of sys.dt are
            simulated. If sys.dt is None or True (undetermined time step), a time
            step of 1.0 is assumed.
    """

    assert isinstance(sys, Bundle), "sys must be of Bundle Type"
    assert isfield(sys, "A"), "Field A is not in sys."
    assert isfield(sys, "B"), "Field B is not in sys."
    assert isfield(sys, "C"), "Field C is not in sys."
    assert isfield(sys, "D"), "Field D is not in sys."
    assert isfield(sys, "E"), "Field E is not in sys."
    assert isfield(sys, "dt"), "Field dt (integration time step) is not in sys."

    dt = 1. if sys.dt in [True, None] else sys.dt

    A, B, C, D = sys.A, sys.B, sys.C, sys.D

    n_states  = A.shape[0]
    n_inputs  = B.shape[1]
    n_outputs = C.shape[0]
    n_steps   = T.shape[0]            # number of simulation steps

    M = np.block([[A * dt, B * dt, np.zeros((n_states, n_inputs))],
                    [np.zeros((n_inputs, n_states + n_inputs)),
                    np.identity(n_inputs)],
                    [np.zeros((n_inputs, n_states + 2 * n_inputs))]])
    expM = sla.expm(M)
    Ad = expM[:n_states, :n_states]
    Bd1 = expM[:n_states, n_states+n_inputs:]
    Bd0 = expM[:n_states, n_states:n_states + n_inputs] - Bd1

    xout = np.zeros((n_states, n_steps))
    xout[:, 0] = X0
    yout = np.zeros((n_outputs, n_steps))

    for i in range(1, n_steps):
        xout[:, i] = (Ad @ xout[:, i-1]
                        + Bd0 @ U[:, i-1] + Bd1 @ U[:, i])
    yout = C @ xout + D @ U


    return xout, yout
