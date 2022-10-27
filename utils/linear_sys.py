__all__ = ["sys_integrator"]
           
           
__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Robust Learning."
__license__ 	= "Microsoft License"
__comment__     = "A collection of general linear system utilities."
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Completed"


import logging
import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg as la
import scipy.linalg as sla
import numpy.random as npr
from .matlab_utils import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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