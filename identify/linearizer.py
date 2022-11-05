__all__ = ["narmax_model", "controlled_output", "find_eq", "linearize"]


__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Robust Learning."
__license__ 	= "Microsoft License"
__comment__     = "A NARMAX model estimator for a car cruise controller."
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__date__        = "November 04, 2022"
__status__ 		= "Completed"


import warnings
import numpy as np

    
def controlled_output(x, u0):
    """Solve the output equation for the H_\infty control problem (eq. 2) 
    in paper.
    """
    if not isinstance(x, np.ndarray): x = np.array(x) 
    if not isinstance(u0, np.ndarray): u0 = np.array(u0)
    C = np.ones((1, x.size))
    D = np.ones((1, u0.size))

    return C@x + D@u0

def find_eq(sys, x0, u0, z0, t=0):
    # TODO: This function seems to return spurious results.
    """
        Find the equilibrium point for an input/output system.

        Returns the value of an equilibrium point given the initial state and
        either input value or desired output value for the equilibrium point.

        Parameters
        ----------
        sys: NARMAX model functor
        x0 : list of initial state values
            Initial guess for the value of the state near the equilibrium point.
        u0 : list of input values, optional
            If `z0` is not specified, sets the equilibrium value of the input.  If
            `z0` is given, provides an initial guess for the value of the input.
            Can be omitted if the system does not have any inputs.
        w0 : list of disturbance values, optional
            If `w0` is not specified, sets the equilibrium value of the input.  If
            `z0` is given, provides an initial guess for the value of the input.
            Can be omitted if the system does not have any inputs.
        z0 : list of output values, optional
            If specified, sets the desired values of the outputs at the
            equilibrium point.
        t : float, optional
            Evaluation time, for time-varying systems
    """

    from scipy.optimize import root

    # Figure out the number of states, inputs, and outputs
    nstates = len(x0)
    ninputs = len(u0)
    noutputs = len(z0)

    # Convert x0, u0, y0 to arrays, if needed
    if np.isscalar(x0):
        x0 = np.ones((nstates,)) * x0
    if np.isscalar(u0):
        u0 = np.ones((ninputs,)) * u0
    # if np.isscalar(w0):
    #     dw = np.random.normal(0, w0, ninputs)
    #     w0 = np.cumsum(dw)
    #     ndisturbs = len(w0)
    if isinstance(z0, list):
        z0 = np.asarray((z0))

    if z0 is None:
        def ode_rhs(z): return sys(z, u, None)
        result = root(narmax_model, x0)
        z = (result.x, u0, controlled_output(result.x, u0))
    else:
        # Take z0 as fixed and minimize over x and u
        def rootfun(z):
            # Split z into x and u
            x, u = np.asarray(([z[0]])), np.asarray(z[1:3]) #, np.asarray(z[3:])

            narm_pred = narmax_model(x, u, u3=None)
            out_pred = controlled_output(x, u)
            # print(rf"narm_pred: {narm_pred.shape}, out_pred: {out_pred.shape}, z0: {z0.shape} out-z0: {(out_pred-z0).shape}")

            temp = np.concatenate((narm_pred, out_pred - z0), axis=0)

            return temp 

        y0 = np.concatenate((x0, u0) )                # Put variables together
        result = root(rootfun, y0)                      # Find the eq point
        
        x, u = np.split(result.x, [nstates])    # Split result back in two
        z = (x, u, controlled_output(x, u))

    iu = [1,2]
    ix = []
    iz = list(range(noutputs))
    idx = list(range(nstates))

    # Get the states and inputs that were not listed as fixed
    state_vars = (range(nstates) if not len(ix)
                    else np.delete(np.array(range(nstates)), ix))
    input_vars = (range(ninputs) if not len(iu)
                    else np.delete(np.array(range(ninputs)), iu))

    # Set the outputs and derivs that will serve as constraints
    output_vars = np.array(iz)
    deriv_vars = np.array(idx)

    # Verify that the number of degrees of freedom all add up correctly
    num_freedoms = len(state_vars) + len(input_vars)
    num_constraints = len(output_vars) + len(deriv_vars)
    if num_constraints != num_freedoms:
        warnings.warn("Number of constraints (%d) does not match number of degrees "
                "of freedom (%d).  Results may be meaningless." % (num_constraints, num_freedoms))

    # Make copies of the state and input variables to avoid overwriting and convert to floats (in case ints were used for initial conditions)
    x = np.array(x0, dtype=float)
    u = np.array(u0, dtype=float)
    dx0 = np.zeros(x.shape)

    # Keep track of the number of states in the set of free variables
    nstate_vars = len(state_vars)

    def rootfun(z):
        # Map the vector of values into the states and inputs
        x[state_vars] = z[:nstate_vars]
        u[input_vars] = z[nstate_vars:]

        # Compute the update and output maps
        dx = controlled_output(x, u) - dx0

        dy = controlled_output(x, u) - z0

        # Map the results into the constrained variables
        return np.concatenate((dx[deriv_vars], dy[output_vars]), axis=0)

    # Set the initial condition for the root finding algorithm
    z0 = np.concatenate((x[state_vars], u[input_vars]), axis=0)

    # Finally, call the root finding function
    result = root(rootfun, z0)

    # Extract out the results and insert into x and u
    x[state_vars] = result.x[:nstate_vars]
    u[input_vars] = result.x[nstate_vars:]
    z = (x, u, controlled_output(x, u))

    return z + (result,)


def narmax_model(x, u, u3=None):
    # print(rf"x: {x} u: {u}, u3: {u3}")
    if len(u)<3:
        u1, u2 = u
    elif len(u)==3:
        u1, u2, u3 = u
    elif len(x) >3:
        x, u1, u2, u3 = x
    if isinstance(x, list): x=x[0]

    result = np.array([6.2518e-02*u2**2 * x -1.2067e-01 * u2**2 * u1 + 5.6692e-10*u2*x**2 + 8.0976e-03*u2**3]).reshape(-1) #squeeze()

    return result


def linearize(x0, u0, w0, z0=None):
    """
    Linearize the nonlinear model about the 
    equilibrium points, x0, u0, w0, z0. Ideally, 
    we should return this equlibrium points from 
    the system using the function `find_eq` above.

    Parameters
    ---------
    x0: (list) initial state
    u0: (list) initial control
    w0: (list) initial disturbance
    z0: (list) initial outputs

    Returns
    -------
    Linearized model:
    A, B1, B2, C, D matrices (Eq. 2 in the IFAC paper).
    """

    nstates = len(x0)
    ninputs = len(u0)
    ndisturb = len(u0)
    noutputs = len(x0) if not z0 else len(z0)

    x0, u0, w0 = np.array(x0), np.array(u0), np.array(w0)

    F0 = narmax_model(x0, u0, w0)
    H0 = controlled_output(x0, u0)

    # Create empty matrices that we can fill up with linearizations
    A = np.zeros((nstates, nstates))        # Dynamics matrix
    B1 = np.zeros((nstates, ninputs))        # Input matrix
    B2 = np.zeros((nstates, ndisturb))       # Output matrix
    C = np.zeros((noutputs, nstates))       # Direct term
    D = np.zeros((noutputs, ninputs))       # Direct term


    eps = 0.02 + np.finfo(np.float64).eps

    # Perturb each of the state variables and compute linearization
    for i in range(nstates):
        dx = np.zeros((nstates,))
        dx[i] = eps
        A[:, i] = (narmax_model(x0 + dx, u0) - F0) / eps
        C[:, i] = (controlled_output(x0 + dx, u0) - H0) / eps


    # Perturb each of the input variables and compute linearization
    for i in range(ninputs):
        du = np.zeros((ninputs,))
        du[i] = eps
        B1[:, i] = (narmax_model(x0, u0 + du) - F0) / eps
        D[:, i] = (controlled_output(x0, u0 + du) - H0) / eps

    # Perturb each of the disturbance variables and compute linearization
    for i in range(ninputs):
        noise_Gauss = np.random.normal(0, 1, ndisturb)
        dw = np.cumsum(noise_Gauss)
        dw[i] = eps
        B2[:, i] = (narmax_model(x0, u0 + dw) +  - F0) / eps    

    return A, B1, B2, C, D