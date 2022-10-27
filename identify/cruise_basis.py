__all__ = ["CruiseControlBasis"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Robust Learning."
__license__ 	= "Microsoft License"
__comment__     = "A NARMAX model estimator for a car cruise controller."
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Completed"

import numpy as np
import sys

sys.path.append("../sysidentpy")
from sysidentpy.narmax_base import InformationMatrix
from sysidentpy.basis_function._basis_function import Polynomial


from itertools import chain, combinations_with_replacement

class CruiseControlBasis(InformationMatrix):
    def __init__(self, xlag, ylag, noise_var=0.005,
                 degree=2, ensemble=False, model_type="NARMAX"):
        """Build custom auto regressivedegree moving average with exogeneous input basis function.
        Follows example 2.3 (eq. 2.7.1 in Billings' book.)
        Generate a new feature matrix consisting of all polynomial combinations
        of the features with degree less than or equal to the specified degree.

        ..math:
            y_k = \sum_{i=1}^{p}\Theta_i \times \prod_{j=0}^{n_x}u_{k-j}^{b_i, j}\prod_{l=1}^{n_e}e_{k-l}^{d_i, l}\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
            \label{eq5:narx}

        where :math:`p` is the number of regressors, :math:`\Theta_i` are the
        model parameters, and :math:`a_i, m, b_i, j` and :math:`d_i, l \in \mathbb{N}`
        are the exponents of the output, input and noise terms, respectively.

        Parameters
        ----------
        ylag : int, default=2
            The maximum lag of the output.
        xlag : int, default=2
            The maximum lag of the input.
        degree : int (max_degree), default=2
            The maximum degree of the polynomial features.
        noise_var : (float) Variance for the process noise;
        max_lag : int
            Target data used on training phase.

        Notes
        -----
        Be aware that the number of features in the output array scales
        signirficantly as the number of inputs, the max lag of the input and output, and
        degree increases. High degrees can cause overfitting.

        """
        self.ensemble = ensemble
        self.repetition = 3


        if isinstance(xlag, int): xlag = range(1, xlag+1)
        if isinstance(ylag, int): ylag = range(1, ylag + 1)

        "get max lag"
        nx = np.max(list(chain.from_iterable([[xlag]])))
        ny = np.max(list(chain.from_iterable([[ylag]])))
        max_lag = np.max([ny, np.max(nx)])

        self.xlag = xlag 
        self.ylag = ylag 
        self.max_lag = max_lag
        self.noise_var = noise_var
        self.degree = degree
        self.non_degree = degree
        self.model_type = model_type

    def fit(self, X, y, predefined_regressors=None):
        """Build the Polynomial information matrix.

        Each columns of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and degree defined by the user.

        Parameters
        ----------
        data : ndarray of floats
            The lagged matrix built with respect to each lag and column.

        predefined_regressors : ndarray of int
        deg: int 
            Degree of the regressors (as would be for polynomials).

            The index of the selected regressors by the Model Structure
            Selection algorithm.

        Returns
        -------
        psi = ndarray of floats
            The lagged matrix built in respect with each lag and column.

        Example:
            reg_matrix = basis_function.fit(X=x_train, y=y_train,  ylag=2, xlag=[[1,1,1], [1,1,1], [1,1,1]], deg=2)
        """

        self.n_inputs = X.shape[1]

        "Get x lagged data"
        if self.n_inputs == 1:
            x_lagged = np.column_stack([self.shift_column(X[:, 0], lag) for lag in self.xlag])
        else:
            x_lagged = np.zeros([len(X), 1])  # just to stack other columns
            # if user input a nested list like [[1, 2], 4], the following
            # line convert it to [[1, 2], [4]].
            # Remember, for multiple inputs all lags must be entered explicitly
            xlag = [[i] if isinstance(i, int) else i for i in self.xlag]
            for col in range(self.n_inputs):
                x_lagged_col = np.column_stack(
                    [self.shift_column(X[:, col], lag) for lag in self.xlag[col]]
                )
                x_lagged = np.column_stack([x_lagged, x_lagged_col])

            x_lagged = x_lagged[:, 1:]  # remove the column of 0 created above

        "Get y lagged data"
        y_lagged = np.column_stack([self.shift_column(y[:,0], lag) for lag in self.ylag])
        P_init = np.concatenate([y_lagged, x_lagged], axis = 1)  # this is the information matrix for polynomial terms only

        "Add Gaussian stochastic noise to the info matrix model."
        noise = np.random.normal(0, self.noise_var, [P_init.shape[0], 1])
        P_matrix = np.concatenate([noise, P_init], axis=1)
        
        "Add the signum terms in P's info matrix model; all things being equal, we'd expect ERR to filter out the unused terms"
        P_matrix = np.concatenate([np.expand_dims(np.sign(x_lagged[:,0]), 1), P_matrix], axis=1)

        "Add the sinusoids of road curvature to the info matrix model"
        P_matrix = np.concatenate([np.expand_dims(np.sin(x_lagged[:,-1]), 1), P_matrix], axis=1)

        "Add the |x|sgn(x) term to P matrix as well."
        P_matrix = np.concatenate([np.abs(y_lagged)*np.sign(y_lagged), P_matrix], axis=1)

        "We'd expect NARMAX to pick up the u1*u2 term but add it ok?"
        P_matrix = np.concatenate([np.expand_dims(x_lagged[:,0]*x_lagged[:,1], 1), P_matrix], axis=1)


        "Get a nonlinear combo of all input-output-noise terms in the polynomial function"
        combos = list(combinations_with_replacement(range(P_matrix.shape[1]), self.non_degree))
        psi    = np.column_stack([
                            np.prod(P_matrix[:, combos[i]], axis=1) for i in range(len(combos))
                        ])

        print("Cruise Basis Final Regression mat: ", P_matrix.shape)
        # Don't know why Wilson tried this but worth exploring (following the Fourier basis extension)
        psi = psi[1:, ]

        return psi

    def transform(self, data, max_lag, predefined_regressors=None):
        return self.fit(data, max_lag, predefined_regressors)

    def predict(self, X, y_init, theta, horizon=None):
        "This is a shoo-in for _basis_function_predict in sysidentpy.narmax_base"
        if X is not None:
            horizon = X.shape[0]
        else:
            horizon += self.max_lag 

        y_pred = np.zeros(horizon, dtype=np.float64)
        y_pred.fill(np.nan)
        y_pred[:, self.max_lag] = y_init[:self.max_lag, 0]

        # remove unneeded initial values 
        to_be_removed = self.max_lag + 1

        for i in range(0, horizon - self.max_lag):
            if self.model_type is "NARMAX":
                


        
