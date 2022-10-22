__all__ = ["Pendulum2d"]

import numpy as np 
import scipy.linalg as sla

class Pendulum2d():
    def __init__(self, n, m, mn, dt, with_vel=True):
        """
            The 2D Pendulum System 
            ----------------------
            n: (int) System dimensions
            m: (int) Control dimensions 
            mn: (int) Noise dimensions
            dt: (float) Time step for integrations
            with_vel: (bool) Form system matrices with their time derivative counterparts
        """
        if with_vel:
            self.state_dim = n*2
            self.control_dim = m*2
            self.noise_dim = mn*2
        else:
            self.state_dim = n 
            self.control_dim = m 
            self.noise_dim = mn
        self.time_step = dt 

        """
            Flag to indicate that (A,B) has indeed been learned to be stabilizable
            and (C, A) has been learned to be observable.
        """
        self.learned = False 

    def system_matrices(self):
        """
            Construct the structure of the system matrices to be learned.
        """
        self.A  = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        self.B1 = np.zeros((self.state_dim, self.control_dim), dtype=np.float64)
        self.B2 = np.zeros((self.state_dim, self.noise_dim), dtype=np.float64)
        self.C  = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        self.D  = np.zeros((self.state_dim, self.control_dim), dtype=np.float64)

    def dynamics(self, cur_x, cur_u):
        """
            Advance a single step of the dynamical system by solving the 
            stochastic differential equation with additive Wiener process noise.

            Parameters
            ----------
            cur_x: current state to integrate.
            cur_u: current control law to use.
        """
        
        # Obtain a derivative of the Wiener noise process as a stochastic white noise
        mu, sigma = 0, 1 
        w = np.random.normal(mu, sigma, size=(2,1))

        # using knowledge of a stabilizable (A, B1) and an observable (C, A), do a forward step of the 
        # dynamics.
        x_next = cur_x + (self.A@cur_x + self.B@cur_u)*self.time_step + self.B2@w*np.sqrt(self.time_step)

        return x_next 
    
    def implicit_euler(t, x0, A, B1, B2, U):
        """Solve the system

            dx / dt = Ax(t) + B1 u(t) + B2 dw(t)/dt,    x(0) = x0,

        over a uniform time domain via the implicit Euler method.

        Parameters
        ----------
        t : (k,) ndarray
            Uniform time array over which to solve the ODE.
        x0 : (n,) ndarray
            Initial condition.
        A : (n, n) ndarray
            State matrix.
        B1 : (n,) or (n, 1) ndarray
            Input matrix.
        B2 : (n,) or (n, 1) ndarray
            Process noise matrix.
        U : (k,) ndarray
            Inputs over the time array.
        W : (k,) ndarray
            Process inputs over the time array.

        Returns
        -------
        x : (n, k) ndarray
            Solution to the ODE at time t; that is, x[:,j] is the
            computed solution corresponding to time t[j].
        """
        # Check and store dimensions.
        tf = len(t)
        n = len(x0)
        B1 = np.ravel(B1)
        assert A.shape == (n, n)
        assert B1.shape == (n,)
        assert U.shape == (tf,)
        I = np.eye(n)

        # Check that the time step is uniform.
        dt = t[1] - t[0]
        assert np.allclose(np.diff(t), dt)

        # Factor I - dt*A for quick solving at each time step.
        factored = sla.lu_factor(I - dt*A)

        # Solve the problem by stepping in time.
        x = np.empty((n, tf))
        x[:,0] = x0.copy()
        for j in range(1, tf):
            x[:, j] = sla.lu_solve(factored, x[:, j-1] + dt*B1*U[j])

        return q

    def learn_ables(self):
        """"
            Given the system structure defined in `system matrices`,
            learn a stabilizable pair (A,B) and an observable pair (C, A)
            within a realization of the system transfer function after gathering data.

            This has to be done in a constrained optimization setting.
        """
        # TODO 
        pass