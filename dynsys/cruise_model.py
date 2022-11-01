__all__ = ["CruiseControlModel"]

from re import M
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import control as ct

class CruiseControlModel():
    def __init__(self, n=1, m=3, d=2):
        """Parameters for the model. 
            n: state dimension (velocity of the car)
            m: control dimension (Throttle, gear ratio, road)
            d: disturbance dimension (Rolling Friction, Aero Drag and Gravity)
        """
        self.n = n
        self.m = m
        self.d = d 

    def gen_bin_seq(self, N, p_swd, nmin=1, interval=[-1.0, 1.0], tol = 0.01, nit_max = 30):
        """
            Function which generates a sequence Generalized Binary Noise (GBN).

            Params
            ------
            N: Sequence length (total number of samples)
            p_swd: desired probability of switching (no switch: 0<x<1 :always switch)
            nmin: minimum number of samples between two switches
            Range: input range
            tol: tolerance on switching probability relative error
            nit_max: maximum number of iterations

            Returns
            -------
            GBN: size(N); Switching probability, Number of times actually switched.

        """
        min_Range = min(interval)
        max_Range = max(interval)
        prob = np.random.random()
        # set first value
        if prob < 0.5:
            gbn = -1.0*np.ones(N)
        else:
            gbn = 1.0*np.ones(N)
        # init. variables
        p_sw = p_sw_b = 2.0             # actual switch probability
        nit = 0; 
        while (np.abs(p_sw - p_swd))/p_swd > tol and nit <= nit_max:
            i_fl = 0; Nsw = 0
            for i in range(N - 1):
                gbn[i + 1] = gbn[i]
                # test switch probability
                if (i - i_fl >= nmin):
                    prob = np.random.random()
                    # track last test of p_sw
                    i_fl = i
                    if (prob < p_swd):
                        # switch and then count it
                        gbn[i + 1] = -gbn[i + 1]
                        Nsw = Nsw + 1
            # check actual switch probability
            p_sw = nmin*(Nsw+1)/N; #print("p_sw", p_sw);
            # set best iteration
            if np.abs(p_sw - p_swd) < np.abs(p_sw_b - p_swd):
                p_sw_b = p_sw
                Nswb = Nsw
                gbn_b = gbn.copy()
            # increase iteration number
            nit = nit + 1; #print("nit", nit)
        # rescale GBN
        for i in range(N):
            if gbn_b[i] > 0.:
                gbn_b[i] = max_Range
            else:
                gbn_b[i] = min_Range

        return gbn_b, p_sw_b, Nswb

    def gen_white_noise(self, l, sigma):
        """Generate a white noise sequence using a zero-mean ans sigma-variance.

            Inputs
            ------
            l: (int) first dim of the noise signal
            sigma: (float or array): if array, each element of the array is used in
            computing a white noise signal.
        """
        if not isinstance(sigma, np.ndarray):
            var = np.array((sigma))
        else:
            var = sigma 
        n = var.size 
        noise = np.zeros((n, l))

        for i in range(n):
            if var[i] < np.finfo(np.float64).eps:
                import sys 
                var[i] = np.finfo(np.float64).eps 
                sys.stdout.write("\033[0;35m")
                print("Warning: Var[", i,
                    "] may be too small, its value is below machine epsilon.")
                sys.stdout.write(" ")
            noise[i,:] = np.random.normal(0., var[i]**0.5, l)
        
        return noise
        
    def motor_torque(self, omega, params={}):
        # Set up the system parameters
        Tm = params.get('Tm', 190.)             # engine torque constant
        omega_m = params.get('omega_m', 420.)   # peak engine angular speed
        beta = params.get('beta', 0.4)          # peak engine rolloff

        return np.clip(Tm * (1 - beta * (omega/omega_m - 1)**2), 0, None)

    def dynamics(self, v, u, params={}):
        """Vehicle dynamics for cruise control system.

        Parameters
        ----------
        v : array
            System state: car velocity in m/s
        u : array
            System input: [throttle, gear, road_slope], where throttle is
            a float between 0 and 1, gear is an integer between 1 and 5,
            and road_slope is in rad.

        Returns
        -------
        float
            Vehicle acceleration

        """
        assert isinstance(u, list), "u must be a list of respective input arrays."
        
        from math import copysign, sin
        sign = lambda x: copysign(1, x)         # define the sign() function
        
        # Set up the system parameters
        m = params.get('m', 1600.)              # vehicle mass, kg
        g = params.get('g', 9.8)                # gravitational constant, m/s^2
        Cr = params.get('Cr', 0.01)             # coefficient of rolling friction
        Cd = params.get('Cd', 0.32)             # drag coefficient
        rho = params.get('rho', 1.3)            # density of air, kg/m^3
        A = params.get('A', 2.4)                # car area, m^2
        alpha = params.get(
            'alpha', [40, 25, 16, 12, 10])      # gear ratio / wheel radius

        # Define variables for vehicle state and inputs
        throttle = u[0]     # vehicle throttle
        gear     = u[1]     # vehicle gear
        theta    = u[2]     # road slope

        # Force generated by the engine

        omega = alpha[int(gear[0])-1] * v      # engine angular speed
        F = omega * self.motor_torque(omega, params) * throttle

        # Disturbance forces
        #
        # The disturbance force Fd has three major components: Fg, the forces due
        # to gravity; Fr, the forces due to rolling friction; and Fa, the
        # aerodynamic drag.

        # Letting the slope of the road be \theta (theta), gravity gives the
        # force Fg = m g sin \theta.
        
        Fg = m * g * np.sin(theta)

        # A simple model of rolling friction is Fr = m g Cr sgn(v), where Cr is
        # the coefficient of rolling friction and sgn(v) is the sign of v (±1) or
        # zero if v = 0.
        
        Fr  = m * g * Cr * np.array([sign(x) for x in v])

        # The aerodynamic drag is proportional to the square of the speed: Fa =
        # 1/2 \rho Cd A |v| v, where \rho is the density of air, Cd is the
        # shape-dependent aerodynamic drag coefficient, and A is the frontal area
        # of the car.

        Fa = 1/2 * rho * Cd * A * abs(v) * v
        
        # Final acceleration on the car
        Fd = Fg + Fr + Fa
        dv = (F - Fd) / m
        
        return dv

    def data_collect(self, ndata=500):
        """Generate input signal for system identification data collection on system
        
            Params
            ------
            n: (int) Total number of datapoints needed.

            Returns I/O Dataset Z = [[U], [Y]]
        """
        # noise exploration variance 
        var = [0.005]
        e   = self.gen_white_noise(ndata, var)

        # generate throttle signals as white noise corrupted sinusoids
        # e = self.gen_white_noise(n, [0.005])
        ux = np.linspace(0, 2*np.pi, e.size)

        uref = (np.sin(ux)+0.7)/2

        # Define the gear and road curvature vectors
        gear = 4 * np.ones((ndata))

        # make road inclination a Wiener process 
        noise_Gauss = np.random.normal(0, 1, ndata)
        dw = np.cumsum(noise_Gauss)
        theta = dw #np.array([4./180. * pi + dw]).squeeze()


        # theta = [
        #     0 if t <= ndata//20 else
        #     4./180. * pi * (t-5) if t <= ndata//10 else
        #     4./180. * pi for t in range(ndata)]

        u = [ 
                uref, #np.clip(uref, 0, 1),
                gear, # pick one gear ratio for the entire data collection regime
                theta
            ]

        # torques (states) for different gears, as a function of velocity 
        v = np.linspace(1, ndata, ndata)

        ode_rhs = self.dynamics(v=v, u=u)

        # use this for NARMAX identification
        return np.asarray(u), ode_rhs
        



