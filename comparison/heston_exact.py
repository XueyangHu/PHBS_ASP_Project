# -*- coding: utf-8 -*-


import numpy as np
import math
from scipy.special import iv  # import modified Bessel function of the first kind
from scipy import interpolate
import scipy.optimize as sopt
from .option_model import OptionModelABC
from .sabr_mc import CondMcBsmModelABC


class HestonModelABC1(OptionModelABC):
    """
    Heston model:
    d ST = r*St*dt + sqrt(Vt)*St*[rho*dWt1+sqrt(1-rho**2)*dWt2]
    d Vt = k*(theta-Vt)*dt + sigma_v*sqrt(Vt)*dWt1
    """

    theta, k, sigma_v, rho, intr, divr, beta = None, None, None, None, None, None, None

    def __init__(self, sigma, theta, k, sigma_v, rho=0, intr=0, divr=0, is_fwd=False):

        # sigma: initial volatility
        # sigma_v: volatility of volatility
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)

        self.theta = theta
        self.k = k
        self.sigma_v = sigma_v
        self.rho = rho
        self.intr = intr
        self.divr = divr


class HestonCondMcBsmModel(CondMcBsmModelABC, HestonModelABC1):
    """
    Heston model with conditional Monte-Carlo simulation
    """

    def vol_paths(self, time):

        np.random.seed(self.rn_seed)

        n_steps = len(time)
        dt = np.insert(np.diff(time), 0, time[0])
        vov_sqrt_dt = self.sigma_v * np.sqrt(dt)

        # Antithetic
        if self.antithetic:
            zz = np.random.normal(size=(n_steps, int(self.n_paths / 2)))
            zz = np.concatenate((zz, -zz), axis=1)
        else:
            zz = np.random.normal(size=(n_steps, self.n_paths))

        V = np.empty([n_steps, self.n_paths])  # variance series: V0, V1,...,VT
        V0 = self.sigma**2
        for i in range(n_steps):
            V0 = V0 + self.k*(self.theta - V0)*dt[i] + self.sigma*vov_sqrt_dt[i]*zz[i, :]  # Euler method
            # V0 = V0 + 0.25 * self.sigma_v**2 * (zz[i, :]**2 - dt[i])  # Milstein method
            V0[V0 < 0] = 0  # variance should be larger than zero
            V[i, :] = V0

        # return normalized sigma, e.g., sigma(0) = 1
        return np.sqrt(V)/self.sigma

    def conditional_fwd_vol(self, texp):

        sigma_final, int_var_std = self.conditional_states(texp)
        
        V0 = self.sigma**2
        VT_ratio = sigma_final**2
        
        # VT = VT_ratio * V0
        # int_var = int_var_std * texp * V0
        
        temp = ((VT_ratio-1)*V0 - self.k * texp * (self.theta - int_var_std*V0)) / self.sigma_v
        # formula: ln(ST/S0) = r*T - 0.5*IT + rho*temp + sqrt(1-rho**2)* W1 * sqrt(IT)
        fwd_cond = np.exp(self.rho * temp - 0.5 * self.rho**2 * int_var_std*V0*texp)
        vol_cond = np.sqrt((1 - self.rho**2) * int_var_std)

        # return normalized forward and volatility
        return fwd_cond, vol_cond

    
class HestonExactMcBsmModel(HestonCondMcBsmModel, HestonModelABC1):
    """
    BSM Heston model with exact Monte-Carlo simulation
    """

    def set_mc_params_new(self, KK, n_paths, rn_seed=None, antithetic=True):
        # KK: truncate gamma expansion at level KK
        self.n_paths = n_paths
        self.KK = KK
        self.antithetic = antithetic
        self.rn_seed = rn_seed

    def conditional_states(self, texp):

        """
        Return conditional states (final sigma, integrated variance) from exact simulation
        Method is Gamma Expansion in Glasserman and Kim (2011)

        Parameters
        ----------
        texp: float
            time-to-expiry

        Returns
        -------
        (sigma_T, integrated variance)
        """

        np.random.seed(self.rn_seed)

        V0 = self.sigma**2
        # sample VT: coef * noncentral chi-squared distribution(NCX2(delta, lambda))
        temp = np.exp(-self.k*texp)
        coef = self.sigma_v**2 * (1-temp) / (4*self.k)
        delta = 4*self.theta*self.k / (self.sigma_v**2)
        lambda1 = 4*self.k*temp / (self.sigma_v**2 * (1-temp)) * V0
        VT = coef * np.random.noncentral_chisquare(delta, lambda1, self.n_paths)

        # sample int_var(integrated variance): Gamma expansion / transform inversion
        # int_var = X1+X2+X3 from formula(2.7) in Glasserman and Kim (2011)

        # Simulation X1: truncated Gamma expansion
        X1 = self.generate_X1_gamma_expansion(VT, texp)

        # Simulation X2: transform inversion
        coth = 1 / np.tanh(self.k * texp * 0.5)
        csch = 1 / np.sinh(self.k * texp * 0.5)
        mu_X2_0 = self.sigma_v**2 * (-2 + self.k*texp*coth) / (4 * self.k**2)
        sigma_square_X2_0 = self.sigma_v**4 * (-8 + 2*self.k*texp*coth + (self.k*texp*csch)**2) / (8 * self.k**4)
        X2 = self.generate_X2_and_Z_AW(mu_X2_0, sigma_square_X2_0, delta, texp, self.n_paths)
        # X2 = self.generate_X2_and_Z_gamma_expansion(delta, texp, self.n_paths)

        # Simulation X3: X3=sum(Z, eta), Z is a special case of X2 with delta=4
        Z = self.generate_X2_and_Z_AW(mu_X2_0, sigma_square_X2_0, 4, texp, self.n_paths*10)
        # Z = self.generate_X2_and_Z_gamma_expansion(4, texp, self.n_paths*10)

        v = 0.5 * delta - 1
        z = 2 * self.k * self.sigma*np.sqrt(VT) * csch / self.sigma_v**2
        eta = self.generate_eta(v, z)

        X3 = np.zeros(len(eta))
        for ii in range(len(eta)):
            X3[ii] = np.sum(Z[np.random.randint(0, len(Z), int(eta[ii]))])

        sigma_final = np.sqrt(VT)
        int_var = X1 + X2 + X3

        return sigma_final/self.sigma, int_var/(texp*V0)

    def generate_X1_gamma_expansion(self, VT, texp):
        """
        Simulation of X1 using truncated Gamma expansion in Glasserman and Kim (2011)

        Parameters
        ----------
        VT : an 1-d array with shape (n_paths,)
            final variance
        texp: float
            time-to-expiry

        Returns
        -------
         an 1-d array with shape (n_paths,), random variables X1
        """
        V0 = self.sigma**2
        # For fixed k, theta, sigma_v, texp, generate some parameters firstly
        range_K = np.arange(1, self.KK+1)
        temp = 4*np.pi**2 * range_K**2
        gamma_n = (self.k**2 * texp**2 + temp) / (2*self.sigma_v**2 * texp**2)
        lambda_n = 4*temp / (self.sigma_v**2 * texp * (self.k**2 * texp**2 + temp))

        E_X1_K_0 = 2*texp / (np.pi**2 * range_K)
        Var_X1_K_0 = 2 * self.sigma_v**2 * texp**3 / (3 * np.pi**4 * range_K**3)

        # the following para will change with VO and VT
        Nn_mean = ((V0+VT)[:, None] * lambda_n[None, :])  # every row K numbers (one path)
        Nn = np.random.poisson(lam=Nn_mean).flatten()
        rv_exp_sum = np.zeros(len(Nn))
        for ii in range(len(Nn)):
            rv_exp_sum[ii] = np.sum(np.random.exponential(scale=1, size=Nn[ii]))
        rv_exp_sum = rv_exp_sum.reshape(len(VT), len(lambda_n))
        X1_main = np.sum((rv_exp_sum / gamma_n), axis=1)

        gamma_mean = (V0+VT) * E_X1_K_0[-1]
        gamma_var = (V0+VT) * Var_X1_K_0[-1]
        beta = gamma_mean / gamma_var
        alpha = gamma_mean * beta
        X1_truncation = np.random.gamma(alpha, 1/beta)
        # X1_truncation = np.random.normal(loc=gamma_mean, scale=np.sqrt(gamma_var))
        X1 = X1_main + X1_truncation

        return X1

    def generate_X2_and_Z_AW(self, mu_X2_0, sigma_square_X2_0, delta, texp, num_rv):
        """
        Simulation of X2 or Z from its CDF based on Abate-Whitt algorithm from formula (4.1) in Glasserman and Kim (2011)

        Parameters
        ----------
        mu_X2_0:  float
            mean of X2 from formula(4.2)
        sigma_square_X2_0: float
            variance of X2 from formula(4.3)
        delta: float
            a parameter, which equals to 4*theta*k / (sigma_v**2) when generating X2 and equals to 4 when generating Z
        texp: float
            time-to-expiry
        num_rv: int
            number of random variables you want to generate

        Returns
        -------
         an 1-d array with shape (num_rv,), random variables X2 or Z
        """

        mu_X2 = delta * mu_X2_0
        sigma_square_X2 = delta * sigma_square_X2_0

        mu_e = mu_X2 + 14 * np.sqrt(sigma_square_X2)
        w = 0.01
        M = 200
        xi = w * mu_X2 + np.arange(M+1) / M * (mu_e - w*mu_X2)  # x1,...,x M+1
        L = lambda x: np.sqrt(2 * self.sigma_v**2 * x + self.k**2)
        fha_2 = lambda x: (L(x)/self.k * (np.sinh(0.5*self.k*texp) / np.sinh(0.5*L(x)*texp)))**(0.5*delta)
        fha_2_vec = np.vectorize(fha_2)
        err_limit = np.pi * 1e-5 * 0.5  # the up limit error of distribution Fx1(x)

        h = 2 * np.pi / (xi + mu_e)
        # increase N to make truncation error less than up limit error, N is sensitive to xi and the model parameter
        F_X2_part = np.zeros(len(xi))
        for pos in range(len(xi)):
            Nfunc = lambda N:  abs(fha_2(-1j * h[pos] * N)) - err_limit * N
            N = int(sopt.brentq(Nfunc, 0, 5000))+1
            N_all = np.arange(1, N+1)
            F_X2_part[pos] = np.sum(np.sin(h[pos] * xi[pos] * N_all) * fha_2_vec(-1j * h[pos] * N_all).real / N_all)

        F_X2 = (h * xi + 2 * F_X2_part) / np.pi

        # Next we can sample from this tabulated distribution using linear interpolation
        rv_uni = np.random.uniform(size=num_rv)
        xi = np.insert(xi, 0, 0.)
        F_X2 = np.insert(F_X2, 0, 0.)
        F_X2_inv = interpolate.interp1d(F_X2, xi, kind="slinear")
        X2 = F_X2_inv(rv_uni)

        return X2

    def generate_eta(self, v, z):
        """
        generate Bessel random variables from inverse of CDF, formula(2.4) in George and Dimitris (2010)

        Parameters
        ----------
        v:  float
            parameter in Bessel distribution
        z: an 1-d array with shape (n_paths,)
            parameter in Bessel distribution

        Returns
        -------
         an 1-d array with shape (n_paths,), Bessel random variables eta
        """

        p0 = np.power(0.5*z, v) / (iv(v, z) * math.gamma(v+1))
        temp = np.arange(1, 31)[:, None]  # Bessel distribution has sort tail, 30 maybe enough
        p = z**2 / (4 * temp * (temp + v))
        p = np.vstack((p0, p)).cumprod(axis=0).cumsum(axis=0)
        rv_uni = np.random.uniform(size=len(z))
        eta = np.sum(p < rv_uni, axis=0)
        
        return eta

    def generate_X2_and_Z_gamma_expansion(self, delta, texp, num_rv):
        """
        Simulation of X2 or Z using truncated Gamma expansion in Glasserman and Kim (2011)

        Parameters
        ----------
        delta: float
            a parameter, which equals to 4*theta*k / (sigma_v**2) when generating X2 and equals to 4 when generating Z
        texp: float
            time-to-expiry
        num_rv: int
            number of random variables you want to generate

        Returns
        -------
         an 1-d array with shape (num_rv,), random variables X2 or Z
        """

        range_K = np.arange(1, self.KK+1)
        temp = 4*np.pi**2 * range_K**2
        gamma_n = (self.k**2 * texp**2 + temp) / (2*self.sigma_v**2 * texp**2)

        rv_gamma = np.random.gamma(0.5*delta, 1, size=(num_rv, self.KK))
        X2_main = np.sum(rv_gamma / gamma_n, axis=1)

        gamma_mean = delta*(self.sigma_v*texp)**2 / (4*np.pi**2 * self.KK)
        gamma_var = delta * (self.sigma_v*texp)**4 / (24 * np.pi**4 * self.KK**3)
        beta = gamma_mean / gamma_var
        alpha = gamma_mean * beta
        X2_truncation = np.random.gamma(alpha, 1/beta, size=num_rv)
        X2 = X2_main + X2_truncation

        return X2