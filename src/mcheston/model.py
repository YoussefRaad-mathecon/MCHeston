"""
HestonModel
===========

Unified Heston stochastic-volatility model: simulation, Monte Carlo pricing,
and exact (Fourier) pricing.

Simulation schemes (path-vectorised):
    - Euler (full truncation)
    - Milstein (full truncation)
    - Truncated Gaussian (TG)
    - Truncated Gaussian + Martingale correction (TG-M)
    - Quadratic Exponential (QE)            [Andersen, 2008]
    - Quadratic Exponential + Martingale correction (QE-M)
    - Broadie-Kaya drift-interpolation (almost-exact)

Exact / semi-analytic pricers:
    - Heston original Fourier inversion
    - Heston "stable" / "Little Trap" formulation (Albrecher et al.)
    - Carr-Madan FFT-style integral
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, chi2, poisson
from scipy.integrate import quad
from scipy.optimize import root_scalar


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_SQRT_2PI_INV = 1.0 / np.sqrt(2.0 * np.pi)


def _phi(x):
    """Standard-normal PDF."""
    return _SQRT_2PI_INV * np.exp(-0.5 * x * x)


def _Phi(x):
    """Standard-normal CDF."""
    return norm.cdf(x)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class HestonModel:
    """Heston stochastic-volatility model.

    Asset and variance dynamics under the pricing measure:
        dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_t^S
        dv_t = kappa_tilde * (theta_tilde - v_t) * dt + sigma * sqrt(v_t) * dW_t^v
        d<W^S, W^v>_t = rho dt

    where kappa_tilde = kappa + lambda_ and theta_tilde = kappa*theta/(kappa+lambda_)
    are the risk-adjusted mean reversion speed and long-run variance.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    v0 : float
        Initial variance.
    r : float
        Risk-free interest rate.
    kappa : float
        Variance mean-reversion speed (physical measure).
    theta : float
        Long-run variance level (physical measure).
    sigma : float
        Volatility-of-variance ("vol of vol").
    rho : float
        Correlation between asset and variance Brownian motions.
    lambda_ : float, default 0.0
        Market price of volatility risk.
    gamma1, gamma2 : float, default (0.5, 0.5)
        Discretisation weights used by the QE/TG log-spot update
        (Andersen central-discretisation scheme; gamma1 + gamma2 = 1).
    alpha : float, default 4.5
        Switching parameter used by the TG scheme grid.
    """

    # ---------------------------- construction --------------------------- #

    def __init__(
        self,
        S0,
        v0,
        r,
        kappa,
        theta,
        sigma,
        rho,
        lambda_=0.0,
        gamma1=0.5,
        gamma2=0.5,
        alpha=4.5,
    ):
        self.S0 = float(S0)
        self.v0 = float(v0)
        self.r = float(r)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.lambda_ = float(lambda_)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.alpha = float(alpha)

        # risk-adjusted parameters
        self.kappa_tilde = self.kappa + self.lambda_
        if self.kappa_tilde <= 0:
            raise ValueError("kappa + lambda_ must be > 0")
        self.theta_tilde = (self.kappa * self.theta) / self.kappa_tilde

        # TG-grid is built lazily; not every parameter set produces a sensible grid
        self._tg_grid = None  # populated on first TG/TG-M call

    # ------------------------- TG-scheme helpers ------------------------- #

    @staticmethod
    def _tg_root_equation(rr, psi):
        """Implicit equation defining r(psi) for the TG matching scheme."""
        phi_r = _phi(rr)
        Phi_r = _Phi(rr)
        lhs = rr * phi_r + Phi_r * (1.0 + rr * rr)
        rhs = (1.0 + psi) * (phi_r + rr * Phi_r) ** 2
        return lhs - rhs

    @classmethod
    def _find_rr(cls, psi, bracket_width=10.0):
        """Solve the TG matching equation for r given psi."""
        a, b = -bracket_width, bracket_width
        f_a = cls._tg_root_equation(a, psi)
        f_b = cls._tg_root_equation(b, psi)
        # widen bracket if needed
        while f_a * f_b > 0:
            a *= 2.0
            b *= 2.0
            f_a = cls._tg_root_equation(a, psi)
            f_b = cls._tg_root_equation(b, psi)
            if abs(a) > 1e6:
                raise ValueError("Could not bracket the TG root for psi=%g" % psi)
        sol = root_scalar(
            cls._tg_root_equation, args=(psi,), bracket=[a, b], method="brentq"
        )
        if not sol.converged:
            raise ValueError("Root finding did not converge for psi=%g" % psi)
        return sol.root

    @staticmethod
    def _f_mu(rr):
        return rr / (_phi(rr) + rr * _Phi(rr))

    @staticmethod
    def _f_sigma(psi, rr):
        return (1.0 / np.sqrt(psi)) / (_phi(rr) + rr * _Phi(rr))

    def _build_tg_grid(self, n_grid=200):
        """Pre-compute the TG (psi, r, f_mu, f_sigma) lookup table.

        The grid is built only on the region where the matching equation is
        well-defined.  We span psi in (psi_min, psi_max) with
            psi_min = small positive number,
            psi_max = max(2, sigma^2 / (2 * kappa_tilde * theta_tilde))
        which always brackets the values typically met during simulation.
        """
        psi_min = max(1.0 / (self.alpha * self.alpha), 1e-4)
        psi_max_raw = (self.sigma ** 2) / (2.0 * self.kappa_tilde * self.theta_tilde)
        psi_max = max(psi_max_raw, 2.0, psi_min * 10.0)

        psi_grid = np.linspace(psi_min, psi_max, n_grid)
        r_grid = np.empty_like(psi_grid)
        for k, psi in enumerate(psi_grid):
            try:
                r_grid[k] = self._find_rr(psi)
            except ValueError:
                # leave NaN; will fall back to the moment-matched branch
                r_grid[k] = np.nan

        valid = ~np.isnan(r_grid)
        psi_grid = psi_grid[valid]
        r_grid = r_grid[valid]

        f_mu = np.array([self._f_mu(rr) for rr in r_grid])
        f_sigma = np.array([self._f_sigma(psi, rr) for psi, rr in zip(psi_grid, r_grid)])

        self._tg_grid = (psi_grid, r_grid, f_mu, f_sigma)

    def _tg_lookup(self, psi):
        """Vectorised nearest-neighbour lookup of (f_mu, f_sigma) on the TG grid."""
        if self._tg_grid is None:
            self._build_tg_grid()
        psi_grid, _r_grid, f_mu_grid, f_sigma_grid = self._tg_grid
        idx = np.searchsorted(psi_grid, psi)
        idx = np.clip(idx, 0, len(psi_grid) - 1)
        return f_mu_grid[idx], f_sigma_grid[idx]

    # ------------------------- shared K-constants ------------------------ #

    def _qe_tg_constants(self, dt):
        """Pre-compute the dt-only constants used by QE/QE-M/TG/TG-M."""
        kt, tt = self.kappa_tilde, self.theta_tilde
        rho, sig = self.rho, self.sigma
        g1, g2 = self.gamma1, self.gamma2

        exp_neg = np.exp(-kt * dt)
        K0 = -rho * kt * tt * dt / sig
        K1 = g1 * dt * (-0.5 + (kt * rho) / sig) - rho / sig
        K2 = g2 * dt * (-0.5 + (kt * rho) / sig) + rho / sig
        K3 = g1 * dt * (1.0 - rho * rho)
        K4 = g2 * dt * (1.0 - rho * rho)
        # Coefficient appearing in the martingale correction
        A = (rho / sig) * (1.0 + kt * g2 * dt) - 0.5 * g2 * dt * rho * rho
        return exp_neg, K0, K1, K2, K3, K4, A

    # =====================================================================
    # SIMULATION
    # =====================================================================
    # All simulators are vectorised over `num_paths`.  They return:
    #     S        : (num_paths, n+1)   asset prices
    #     v_zero   : int                total count of zero-variance events
    # =====================================================================

    def simulateEuler(self, T, n, num_paths, rng=None):
        """Euler full-truncation scheme."""
        rng = np.random.default_rng() if rng is None else rng
        dt = T / n
        sqrt_dt = np.sqrt(dt)

        S = np.empty((num_paths, n + 1))
        v = np.empty((num_paths, n + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        v_zero = 0

        Z1 = rng.standard_normal((num_paths, n))
        Z2 = rng.standard_normal((num_paths, n))
        Zv = Z1
        Zs = self.rho * Z1 + np.sqrt(1.0 - self.rho ** 2) * Z2

        kt, tt, sig, r = self.kappa_tilde, self.theta_tilde, self.sigma, self.r

        for i in range(1, n + 1):
            v_prev = v[:, i - 1]
            v_pos = np.maximum(v_prev, 0.0)
            sqrt_v = np.sqrt(v_pos)

            v_new = v_prev + kt * (tt - v_prev) * dt + sig * sqrt_v * sqrt_dt * Zv[:, i - 1]
            v_zero += int(np.sum(v_new <= 0))
            v[:, i] = np.maximum(v_new, 0.0)

            S[:, i] = S[:, i - 1] + r * S[:, i - 1] * dt + sqrt_v * S[:, i - 1] * sqrt_dt * Zs[:, i - 1]

        return S, v_zero

    def simulateMilstein(self, T, n, num_paths, rng=None):
        """Milstein scheme with full truncation."""
        rng = np.random.default_rng() if rng is None else rng
        dt = T / n
        sqrt_dt = np.sqrt(dt)

        S = np.empty((num_paths, n + 1))
        v = np.empty((num_paths, n + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        v_zero = 0

        Z1 = rng.standard_normal((num_paths, n))
        Z2 = rng.standard_normal((num_paths, n))
        Zv = Z1
        Zs = self.rho * Z1 + np.sqrt(1.0 - self.rho ** 2) * Z2

        kt, tt, sig, r = self.kappa_tilde, self.theta_tilde, self.sigma, self.r

        for i in range(1, n + 1):
            v_prev = v[:, i - 1]
            v_pos = np.maximum(v_prev, 0.0)
            sqrt_v = np.sqrt(v_pos)

            v_new = (
                v_prev
                + kt * (tt - v_prev) * dt
                + sig * sqrt_v * sqrt_dt * Zv[:, i - 1]
                + 0.25 * sig ** 2 * dt * (Zv[:, i - 1] ** 2 - 1.0)
            )
            v_zero += int(np.sum(v_new < 0))
            v[:, i] = np.maximum(v_new, 0.0)

            S[:, i] = S[:, i - 1] + r * S[:, i - 1] * dt + sqrt_v * S[:, i - 1] * sqrt_dt * Zs[:, i - 1]

        return S, v_zero

    def simulateQE(self, T, n, num_paths, C=1.5, rng=None):
        """Andersen (2008) Quadratic-Exponential scheme."""
        return self._simulate_qe(T, n, num_paths, C, martingale_correction=False, rng=rng)

    def simulateQEM(self, T, n, num_paths, C=1.5, rng=None):
        """Andersen (2008) QE scheme with martingale correction (QE-M)."""
        return self._simulate_qe(T, n, num_paths, C, martingale_correction=True, rng=rng)

    def _simulate_qe(self, T, n, num_paths, C, martingale_correction, rng):
        rng = np.random.default_rng() if rng is None else rng
        dt = T / n
        exp_neg, K0, K1, K2, K3, K4, A = self._qe_tg_constants(dt)
        kt, tt, sig, r = self.kappa_tilde, self.theta_tilde, self.sigma, self.r

        S = np.empty((num_paths, n + 1))
        v = np.empty((num_paths, n + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        v_zero = 0

        Uv = rng.uniform(size=(num_paths, n))
        Zv = norm.ppf(Uv)
        Zs = rng.standard_normal((num_paths, n))

        log_S = np.full(num_paths, np.log(self.S0))

        for i in range(1, n + 1):
            v_prev = v[:, i - 1]
            m = tt + (v_prev - tt) * exp_neg
            s2 = (
                v_prev * sig ** 2 * exp_neg * (1 - exp_neg) / kt
                + tt * sig ** 2 * (1 - exp_neg) ** 2 / (2 * kt)
            )
            # guard against numerical zero-mean
            psi = np.where(m > 1e-300, s2 / np.where(m > 1e-300, m * m, 1.0), np.inf)

            v_new = np.empty(num_paths)
            K0_path = np.full(num_paths, K0)

            # Branch 1: psi <= C   ->  quadratic representation
            mask_q = psi <= C
            if np.any(mask_q):
                psi_q = psi[mask_q]
                m_q = m[mask_q]
                # numerical safety: psi_q can be tiny -> 2/psi - 1 is huge, fine
                two_over_psi = 2.0 / psi_q
                b2 = two_over_psi - 1.0 + np.sqrt(two_over_psi) * np.sqrt(np.maximum(two_over_psi - 1.0, 0.0))
                a = m_q / (1.0 + b2)
                v_new[mask_q] = a * (np.sqrt(b2) + Zv[mask_q, i - 1]) ** 2

                if martingale_correction:
                    # K0* such that E[exp(log S_{i+1}-log S_i)/(... )] is martingale
                    # Andersen (2008), eq. (33).  Falls back gracefully if 1-2A*a <= 0.
                    one_minus_2Aa = 1.0 - 2.0 * A * a
                    safe = one_minus_2Aa > 1e-12
                    K0star = np.full_like(a, K0)  # fallback to plain K0 when unsafe
                    K0star[safe] = (
                        -(A * b2[safe] * a[safe]) / one_minus_2Aa[safe]
                        + 0.5 * np.log(one_minus_2Aa[safe])
                        - (K1 + 0.5 * K3) * v_prev[mask_q][safe]
                    )
                    K0_path_q = K0_path[mask_q].copy()
                    K0_path_q[safe] = K0star[safe]
                    K0_path[mask_q] = K0_path_q

            # Branch 2: psi > C   ->  exponential representation
            mask_e = ~mask_q
            if np.any(mask_e):
                psi_e = psi[mask_e]
                m_e = m[mask_e]
                p = (psi_e - 1.0) / (psi_e + 1.0)
                # avoid division by zero when m_e is exactly 0
                beta = np.where(m_e > 1e-300, (1.0 - p) / np.where(m_e > 1e-300, m_e, 1.0), np.inf)

                Uv_e = Uv[mask_e, i - 1]
                v_branch = np.where(
                    Uv_e <= p,
                    0.0,
                    np.log(np.maximum((1.0 - p) / np.maximum(1.0 - Uv_e, 1e-300), 1e-300)) / np.where(beta > 0, beta, 1.0),
                )
                v_new[mask_e] = v_branch

                if martingale_correction:
                    # Andersen (2008), eq. (34); fall back if A>=beta.
                    safe = (beta - A) > 1e-12
                    K0star = np.full_like(p, K0)
                    K0star[safe] = (
                        -np.log(p[safe] + (beta[safe] * (1.0 - p[safe])) / (beta[safe] - A))
                        - (K1 + 0.5 * K3) * v_prev[mask_e][safe]
                    )
                    K0_path_e = K0_path[mask_e].copy()
                    K0_path_e[safe] = K0star[safe]
                    K0_path[mask_e] = K0_path_e

            v_zero += int(np.sum(v_new <= 0))
            v[:, i] = np.maximum(v_new, 0.0)

            log_S = (
                log_S
                + r * dt
                + K0_path
                + K1 * v_prev
                + K2 * v[:, i]
                + np.sqrt(np.maximum(K3 * v_prev + K4 * v[:, i], 0.0)) * Zs[:, i - 1]
            )
            S[:, i] = np.exp(log_S)

        return S, v_zero

    def simulateTG(self, T, n, num_paths, rng=None):
        """Truncated-Gaussian scheme."""
        return self._simulate_tg(T, n, num_paths, martingale_correction=False, rng=rng)

    def simulateTGM(self, T, n, num_paths, rng=None):
        """Truncated-Gaussian + Martingale-correction scheme."""
        return self._simulate_tg(T, n, num_paths, martingale_correction=True, rng=rng)

    def _simulate_tg(self, T, n, num_paths, martingale_correction, rng):
        rng = np.random.default_rng() if rng is None else rng
        dt = T / n
        exp_neg, K0, K1, K2, K3, K4, A = self._qe_tg_constants(dt)
        kt, tt, sig, r = self.kappa_tilde, self.theta_tilde, self.sigma, self.r

        S = np.empty((num_paths, n + 1))
        v = np.empty((num_paths, n + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        v_zero = 0

        Uv = rng.uniform(size=(num_paths, n))
        Zv = norm.ppf(Uv)
        Zs = rng.standard_normal((num_paths, n))

        log_S = np.full(num_paths, np.log(self.S0))

        for i in range(1, n + 1):
            v_prev = v[:, i - 1]
            m = tt + (v_prev - tt) * exp_neg
            s2 = (
                v_prev * sig ** 2 * exp_neg * (1 - exp_neg) / kt
                + tt * sig ** 2 * (1 - exp_neg) ** 2 / (2 * kt)
            )
            m_safe = np.where(m > 1e-300, m, 1e-300)
            psi = s2 / (m_safe * m_safe)

            # branch decision: the standard moment-matched branch is used when
            # 1/sqrt(psi) > alpha (i.e. psi small), otherwise the calibrated TG grid.
            mean = np.empty(num_paths)
            sd = np.empty(num_paths)
            mask_simple = (1.0 / np.sqrt(np.maximum(psi, 1e-300))) > self.alpha

            mean[mask_simple] = m[mask_simple]
            sd[mask_simple] = np.sqrt(np.maximum(s2[mask_simple], 0.0))

            if np.any(~mask_simple):
                f_mu, f_sig = self._tg_lookup(psi[~mask_simple])
                mean[~mask_simple] = f_mu * m[~mask_simple]
                sd[~mask_simple] = f_sig * np.sqrt(np.maximum(s2[~mask_simple], 0.0))

            v_new = np.maximum(mean + sd * Zv[:, i - 1], 0.0)
            v_zero += int(np.sum((mean + sd * Zv[:, i - 1]) < 0))
            v[:, i] = v_new

            if martingale_correction:
                # M = E[exp(A*nu_T)] under the truncated Gaussian distribution.
                # Andersen (2008), eq. (37).  Fallback to K0 if mean=0 / SD=0.
                sd_safe = np.where(sd > 1e-12, sd, 1e-12)
                d1 = mean / sd_safe + A * sd_safe
                d2 = mean / sd_safe
                M = np.exp(A * mean + 0.5 * sd_safe ** 2 * A ** 2) * norm.cdf(d1) + norm.cdf(-d2)
                M = np.where(M > 1e-300, M, 1e-300)
                K0_path = -np.log(M) - (K1 + 0.5 * K3) * v_prev
            else:
                K0_path = K0

            log_S = (
                log_S
                + r * dt
                + K0_path
                + K1 * v_prev
                + K2 * v[:, i]
                + np.sqrt(np.maximum(K3 * v_prev + K4 * v[:, i], 0.0)) * Zs[:, i - 1]
            )
            S[:, i] = np.exp(log_S)

        return S, v_zero

    def simulateBroadieKaya(self, T, n, num_paths, rng=None):
        """Broadie-Kaya drift-interpolation almost-exact scheme.

        Samples v_T from its exact non-central chi-squared transition
        distribution, then approximates the integrated variance by a trapezoid
        and the stochastic-integral term by the discretisation of the Heston
        identity.  See Broadie & Kaya (2006).
        """
        rng = np.random.default_rng() if rng is None else rng
        dt = T / n

        kappa, theta, sig, rho, r = self.kappa, self.theta, self.sigma, self.rho, self.r
        d = 4.0 * theta * kappa / (sig ** 2)            # df of nc chi^2
        exp_neg = np.exp(-kappa * dt)
        c = (sig ** 2) * (1.0 - exp_neg) / (4.0 * kappa)

        S = np.empty((num_paths, n + 1))
        v = np.empty((num_paths, n + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        v_zero = 0

        log_S = np.full(num_paths, np.log(self.S0))

        for i in range(1, n + 1):
            v_prev = v[:, i - 1]
            # non-centrality parameter
            nc = v_prev * exp_neg / c

            if d > 1.0:
                # decomposition: chi^2_{d-1} + (Z + sqrt(nc))^2
                Z = rng.standard_normal(num_paths)
                X = chi2.rvs(d - 1.0, size=num_paths, random_state=rng)
                v_new = c * ((Z + np.sqrt(np.maximum(nc, 0.0))) ** 2 + X)
            else:
                # mixture: Poisson(nc/2) draws then chi^2_{d + 2N}
                N_pois = poisson.rvs(0.5 * np.maximum(nc, 0.0), random_state=rng)
                # chi2.rvs broadcasts df elementwise
                v_new = c * chi2.rvs(d + 2.0 * N_pois, random_state=rng)

            v_zero += int(np.sum(v_new <= 1e-12))
            v[:, i] = np.maximum(v_new, 0.0)

            # trapezoid for ∫ v_s ds and the Heston identity for ∫ sqrt(v_s) dW^v
            integrated = 0.5 * (v_prev + v[:, i]) * dt
            sqrt_int_dW = (1.0 / sig) * (
                v[:, i] - v_prev - kappa * theta * dt + kappa * integrated
            )

            mu = r * dt - 0.5 * integrated + rho * sqrt_int_dW
            var_term = (1.0 - rho ** 2) * integrated
            Z_s = rng.standard_normal(num_paths)
            log_S = log_S + mu + np.sqrt(np.maximum(var_term, 0.0)) * Z_s
            S[:, i] = np.exp(log_S)

        return S, v_zero

    # =====================================================================
    # MONTE CARLO PRICING
    # =====================================================================

    _SIM_TABLE = {
        "Euler": "simulateEuler",
        "Milstein": "simulateMilstein",
        "QE": "simulateQE",
        "QEM": "simulateQEM",
        "TG": "simulateTG",
        "TGM": "simulateTGM",
        "BroadieKaya": "simulateBroadieKaya",
        # legacy aliases used by older drivers
        "EulerDisc": "simulateEuler",
        "MilsteinDisc": "simulateMilstein",
        "QEDisc": "simulateQE",
        "QEMDisc": "simulateQEM",
        "TGDisc": "simulateTG",
        "TGMDisc": "simulateTGM",
    }

    def priceCallMC(self, K, T, n, num_paths, method, rng=None, conf_level=0.95):
        """Monte Carlo price of a European call option.

        Parameters
        ----------
        K, T : float
            Strike and maturity.
        n : int
            Number of timesteps.
        num_paths : int
            Number of MC paths.
        method : str
            One of {Euler, Milstein, QE, QEM, TG, TGM, BroadieKaya}.
        rng : np.random.Generator or None
        conf_level : float
            Confidence level for the half-width of the CI (default 95%).

        Returns
        -------
        price : float
        std_err : float
        ci_half_width : float
            Half-width of the confidence interval (price +/- ci_half_width).
        v_zero_count : int
        """
        if method not in self._SIM_TABLE:
            raise ValueError(
                f"Unknown method {method!r}; valid: {sorted(set(self._SIM_TABLE))}"
            )
        simulate = getattr(self, self._SIM_TABLE[method])
        S, v_zero = simulate(T, n, num_paths, rng=rng)

        payoffs = np.maximum(S[:, -1] - K, 0.0)
        disc_payoffs = np.exp(-self.r * T) * payoffs
        price = float(np.mean(disc_payoffs))
        std_err = float(np.std(disc_payoffs, ddof=1) / np.sqrt(num_paths))
        z = norm.ppf(0.5 + conf_level / 2.0)
        ci_half = z * std_err
        return price, std_err, ci_half, v_zero

    # backwards-compatible API
    def priceHestonCallViaMC(self, K, T, n, num_paths, method, rng=None):
        """Legacy wrapper. Returns (price, v_zero_count, ci_half_width)."""
        price, _stderr, ci, v_zero = self.priceCallMC(K, T, n, num_paths, method, rng=rng)
        return price, v_zero, ci

    # =====================================================================
    # EXACT / SEMI-ANALYTIC PRICING
    # =====================================================================

    def _char_fn_heston(self, u, T, j, formulation="original"):
        """Heston conditional characteristic function of log(S_T).

        formulation : {'original', 'stable'}
            'original'  reproduces Heston (1993) -- numerically unstable for
                        large maturities due to branch-cut crossings of log(.)
            'stable'    Albrecher et al. (2007) ``Little Trap'' formulation
                        which keeps the integrand smooth.
        """
        i = 1j
        kappa, theta, sigma, rho, lam = (
            self.kappa, self.theta, self.sigma, self.rho, self.lambda_
        )
        a = kappa * theta
        b1 = kappa + lam - rho * sigma
        b2 = kappa + lam
        u1, u2 = 0.5, -0.5
        b_j = b1 if j == 1 else b2
        u_j = u1 if j == 1 else u2

        d = np.sqrt((rho * sigma * i * u - b_j) ** 2 - sigma ** 2 * (2.0 * u_j * i * u - u ** 2))

        if formulation == "original":
            g = (b_j - rho * sigma * i * u + d) / (b_j - rho * sigma * i * u - d)
            C = self.r * i * u * T + (a / sigma ** 2) * (
                (b_j - rho * sigma * i * u + d) * T
                - 2.0 * np.log((1.0 - g * np.exp(d * T)) / (1.0 - g))
            )
            D = ((b_j - rho * sigma * i * u + d) / sigma ** 2) * (
                (1.0 - np.exp(d * T)) / (1.0 - g * np.exp(d * T))
            )
        elif formulation == "stable":
            # The "Little Trap" rewrites g -> 1/g and d -> -d so that the
            # exponential is exp(-dT) and stays bounded.
            g = (b_j - rho * sigma * i * u - d) / (b_j - rho * sigma * i * u + d)
            C = self.r * i * u * T + (a / sigma ** 2) * (
                (b_j - rho * sigma * i * u - d) * T
                - 2.0 * np.log((1.0 - g * np.exp(-d * T)) / (1.0 - g))
            )
            D = ((b_j - rho * sigma * i * u - d) / sigma ** 2) * (
                (1.0 - np.exp(-d * T)) / (1.0 - g * np.exp(-d * T))
            )
        else:
            raise ValueError(f"Unknown formulation: {formulation}")

        return np.exp(C + D * self.v0 + i * u * np.log(self.S0))

    def priceCallFourier(self, K, T, formulation="stable", upper=200.0):
        """Heston Fourier-inversion price of a European call (P1, P2 form)."""
        i = 1j

        def integrand(u, j):
            phi = self._char_fn_heston(u, T, j, formulation=formulation)
            return np.real(np.exp(-i * u * np.log(K)) * phi / (i * u))

        Q1, e1 = quad(integrand, 1e-12, upper, args=(1,), limit=200)
        Q2, e2 = quad(integrand, 1e-12, upper, args=(2,), limit=200)
        P1 = 0.5 + Q1 / np.pi
        P2 = 0.5 + Q2 / np.pi
        price = self.S0 * P1 - np.exp(-self.r * T) * K * P2
        err = (e1 + e2) / np.pi
        return price, err

    def priceCallCarrMadan(self, K, T, alpha=1.5, upper=200.0, formulation="stable"):
        """Carr-Madan damped-exponential European-call price.

        Uses the unconditional characteristic function of log(S_T)
        (j=2 in the P1/P2 decomposition).
        """
        i = 1j

        def integrand(u):
            phi_shift = self._char_fn_heston(
                u - i * (alpha + 1.0), T, j=2, formulation=formulation
            )
            num = np.exp(-self.r * T) * np.exp(-i * u * np.log(K)) * phi_shift
            den = alpha ** 2 + alpha - u ** 2 + i * (2.0 * alpha + 1.0) * u
            return np.real(num / den)

        integral, err = quad(integrand, 0.0, upper, limit=200)
        price = np.exp(-alpha * np.log(K)) * integral / np.pi
        return price, err / np.pi
