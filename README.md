# MCHeston

[![CI](https://github.com/YoussefRaad-mathecon/MCHeston/actions/workflows/ci.yml/badge.svg)](https://github.com/YoussefRaad-mathecon/MCHeston/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/MCHeston.svg)](https://pypi.org/project/MCHeston/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Heston stochastic-volatility model: Monte Carlo simulation and option pricing
in pure Python (NumPy / SciPy).

## Features

* **Seven simulation schemes**, all path-vectorised:
  Euler, Milstein, Quadratic-Exponential (QE), QE + martingale-correction (QE-M),
  Truncated-Gaussian (TG), TG + martingale-correction (TG-M), and Broadie-Kaya
  drift-interpolation.
* **Three exact / semi-analytic pricers**: Heston (1993) original Fourier
  inversion, the Albrecher et al. "Little Trap" stable formulation, and
  Carr-Madan damped-exponential.
* All exact methods agree to within 1e-10 on the Andersen (2008) reference
  prices.

## Installation

```bash
pip install MCHeston
```

For development (tests, build tooling):

```bash
pip install MCHeston[dev]
```

## Quick start

```python
import numpy as np
from mcheston import HestonModel

# Define the model
model = HestonModel(
    S0=100,         # spot price
    v0=0.04,        # initial variance
    r=0.0,          # risk-free rate
    kappa=0.5,      # variance mean-reversion speed
    theta=0.04,     # long-run variance
    sigma=1.0,      # volatility of variance
    rho=-0.9,       # correlation
    lambda_=0.0,    # market price of vol risk
)

# Exact price via stable Fourier inversion
price, err = model.priceCallFourier(K=100, T=10.0, formulation="stable")
print(f"Exact price: {price:.4f}")

# Monte Carlo with the recommended QE-M scheme
rng = np.random.default_rng(seed=42)
price, std_err, ci_half, v_zero_count = model.priceCallMC(
    K=100, T=10.0, n=80, num_paths=100_000, method="QEM", rng=rng,
)
print(f"MC price: {price:.4f} +/- {ci_half:.4f}  (95% CI)")
```

## API at a glance

```python
HestonModel(S0, v0, r, kappa, theta, sigma, rho, lambda_=0.0,
            gamma1=0.5, gamma2=0.5, alpha=4.5)

# Path simulation - all return (S_paths, v_zero_count)
model.simulateEuler(T, n, num_paths, rng=None)
model.simulateMilstein(T, n, num_paths, rng=None)
model.simulateQE(T, n, num_paths, C=1.5, rng=None)
model.simulateQEM(T, n, num_paths, C=1.5, rng=None)
model.simulateTG(T, n, num_paths, rng=None)
model.simulateTGM(T, n, num_paths, rng=None)
model.simulateBroadieKaya(T, n, num_paths, rng=None)

# Monte Carlo pricing
model.priceCallMC(K, T, n, num_paths, method, rng=None, conf_level=0.95)
# returns: (price, std_err, ci_half_width, v_zero_count)

# Exact pricing
model.priceCallFourier(K, T, formulation="stable", upper=200.0)
# returns: (price, numerical_error)
model.priceCallCarrMadan(K, T, alpha=1.5, upper=200.0, formulation="stable")
# returns: (price, numerical_error)
```

`method` for `priceCallMC` is one of:
`"Euler"`, `"Milstein"`, `"QE"`, `"QEM"`, `"TG"`, `"TGM"`, `"BroadieKaya"`.

## When to use which scheme

* **QE-M** is the recommended default. Fast, accurate, and robust to Feller
  violations.
* **Broadie-Kaya** has the smallest discretisation bias (it samples v_T from
  its exact non-central chi-squared distribution) but is slower per step.
* **Euler / Milstein** should not be used when the Feller condition
  `2*kappa*theta >= sigma^2` is violated - they pick up large positive bias
  from the variance hitting zero.

## References

* Heston, S. L. (1993). *A closed-form solution for options with stochastic
  volatility with applications to bond and currency options*. Review of
  Financial Studies, 6(2), 327-343.
* Andersen, L. B. G. (2008). *Simple and efficient simulation of the Heston
  stochastic volatility model*. Journal of Computational Finance, 11(3), 1-42.
* Broadie, M., & Kaya, O. (2006). *Exact simulation of stochastic volatility
  and other affine jump diffusion processes*. Operations Research, 54(2),
  217-231.
* Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
  *The little Heston trap*. Wilmott Magazine, January, 83-92.
* Carr, P., & Madan, D. (1999). *Option valuation using the fast Fourier
  transform*. Journal of Computational Finance, 2(4), 61-73.

## License

MIT. See [LICENSE](LICENSE).

## Author

Youssef Raad ([@YoussefRaad-mathecon](https://github.com/YoussefRaad-mathecon))
