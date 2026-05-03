"""MCHeston: Heston stochastic-volatility option pricing.

Public API:
    HestonModel  -- main model class with simulation and pricing methods
"""

from mcheston.model import HestonModel

__version__ = "0.1.0"
__all__ = ["HestonModel", "__version__"]
