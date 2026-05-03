"""Tests for MCHeston.

Run with:
    pytest

After installing the package (`pip install -e .[dev]`), no sys.path
manipulation is needed - `mcheston` is importable directly.
"""

import numpy as np
import pytest

from mcheston import HestonModel


# ---------------------------------------------------------------------------
# Reference values  (Andersen 2008 Case II, T = 10)
# Computed with the stable Fourier formulation.
# ---------------------------------------------------------------------------
ANDERSEN_CASE_II = dict(
    S0=100, v0=0.04, r=0.0, kappa=0.5, theta=0.04,
    sigma=1.0, rho=-0.9, lambda_=0.0,
)
ANDERSEN_T = 10.0
ANDERSEN_REF = {
    60: 44.329975068269974,
    100: 13.084670136959673,
    140: 0.295774435299149,
}


@pytest.fixture
def model():
    return HestonModel(**ANDERSEN_CASE_II)


# ---------------------------------------------------------------------------
# Exact pricers must agree with each other and with the literature reference.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K, expected", ANDERSEN_REF.items())
def test_stable_fourier_matches_reference(model, K, expected):
    price, err = model.priceCallFourier(K, ANDERSEN_T, formulation="stable")
    assert price == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("K", [60, 100, 140])
def test_carr_madan_matches_fourier(model, K):
    p_fourier, _ = model.priceCallFourier(K, ANDERSEN_T, formulation="stable")
    p_cm, _ = model.priceCallCarrMadan(K, ANDERSEN_T)
    assert p_cm == pytest.approx(p_fourier, abs=1e-6)


def test_original_formulation_works_for_short_maturity():
    """The original formulation is fine for T<=1."""
    m = HestonModel(S0=100, v0=0.06, r=0.05, kappa=2.0, theta=0.06,
                    sigma=0.3, rho=-0.5, lambda_=0.0)
    p_orig, _ = m.priceCallFourier(100, 1.0, formulation="original")
    p_stab, _ = m.priceCallFourier(100, 1.0, formulation="stable")
    assert p_orig == pytest.approx(p_stab, abs=1e-6)


# ---------------------------------------------------------------------------
# MC schemes must converge to the exact price within their CI.
# ---------------------------------------------------------------------------

EASY_PARAMS = dict(
    S0=100, v0=0.06, r=0.05, kappa=2.0, theta=0.06,
    sigma=0.3, rho=-0.5, lambda_=0.0,
)
EASY_T = 1.0
EASY_K = 100.0


@pytest.mark.parametrize("method", [
    "Euler", "Milstein", "QE", "QEM", "TG", "TGM", "BroadieKaya",
])
def test_mc_within_ci_under_feller(method):
    """All schemes should price within 95% CI when Feller condition holds."""
    m = HestonModel(**EASY_PARAMS)
    ref, _ = m.priceCallFourier(EASY_K, EASY_T, formulation="stable")
    rng = np.random.default_rng(seed=42)
    price, std_err, ci, _vz = m.priceCallMC(EASY_K, EASY_T, n=50, num_paths=50_000,
                                             method=method, rng=rng)
    assert abs(price - ref) <= 3.0 * std_err, \
        f"{method}: price={price:.4f} ref={ref:.4f} se={std_err:.4f}"


@pytest.mark.parametrize("method", ["QE", "QEM", "BroadieKaya"])
def test_mc_under_feller_violation(method):
    """QE/QE-M/BK are the schemes that should still work when Feller is violated."""
    m = HestonModel(**ANDERSEN_CASE_II)
    K = 100
    ref = ANDERSEN_REF[K]
    rng = np.random.default_rng(seed=42)
    price, std_err, ci, _vz = m.priceCallMC(K, ANDERSEN_T, n=80, num_paths=50_000,
                                             method=method, rng=rng)
    assert abs(price - ref) <= 4.0 * std_err, \
        f"{method}: price={price:.4f} ref={ref:.4f} se={std_err:.4f}"


# ---------------------------------------------------------------------------
# Sanity checks on the simulator output.
# ---------------------------------------------------------------------------

def test_simulator_shapes():
    m = HestonModel(**EASY_PARAMS)
    rng = np.random.default_rng(0)
    S, v_zero = m.simulateQE(T=1.0, n=50, num_paths=200, rng=rng)
    assert S.shape == (200, 51)
    assert isinstance(v_zero, int)


def test_legacy_api_still_works():
    """priceHestonCallViaMC must keep its 3-tuple return signature."""
    m = HestonModel(**EASY_PARAMS)
    rng = np.random.default_rng(0)
    price, vz, ci = m.priceHestonCallViaMC(EASY_K, EASY_T, 50, 5_000, "QE", rng=rng)
    assert isinstance(price, float)
    assert ci > 0


def test_unknown_method_raises():
    m = HestonModel(**EASY_PARAMS)
    with pytest.raises(ValueError):
        m.priceCallMC(100, 1.0, 10, 100, "NotAScheme")


def test_zero_T_returns_intrinsic():
    """As T -> 0, the call price tends to max(S0-K, 0)."""
    m = HestonModel(**EASY_PARAMS)
    p, _ = m.priceCallFourier(K=80.0, T=1e-4, formulation="stable")
    assert p == pytest.approx(20.0, abs=0.5)


def test_package_version():
    import mcheston
    assert hasattr(mcheston, "__version__")
    assert isinstance(mcheston.__version__, str)
