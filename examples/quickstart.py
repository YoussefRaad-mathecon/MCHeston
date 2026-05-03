"""Quickstart example for MCHeston.

Run with:
    python examples/quickstart.py
"""

import numpy as np

from mcheston import HestonModel


def main():
    # Andersen (2008) Case II - the classic stress-test parameters
    # (Feller condition is violated: 2*kappa*theta = 0.04 < sigma^2 = 1.0)
    model = HestonModel(
        S0=100,
        v0=0.04,
        r=0.0,
        kappa=0.5,
        theta=0.04,
        sigma=1.0,
        rho=-0.9,
        lambda_=0.0,
    )

    K = 100
    T = 10.0

    # 1. Exact (semi-analytic) price via Fourier inversion
    print("=== Exact prices ===")
    p_fourier, err = model.priceCallFourier(K, T, formulation="stable")
    print(f"  Fourier (stable):  {p_fourier:.6f}  (numerical error ~ {err:.1e})")

    p_cm, err = model.priceCallCarrMadan(K, T)
    print(f"  Carr-Madan:        {p_cm:.6f}  (numerical error ~ {err:.1e})")

    # 2. Monte Carlo prices using each scheme
    print("\n=== Monte Carlo prices (100k paths, 80 steps) ===")
    print(f"{'method':<14}{'price':>10}{'95% CI half':>14}{'bias':>10}")
    print("-" * 50)

    rng_seed = 42
    for method in ["Euler", "Milstein", "QE", "QEM", "TG", "TGM", "BroadieKaya"]:
        rng = np.random.default_rng(rng_seed)
        price, std_err, ci, v_zero = model.priceCallMC(
            K=K, T=T, n=80, num_paths=100_000, method=method, rng=rng,
        )
        bias = price - p_fourier
        print(f"{method:<14}{price:>10.4f}{ci:>14.4f}{bias:>+10.4f}")


if __name__ == "__main__":
    main()
