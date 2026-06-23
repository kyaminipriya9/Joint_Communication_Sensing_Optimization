"""
Communication-only LDT + nonhomogeneous-QT / D-lambda optimizer.

Use this as a separate module from your main script.

In your main script, import it as:

    from comm_lambda_qt import (
        run_communication_only_lambda_qt,
        run_classical_lambda_qt_comparison,
        plot_comm_only_method_comparison,
    )

This module intentionally does NOT import your main script. Instead, you pass
function handles from your existing code to avoid circular imports:

    compute_sinr_exact_fn=compute_SINR_exact
    compute_wsr_exact_fn=compute_WSR_exact
    update_auxiliary_fn=update_auxiliary
    classical_runner_fn=run_communication_only_optimisation  # only for comparison

Variables optimized:
    P_bar[k]   : communication pilot/channel-estimation power
    P_tilda[k] : communication data power

Fixed in this communication-only version:
    P_prime_fixed[k] : sensing power, or zeros if sensing is disabled

Constraint:
    P_bar[k] + P_tilda[k] + P_prime_fixed[k] <= P_total_max[k]

Main idea:
    Build the LDT-correct D/lambda values every iteration, but update the
    powers with the same direct-P gradient structure used in unfolding.

    This normalized version applies the same max-gradient normalization used
    by the communication unfolding code before the lambda step:

        grad_norm = Pmax / (K * max(abs(grad))) * grad
        P_bar     <- P_bar   + (1/lambda_bar)   * grad_norm_bar
        P_tilda   <- P_tilda + (1/lambda_tilda) * grad_norm_tilda

    The analytical lambda is still saved as alpha = 1/lambda. The actual
    per-user multiplier applied to the raw gradient is also stored as
    effective_alpha = alpha * normalization_scale.

    P_prime is kept fixed for this communication-only version. For pure
    communication-only experiments, pass P_prime_fixed = zeros(K).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


Array = np.ndarray


# -----------------------------------------------------------------------------
# Utility helpers
# INTERNAL HELPER: not called directly by the main file, but used by clipping and normalization to handle scalar/vector Pmax consistently.
# -----------------------------------------------------------------------------
def _as_power_vector(P_total_max: Optional[Any], K: int) -> Optional[Array]:
    """Return P_total_max as a length-K float vector."""
    if P_total_max is None:
        return None
    if np.isscalar(P_total_max):
        return float(P_total_max) * np.ones(K, dtype=float)
    return np.asarray(P_total_max, dtype=float).reshape(K)


# -----------------------------------------------------------------------------
# Utility helpers
# OPTIONAL (not used in main file): safe square-root helper retained for numerical experiments; not used by the current active call path.
# -----------------------------------------------------------------------------
def _safe_sqrt_np(x: Any, eps: float = 1e-12) -> Array:
    """Numerically safe elementwise sqrt."""
    return np.sqrt(np.maximum(np.asarray(x, dtype=float), eps))


# -----------------------------------------------------------------------------
# Gradient normalization
# INTERNAL HELPER: normalizes P_bar/P_tilda gradients before applying the lambda step.
# -----------------------------------------------------------------------------
def normalize_single_power_grad_np(
    grad: Any,
    P_total_max: Optional[Any],
    K: int,
    eps: float = 1e-12,
) -> tuple[Array, Array, float]:
    """
    Normalize one direct-power gradient block like the communication unfolding code.

    For scalar Pmax, this is exactly:

        grad_norm = Pmax / (K * max(abs(grad))) * grad

    For vector Pmax, each user receives its own Pmax[k] in the scaling factor,
    while the max-gradient denominator is still computed over the whole block.

    Returns
    -------
    grad_norm : ndarray, shape (K,)
        The normalized gradient used by the lambda update.
    scale : ndarray, shape (K,)
        The elementwise multiplier such that grad_norm = scale * grad.
    rho : float
        max(abs(grad)) after eps protection.
    """
    grad_np = np.asarray(grad, dtype=float).reshape(K)
    rho = float(max(np.max(np.abs(grad_np)), eps))

    Pmax = _as_power_vector(P_total_max, K)
    if Pmax is None:
        # Without a power budget, the unfolding-style normalization is undefined.
        # Keep the raw gradient and report unit scale.
        scale = np.ones(K, dtype=float)
        return grad_np.copy(), scale, rho

    scale = np.asarray(Pmax, dtype=float).reshape(K) / (rho * float(K))
    return scale * grad_np, scale, rho


# -----------------------------------------------------------------------------
# History/printing helpers
# INTERNAL HELPER: formats stored histories for summaries and text-file export.
# -----------------------------------------------------------------------------
def _flatten_history(arr: Any) -> Array:
    """Return history array with shape [iters, -1] for compact printing."""
    arr_np = np.asarray(arr, dtype=float)
    if arr_np.size == 0:
        return np.empty((0, 0), dtype=float)
    if arr_np.ndim == 0:
        return arr_np.reshape(1, 1)
    if arr_np.ndim == 1:
        return arr_np.reshape(-1, 1)
    return arr_np.reshape(arr_np.shape[0], -1)


# -----------------------------------------------------------------------------
# History/printing helpers
# INTERNAL HELPER: builds compact min/mean/max lines for alpha/lambda histories.
# -----------------------------------------------------------------------------
def _history_stats_line(name: str, hist: Any) -> str:
    """Compact min/mean/max line for a stored step/lambda history."""
    H = _flatten_history(hist)
    if H.size == 0:
        return f"{name:<18}: empty"
    return (
        f"{name:<18}: "
        f"first_mean={H[0].mean():.6e}, "
        f"last_mean={H[-1].mean():.6e}, "
        f"min={H.min():.6e}, max={H.max():.6e}"
    )


# -----------------------------------------------------------------------------
# Public reporting API
# MAIN-FILE API: imported by the main file to print communication lambda step-size summaries.
# -----------------------------------------------------------------------------
def print_lambda_step_size_summary(
    result: Dict[str, Any],
    *,
    label: str = "D/lambda QT step-size summary",
    last_n: int = 10,
) -> None:
    """
    Print alpha and lambda histories from run_communication_only_lambda_qt.

    alpha = 1 / lambda is the actual direct-P step size used by the
    D/lambda communication optimizer.
    """
    alpha_bar = _flatten_history(result.get("alpha_bar_history", []))
    alpha_tilda = _flatten_history(result.get("alpha_tilda_history", []))
    lambda_bar = _flatten_history(result.get("lambda_bar_history", []))
    lambda_tilda = _flatten_history(result.get("lambda_tilda_history", []))

    print("\n" + "=" * 96)
    print(label)
    print("=" * 96)
    print(_history_stats_line("alpha_bar", alpha_bar))
    print(_history_stats_line("alpha_tilda", alpha_tilda))
    print(_history_stats_line("lambda_bar", lambda_bar))
    print(_history_stats_line("lambda_tilda", lambda_tilda))

    if alpha_bar.size == 0 or alpha_tilda.size == 0:
        return

    n_iter = alpha_bar.shape[0]
    start = max(0, n_iter - int(last_n))
    print("\nLast step-size values, averaged over users:")
    print(
        f"{'iter':>6} | {'mean alpha_bar':>16} | {'mean alpha_tilda':>18} | "
        f"{'mean lambda_bar':>17} | {'mean lambda_tilda':>19}"
    )
    print("-" * 88)
    for i in range(start, n_iter):
        lb_mean = lambda_bar[i].mean() if lambda_bar.size else float("nan")
        lt_mean = lambda_tilda[i].mean() if lambda_tilda.size else float("nan")
        print(
            f"{i + 1:6d} | {alpha_bar[i].mean():16.6e} | {alpha_tilda[i].mean():18.6e} | "
            f"{lb_mean:17.6e} | {lt_mean:19.6e}"
        )


# -----------------------------------------------------------------------------
# Public reporting API
# MAIN-FILE API: imported by the main file to save communication lambda step-size summaries.
# -----------------------------------------------------------------------------
def save_lambda_step_size_summary(
    result: Dict[str, Any],
    save_path: str,
) -> None:
    """
    Save per-iteration mean alpha/lambda values to a text file.

    Columns:
        iter, mean_alpha_bar, mean_alpha_tilda, mean_lambda_bar, mean_lambda_tilda
    """
    alpha_bar = _flatten_history(result.get("alpha_bar_history", []))
    alpha_tilda = _flatten_history(result.get("alpha_tilda_history", []))
    lambda_bar = _flatten_history(result.get("lambda_bar_history", []))
    lambda_tilda = _flatten_history(result.get("lambda_tilda_history", []))

    if alpha_bar.size == 0 or alpha_tilda.size == 0:
        raise ValueError("No alpha history found. Run lambda-QT before saving step sizes.")

    n_iter = alpha_bar.shape[0]
    table = np.column_stack(
        [
            np.arange(1, n_iter + 1),
            alpha_bar.mean(axis=1),
            alpha_tilda.mean(axis=1),
            lambda_bar.mean(axis=1),
            lambda_tilda.mean(axis=1),
        ]
    )
    np.savetxt(
        save_path,
        table,
        header="iter mean_alpha_bar mean_alpha_tilda mean_lambda_bar mean_lambda_tilda",
        fmt=["%d", "%.12e", "%.12e", "%.12e", "%.12e"],
    )
    print(f"[SAVE] lambda/D step-size summary: {save_path}")


# -----------------------------------------------------------------------------
# Power-constraint clipping
# INTERNAL HELPER: clips P_bar under the per-user power budget with fixed P_tilda and fixed P_prime.
# -----------------------------------------------------------------------------
def clip_P_bar_joint(
    P_bar_new: Any,
    P_tilda: Any,
    P_prime: Any,
    P_total_max: Optional[Any],
    eps: float = 1e-12,
) -> Array:
    """Clip P_bar for fixed P_tilda and P_prime."""
    P_bar_new = np.asarray(P_bar_new, dtype=float).reshape(-1)
    if P_total_max is None:
        return np.maximum(P_bar_new, eps)

    K = len(P_bar_new)
    Pmax = _as_power_vector(P_total_max, K)
    assert Pmax is not None
    upper = np.maximum(Pmax - np.asarray(P_tilda, dtype=float) - np.asarray(P_prime, dtype=float) - eps, eps)
    return np.clip(P_bar_new, eps, upper)


# -----------------------------------------------------------------------------
# Power-constraint clipping
# INTERNAL HELPER: clips P_tilda under the per-user power budget with fixed P_bar and fixed P_prime.
# -----------------------------------------------------------------------------
def clip_P_tilda_joint(
    P_tilda_new: Any,
    P_bar: Any,
    P_prime: Any,
    P_total_max: Optional[Any],
    eps: float = 1e-12,
) -> Array:
    """Clip P_tilda for fixed P_bar and P_prime."""
    P_tilda_new = np.asarray(P_tilda_new, dtype=float).reshape(-1)
    if P_total_max is None:
        return np.maximum(P_tilda_new, eps)

    K = len(P_tilda_new)
    Pmax = _as_power_vector(P_total_max, K)
    assert Pmax is not None
    upper = np.maximum(Pmax - np.asarray(P_bar, dtype=float) - np.asarray(P_prime, dtype=float) - eps, eps)
    return np.clip(P_tilda_new, eps, upper)


# -----------------------------------------------------------------------------
# State evaluation
# INTERNAL HELPER: evaluates SINR, WSR, P_th and P_adc for the current communication powers.
# -----------------------------------------------------------------------------
def evaluate_comm_only_state(
    *,
    K: int,
    w: Any,
    d: float,
    tau: float,
    kappa_S: Any,
    kappa_V1: Any,
    kappa_K1: Any,
    kappa_V0: Any,
    kappa_K0: Any,
    kappa_DAC1: Any,
    kappa_DAC0: Any,
    kappa_Th1: Any,
    kappa_ADC1: Any,
    kappa_M1: Any,
    kappa_M0: Any,
    kappa_Th0: Any,
    kappa_ADC0: Any,
    P_bar: Any,
    P_tilda: Any,
    compute_sinr_exact_fn: Callable[..., float],
    compute_wsr_exact_fn: Callable[..., float],
) -> Dict[str, Any]:
    """Evaluate communication-only WSR, SINR, P_th and P_adc."""
    P_bar = np.asarray(P_bar, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda, dtype=float).reshape(K)
    w = np.asarray(w, dtype=float).reshape(K)

    kappa_S = np.asarray(kappa_S, dtype=float).reshape(K)
    kappa_V1 = np.asarray(kappa_V1, dtype=float).reshape(K)
    kappa_K1 = np.asarray(kappa_K1, dtype=float).reshape(K)
    kappa_V0 = np.asarray(kappa_V0, dtype=float).reshape(K)
    kappa_K0 = np.asarray(kappa_K0, dtype=float).reshape(K)
    kappa_DAC1 = np.asarray(kappa_DAC1, dtype=float).reshape(K)
    kappa_DAC0 = np.asarray(kappa_DAC0, dtype=float).reshape(K)
    kappa_Th1 = np.asarray(kappa_Th1, dtype=float).reshape(K)
    kappa_ADC1 = np.asarray(kappa_ADC1, dtype=float).reshape(K)
    kappa_Th0 = np.asarray(kappa_Th0, dtype=float).reshape(K)
    kappa_ADC0 = np.asarray(kappa_ADC0, dtype=float).reshape(K)
    kappa_M1 = np.asarray(kappa_M1, dtype=float)
    kappa_M0 = np.asarray(kappa_M0, dtype=float)

    P_th = P_bar * kappa_Th1 + kappa_Th0
    P_adc = P_bar * kappa_ADC1 + kappa_ADC0

    gamma = np.array(
        [
            compute_sinr_exact_fn(
                k,
                K,
                kappa_S,
                kappa_V1,
                kappa_K1,
                kappa_V0,
                kappa_K0,
                kappa_DAC1,
                kappa_DAC0,
                kappa_M1,
                kappa_M0,
                kappa_Th1,
                kappa_ADC1,
                P_bar,
                P_tilda,
                P_th,
                P_adc,
            )
            for k in range(K)
        ],
        dtype=float,
    )

    comm_wsr = compute_wsr_exact_fn(
        K,
        w,
        kappa_S,
        kappa_V1,
        kappa_K1,
        kappa_V0,
        kappa_K0,
        kappa_DAC1,
        kappa_DAC0,
        kappa_M1,
        kappa_M0,
        kappa_Th1,
        kappa_ADC1,
        P_bar,
        P_tilda,
        P_th,
        P_adc,
        d,
        tau,
    )

    return {
        "comm_wsr": float(comm_wsr),
        "gamma": gamma,
        "P_th": P_th,
        "P_adc": P_adc,
    }


# -----------------------------------------------------------------------------
# D/lambda construction
# INTERNAL HELPER: builds the D matrix and lambda values for the P_bar update.
# -----------------------------------------------------------------------------
def build_D_lambda_P_bar(
    *,
    K: int,
    w: Any,
    gamma: Any,
    mu: Any,
    P_bar: Any,
    P_tilda: Any,
    kappa_S: Any,
    kappa_V1: Any,
    kappa_K1: Any,
    kappa_V0: Any,
    kappa_K0: Any,
    kappa_DAC1: Any,
    kappa_DAC0: Any,
    kappa_Th1: Any,
    kappa_ADC1: Any,
    kappa_M1: Any,
    kappa_M0: Any,
    kappa_Th0: Any,
    kappa_ADC0: Any,
    eps_lambda: float = 1e-30,
    lambda_mode: str = "actual",
) -> Dict[str, Array]:
    """
    Build D matrix and lambda values for the P_bar update.

    Variable:
        x_bar[k] = sqrt(P_bar[k]).

    For fixed P_tilda, after LDT the ratio is

        (a_bar[k] * x_bar[k]^2)
        / ((a_bar[k] + b_bar[k]) * x_bar[k]^2 + c_bar[k]).

    The augmented D matrix is

        D_bar_aug[k] = mu[k]^2 * diag(a_bar[k] + b_bar[k], c_bar[k]).

    The actual scalar optimized coordinate uses only the first diagonal entry:

        D_bar_update[k] = mu[k]^2 * (a_bar[k] + b_bar[k]).

    lambda_mode:
        actual  : lambda = D_11. Recommended for scalar P_bar update.
        aug_max : lambda = max(D_11, D_22). Conservative.
        aug_fro : lambda = ||D_aug||_F. Conservative.
    """
    P_bar = np.asarray(P_bar, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda, dtype=float).reshape(K)
    w = np.asarray(w, dtype=float).reshape(K)
    gamma = np.asarray(gamma, dtype=float).reshape(K)
    mu = np.asarray(mu, dtype=float).reshape(K)

    kappa_S = np.asarray(kappa_S, dtype=float).reshape(K)
    kappa_V1 = np.asarray(kappa_V1, dtype=float).reshape(K)
    kappa_K1 = np.asarray(kappa_K1, dtype=float).reshape(K)
    kappa_V0 = np.asarray(kappa_V0, dtype=float).reshape(K)
    kappa_K0 = np.asarray(kappa_K0, dtype=float).reshape(K)
    kappa_DAC1 = np.asarray(kappa_DAC1, dtype=float).reshape(K)
    kappa_DAC0 = np.asarray(kappa_DAC0, dtype=float).reshape(K)
    kappa_Th1 = np.asarray(kappa_Th1, dtype=float).reshape(K)
    kappa_ADC1 = np.asarray(kappa_ADC1, dtype=float).reshape(K)
    kappa_Th0 = np.asarray(kappa_Th0, dtype=float).reshape(K)
    kappa_ADC0 = np.asarray(kappa_ADC0, dtype=float).reshape(K)
    kappa_M1 = np.asarray(kappa_M1, dtype=float)
    kappa_M0 = np.asarray(kappa_M0, dtype=float)

    rho = w * (1.0 + gamma)

    a_bar = np.zeros(K, dtype=float)
    b_bar = np.zeros(K, dtype=float)
    c_bar = np.zeros(K, dtype=float)

    D_aug = np.zeros((K, 2, 2), dtype=float)
    D_update = np.zeros(K, dtype=float)
    lambda_bar = np.zeros(K, dtype=float)
    linear_bar = np.zeros(K, dtype=float)

    for k in range(K):
        # Num_k = a_bar[k] * P_bar[k]
        a_bar[k] = P_tilda[k] * kappa_S[k]

        # Denominator slope with respect to P_bar[k], excluding numerator.
        b_bar[k] = (
            P_tilda[k] * (kappa_V1[k] + kappa_K1[k])
            + sum(P_tilda[i] * kappa_M1[k, i] for i in range(K) if i != k)
            + kappa_DAC1[k]
            + kappa_Th1[k]
            + kappa_ADC1[k]
        )

        # Denominator part independent of P_bar[k].
        c_bar[k] = (
            P_tilda[k] * (kappa_V0[k] + kappa_K0[k])
            + sum(P_tilda[i] * kappa_M0[k, i] for i in range(K) if i != k)
            + kappa_DAC0[k]
            + kappa_Th0[k]
            + kappa_ADC0[k]
        )

        d11 = (mu[k] ** 2) * (a_bar[k] + b_bar[k])
        d22 = (mu[k] ** 2) * c_bar[k]

        D_aug[k, 0, 0] = d11
        D_aug[k, 1, 1] = d22
        D_update[k] = d11

        if lambda_mode == "actual":
            lambda_bar[k] = max(d11, eps_lambda)
        elif lambda_mode == "aug_max":
            lambda_bar[k] = max(d11, d22, eps_lambda)
        elif lambda_mode == "aug_fro":
            lambda_bar[k] = max(np.sqrt(d11**2 + d22**2), eps_lambda)
        else:
            raise ValueError("lambda_mode must be 'actual', 'aug_max', or 'aug_fro'.")

        # Linear QT term in x_bar: mu_k * sqrt(rho_k * a_bar[k]).
        linear_bar[k] = mu[k] * np.sqrt(max(rho[k] * a_bar[k], 0.0))

    return {
        "rho": rho,
        "a_bar": a_bar,
        "b_bar": b_bar,
        "c_bar": c_bar,
        "D_aug": D_aug,
        "D_update": D_update,
        "lambda": lambda_bar,
        "linear": linear_bar,
    }


# -----------------------------------------------------------------------------
# D/lambda construction
# INTERNAL HELPER: builds the D matrix and lambda values for the P_tilda update.
# -----------------------------------------------------------------------------
def build_D_lambda_P_tilda(
    *,
    K: int,
    w: Any,
    gamma: Any,
    mu: Any,
    P_bar: Any,
    P_tilda: Any,
    kappa_S: Any,
    kappa_V1: Any,
    kappa_K1: Any,
    kappa_V0: Any,
    kappa_K0: Any,
    kappa_M1: Any,
    kappa_M0: Any,
    eps_lambda: float = 1e-30,
) -> Dict[str, Array]:
    """
    Build D and lambda values for the P_tilda update.

    Variable:
        x_tilda[k] = sqrt(P_tilda[k]).

    D_tilda[k] receives contributions from all users' ratios:

        D_tilda[k] = mu[k]^2 * self_coeff(k,k)
                     + sum_{j != k} mu[j]^2 * cross_coeff(j,k).
    """
    P_bar = np.asarray(P_bar, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda, dtype=float).reshape(K)
    w = np.asarray(w, dtype=float).reshape(K)
    gamma = np.asarray(gamma, dtype=float).reshape(K)
    mu = np.asarray(mu, dtype=float).reshape(K)

    kappa_S = np.asarray(kappa_S, dtype=float).reshape(K)
    kappa_V1 = np.asarray(kappa_V1, dtype=float).reshape(K)
    kappa_K1 = np.asarray(kappa_K1, dtype=float).reshape(K)
    kappa_V0 = np.asarray(kappa_V0, dtype=float).reshape(K)
    kappa_K0 = np.asarray(kappa_K0, dtype=float).reshape(K)
    kappa_M1 = np.asarray(kappa_M1, dtype=float)
    kappa_M0 = np.asarray(kappa_M0, dtype=float)

    rho = w * (1.0 + gamma)
    a_tilda = P_bar * kappa_S

    # D_contrib[j, k] = contribution of user-j ratio to variable P_tilda[k].
    D_contrib = np.zeros((K, K), dtype=float)

    for j in range(K):
        for k in range(K):
            if j == k:
                coeff_jk = (
                    P_bar[j] * (kappa_S[j] + kappa_V1[j] + kappa_K1[j])
                    + kappa_V0[j]
                    + kappa_K0[j]
                )
            else:
                coeff_jk = P_bar[j] * kappa_M1[j, k] + kappa_M0[j, k]

            D_contrib[j, k] = (mu[j] ** 2) * coeff_jk

    D_update = np.sum(D_contrib, axis=0)
    D_matrix = np.diag(D_update)
    lambda_tilda = np.maximum(D_update, eps_lambda)

    linear_tilda = np.zeros(K, dtype=float)
    for k in range(K):
        # Linear QT term in x_tilda: mu_k * sqrt(rho_k * a_tilda[k]).
        linear_tilda[k] = mu[k] * np.sqrt(max(rho[k] * a_tilda[k], 0.0))

    return {
        "rho": rho,
        "a_tilda": a_tilda,
        "D_contrib": D_contrib,
        "D_matrix": D_matrix,
        "D_update": D_update,
        "lambda": lambda_tilda,
        "linear": linear_tilda,
    }


# -----------------------------------------------------------------------------
# Block update
# INTERNAL HELPER: performs one normalized D/lambda update for P_bar.
# -----------------------------------------------------------------------------
def update_P_bar_using_D_lambda(
    *,
    K: int,
    w: Any,
    gamma: Any,
    mu: Any,
    P_bar: Any,
    P_tilda: Any,
    P_prime_fixed: Any,
    P_total_max: Optional[Any],
    kappa_S: Any,
    kappa_V1: Any,
    kappa_K1: Any,
    kappa_V0: Any,
    kappa_K0: Any,
    kappa_DAC1: Any,
    kappa_DAC0: Any,
    kappa_Th1: Any,
    kappa_ADC1: Any,
    kappa_M1: Any,
    kappa_M0: Any,
    kappa_Th0: Any,
    kappa_ADC0: Any,
    eps_power: float = 1e-12,
    eps_lambda: float = 1e-30,
    lambda_mode: str = "actual",
    normalize_gradient: bool = True,
) -> tuple[Array, Dict[str, Array]]:
    """
    Unfolding-style direct-P gradient update for P_bar with alpha = 1/lambda.

    This intentionally does NOT update x_bar = sqrt(P_bar).  It keeps the same
    gradient structure used in the unfolding code and only replaces the learned
    step size by alpha_bar[k] = 1 / lambda_bar[k].
    """
    P_bar = np.asarray(P_bar, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda, dtype=float).reshape(K)
    w = np.asarray(w, dtype=float).reshape(K)
    gamma = np.asarray(gamma, dtype=float).reshape(K)
    mu = np.asarray(mu, dtype=float).reshape(K)

    kappa_S = np.asarray(kappa_S, dtype=float).reshape(K)
    kappa_V1 = np.asarray(kappa_V1, dtype=float).reshape(K)
    kappa_K1 = np.asarray(kappa_K1, dtype=float).reshape(K)
    kappa_DAC1 = np.asarray(kappa_DAC1, dtype=float).reshape(K)
    kappa_Th1 = np.asarray(kappa_Th1, dtype=float).reshape(K)
    kappa_ADC1 = np.asarray(kappa_ADC1, dtype=float).reshape(K)
    kappa_M1 = np.asarray(kappa_M1, dtype=float)

    D_info = build_D_lambda_P_bar(
        K=K,
        w=w,
        gamma=gamma,
        mu=mu,
        P_bar=P_bar,
        P_tilda=P_tilda,
        kappa_S=kappa_S,
        kappa_V1=kappa_V1,
        kappa_K1=kappa_K1,
        kappa_V0=kappa_V0,
        kappa_K0=kappa_K0,
        kappa_DAC1=kappa_DAC1,
        kappa_DAC0=kappa_DAC0,
        kappa_Th1=kappa_Th1,
        kappa_ADC1=kappa_ADC1,
        kappa_M1=kappa_M1,
        kappa_M0=kappa_M0,
        kappa_Th0=kappa_Th0,
        kappa_ADC0=kappa_ADC0,
        eps_lambda=eps_lambda,
        lambda_mode=lambda_mode,
    )

    # Same direct-P gradient form as the unfolding code:
    #   a = w * (1 + gamma) * kappa_S
    #   H_bar = P_tilda*(kappa_S+kappa_V1+kappa_K1)
    #           + sum_{i!=k} P_tilda[i]*kappa_M1[k,i]
    #           + kappa_DAC1 + kappa_Th1 + kappa_ADC1
    #   grad_P_bar = mu*sqrt(a*P_tilda)/sqrt(P_bar) - mu^2*H_bar
    a = w * (1.0 + gamma) * kappa_S

    H_bar = np.zeros(K, dtype=float)
    for k in range(K):
        H_bar[k] = (
            P_tilda[k] * (kappa_S[k] + kappa_V1[k] + kappa_K1[k])
            + sum(P_tilda[i] * kappa_M1[k, i] for i in range(K) if i != k)
            + kappa_DAC1[k]
            + kappa_Th1[k]
            + kappa_ADC1[k]
        )

    grad_P_bar_raw = (
        mu * np.sqrt(np.maximum(a * P_tilda, eps_power))
        / np.sqrt(np.maximum(P_bar, eps_power))
        - (mu ** 2) * H_bar
    )

    if normalize_gradient:
        grad_P_bar, grad_norm_scale_bar, grad_norm_rho_bar = normalize_single_power_grad_np(
            grad_P_bar_raw,
            P_total_max=P_total_max,
            K=K,
            eps=eps_power,
        )
    else:
        grad_P_bar = grad_P_bar_raw.copy()
        grad_norm_scale_bar = np.ones(K, dtype=float)
        grad_norm_rho_bar = float(max(np.max(np.abs(grad_P_bar_raw)), eps_power))

    alpha_bar = 1.0 / np.maximum(D_info["lambda"], eps_lambda)
    effective_alpha_bar = alpha_bar * grad_norm_scale_bar
    P_bar_candidate = P_bar + alpha_bar * grad_P_bar

    P_bar_new = clip_P_bar_joint(
        P_bar_new=P_bar_candidate,
        P_tilda=P_tilda,
        P_prime=P_prime_fixed,
        P_total_max=P_total_max,
        eps=eps_power,
    )

    D_info["H_bar"] = H_bar
    D_info["gradient_raw"] = grad_P_bar_raw
    D_info["gradient"] = grad_P_bar
    D_info["gradient_normalized"] = grad_P_bar
    D_info["gradient_normalization_scale"] = grad_norm_scale_bar
    D_info["gradient_normalization_rho"] = grad_norm_rho_bar
    D_info["normalize_gradient"] = bool(normalize_gradient)
    D_info["alpha"] = alpha_bar
    D_info["effective_alpha"] = effective_alpha_bar
    D_info["P_bar_candidate"] = P_bar_candidate

    return P_bar_new, D_info

# -----------------------------------------------------------------------------
# Block update
# INTERNAL HELPER: performs one normalized D/lambda update for P_tilda.
# -----------------------------------------------------------------------------
def update_P_tilda_using_D_lambda(
    *,
    K: int,
    w: Any,
    gamma: Any,
    mu: Any,
    P_bar: Any,
    P_tilda: Any,
    P_prime_fixed: Any,
    P_total_max: Optional[Any],
    kappa_S: Any,
    kappa_V1: Any,
    kappa_K1: Any,
    kappa_V0: Any,
    kappa_K0: Any,
    kappa_M1: Any,
    kappa_M0: Any,
    eps_power: float = 1e-12,
    eps_lambda: float = 1e-30,
    normalize_gradient: bool = True,
) -> tuple[Array, Dict[str, Array]]:
    """
    Unfolding-style direct-P gradient update for P_tilda with alpha = 1/lambda.

    This keeps the same P_tilda gradient structure used in the unfolding code.
    P_bar can be the newly updated P_bar from the previous block; gamma and mu
    remain fixed from the current outer iteration, as in one unfolding layer.
    """
    P_bar = np.asarray(P_bar, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda, dtype=float).reshape(K)
    w = np.asarray(w, dtype=float).reshape(K)
    gamma = np.asarray(gamma, dtype=float).reshape(K)
    mu = np.asarray(mu, dtype=float).reshape(K)

    kappa_S = np.asarray(kappa_S, dtype=float).reshape(K)
    kappa_V1 = np.asarray(kappa_V1, dtype=float).reshape(K)
    kappa_K1 = np.asarray(kappa_K1, dtype=float).reshape(K)
    kappa_V0 = np.asarray(kappa_V0, dtype=float).reshape(K)
    kappa_K0 = np.asarray(kappa_K0, dtype=float).reshape(K)
    kappa_M1 = np.asarray(kappa_M1, dtype=float)
    kappa_M0 = np.asarray(kappa_M0, dtype=float)

    D_info = build_D_lambda_P_tilda(
        K=K,
        w=w,
        gamma=gamma,
        mu=mu,
        P_bar=P_bar,
        P_tilda=P_tilda,
        kappa_S=kappa_S,
        kappa_V1=kappa_V1,
        kappa_K1=kappa_K1,
        kappa_V0=kappa_V0,
        kappa_K0=kappa_K0,
        kappa_M1=kappa_M1,
        kappa_M0=kappa_M0,
        eps_lambda=eps_lambda,
    )

    # Same direct-P gradient form as the unfolding code:
    #   a = w * (1 + gamma) * kappa_S
    #   self_tilda = P_bar*(kappa_S+kappa_V1+kappa_K1) + kappa_V0 + kappa_K0
    #   interference_to_others[k] = sum_{j!=k} mu[j]^2 *
    #       (P_bar[j]*kappa_M1[j,k] + kappa_M0[j,k])
    #   H_tilda = mu^2*self_tilda + interference_to_others
    #   grad_P_tilda = mu*sqrt(a*P_bar)/sqrt(P_tilda) - H_tilda
    a = w * (1.0 + gamma) * kappa_S

    self_tilda = P_bar * (kappa_S + kappa_V1 + kappa_K1) + kappa_V0 + kappa_K0

    interference_to_others = np.zeros(K, dtype=float)
    for k in range(K):
        interference_to_others[k] = sum(
            (mu[j] ** 2) * (P_bar[j] * kappa_M1[j, k] + kappa_M0[j, k])
            for j in range(K)
            if j != k
        )

    H_tilda = (mu ** 2) * self_tilda + interference_to_others

    grad_P_tilda_raw = (
        mu * np.sqrt(np.maximum(a * P_bar, eps_power))
        / np.sqrt(np.maximum(P_tilda, eps_power))
        - H_tilda
    )

    if normalize_gradient:
        grad_P_tilda, grad_norm_scale_tilda, grad_norm_rho_tilda = normalize_single_power_grad_np(
            grad_P_tilda_raw,
            P_total_max=P_total_max,
            K=K,
            eps=eps_power,
        )
    else:
        grad_P_tilda = grad_P_tilda_raw.copy()
        grad_norm_scale_tilda = np.ones(K, dtype=float)
        grad_norm_rho_tilda = float(max(np.max(np.abs(grad_P_tilda_raw)), eps_power))

    alpha_tilda = 1.0 / np.maximum(D_info["lambda"], eps_lambda)
    effective_alpha_tilda = alpha_tilda * grad_norm_scale_tilda
    P_tilda_candidate = P_tilda + alpha_tilda * grad_P_tilda

    P_tilda_new = clip_P_tilda_joint(
        P_tilda_new=P_tilda_candidate,
        P_bar=P_bar,
        P_prime=P_prime_fixed,
        P_total_max=P_total_max,
        eps=eps_power,
    )

    D_info["self_tilda"] = self_tilda
    D_info["interference_to_others"] = interference_to_others
    D_info["H_tilda"] = H_tilda
    D_info["gradient_raw"] = grad_P_tilda_raw
    D_info["gradient"] = grad_P_tilda
    D_info["gradient_normalized"] = grad_P_tilda
    D_info["gradient_normalization_scale"] = grad_norm_scale_tilda
    D_info["gradient_normalization_rho"] = grad_norm_rho_tilda
    D_info["normalize_gradient"] = bool(normalize_gradient)
    D_info["alpha"] = alpha_tilda
    D_info["effective_alpha"] = effective_alpha_tilda
    D_info["P_tilda_candidate"] = P_tilda_candidate

    return P_tilda_new, D_info

# -----------------------------------------------------------------------------
# Main optimizer
# MAIN-FILE API: runs the communication-only D/lambda-QT optimizer.
# -----------------------------------------------------------------------------
def run_communication_only_lambda_qt(
    *,
    K: int,
    w: Any,
    d: float,
    tau: float,
    kappa_S: Any,
    kappa_V1: Any,
    kappa_K1: Any,
    kappa_V0: Any,
    kappa_K0: Any,
    kappa_DAC1: Any,
    kappa_DAC0: Any,
    kappa_Th1: Any,
    kappa_ADC1: Any,
    kappa_M1: Any,
    kappa_M0: Any,
    kappa_Th0: Any,
    kappa_ADC0: Any,
    P_bar_init: Any,
    P_tilda_init: Any,
    P_prime_fixed: Any,
    P_total_max: Optional[Any] = None,
    max_iters: int = 200,
    epsilon: float = 1e-6,
    eps_power: float = 1e-12,
    eps_lambda: float = 1e-30,
    lambda_mode_bar: str = "actual",
    normalize_lambda_gradients: bool = True,
    use_backtracking: bool = True,
    verbose: bool = False,
    print_step_sizes: bool = False,
    step_print_every: int = 50,
    step_print_last_n: int = 10,
    compute_sinr_exact_fn: Callable[..., float] | None = None,
    compute_wsr_exact_fn: Callable[..., float] | None = None,
    update_auxiliary_fn: Callable[..., Array] | None = None,
) -> Dict[str, Any]:
    """
    Communication-only D/lambda unfolding-style gradient optimizer.

    Required callbacks from your existing code:
        compute_sinr_exact_fn = compute_SINR_exact
        compute_wsr_exact_fn  = compute_WSR_exact
        update_auxiliary_fn   = update_auxiliary

    P_prime_fixed is kept fixed. For communication-only runs, use zeros(K).

    If normalize_lambda_gradients=True, each raw direct-P gradient is first
    normalized as Pmax/(K*max(abs(grad))) * grad, matching the communication
    unfolding code. The lambda step alpha=1/lambda is then applied to this
    normalized gradient.

    Stores at every iteration:
        D_bar_aug_history       : shape (iters, K, 2, 2)
        D_bar_update_history    : shape (iters, K)
        lambda_bar_history      : shape (iters, K)
        D_tilda_matrix_history  : shape (iters, K, K), diagonal matrix
        D_tilda_update_history  : shape (iters, K)
        lambda_tilda_history    : shape (iters, K)
    """
    if compute_sinr_exact_fn is None:
        raise ValueError("Pass compute_sinr_exact_fn=compute_SINR_exact from your main code.")
    if compute_wsr_exact_fn is None:
        raise ValueError("Pass compute_wsr_exact_fn=compute_WSR_exact from your main code.")
    if update_auxiliary_fn is None:
        raise ValueError("Pass update_auxiliary_fn=update_auxiliary from your main code.")

    P_bar = np.asarray(P_bar_init, dtype=float).reshape(K).copy()
    P_tilda = np.asarray(P_tilda_init, dtype=float).reshape(K).copy()
    if P_prime_fixed is None:
        # Communication-only case: no sensing power is optimized.
        # Keep P_prime as a zero placeholder only for the power constraint.
        P_prime_fixed = np.zeros(K, dtype=float)
    else:
        P_prime_fixed = np.asarray(P_prime_fixed, dtype=float).reshape(K).copy()
    w = np.asarray(w, dtype=float).reshape(K)

    P_bar = clip_P_bar_joint(P_bar, P_tilda, P_prime_fixed, P_total_max, eps_power)
    P_tilda = clip_P_tilda_joint(P_tilda, P_bar, P_prime_fixed, P_total_max, eps_power)

    initial_state = evaluate_comm_only_state(
        K=K,
        w=w,
        d=d,
        tau=tau,
        kappa_S=kappa_S,
        kappa_V1=kappa_V1,
        kappa_K1=kappa_K1,
        kappa_V0=kappa_V0,
        kappa_K0=kappa_K0,
        kappa_DAC1=kappa_DAC1,
        kappa_DAC0=kappa_DAC0,
        kappa_Th1=kappa_Th1,
        kappa_ADC1=kappa_ADC1,
        kappa_M1=kappa_M1,
        kappa_M0=kappa_M0,
        kappa_Th0=kappa_Th0,
        kappa_ADC0=kappa_ADC0,
        P_bar=P_bar,
        P_tilda=P_tilda,
        compute_sinr_exact_fn=compute_sinr_exact_fn,
        compute_wsr_exact_fn=compute_wsr_exact_fn,
    )

    full_comm_history = [initial_state["comm_wsr"]]
    comm_history = []

    D_bar_aug_history = []
    D_bar_update_history = []
    lambda_bar_history = []
    grad_bar_history = []
    grad_bar_raw_history = []
    alpha_bar_history = []
    effective_alpha_bar_history = []
    grad_bar_norm_scale_history = []

    D_tilda_matrix_history = []
    D_tilda_update_history = []
    lambda_tilda_history = []
    grad_tilda_history = []
    grad_tilda_raw_history = []
    alpha_tilda_history = []
    effective_alpha_tilda_history = []
    grad_tilda_norm_scale_history = []

    P_bar_history = [P_bar.copy()]
    P_tilda_history = [P_tilda.copy()]

    converged = False

    for t in range(max_iters):
        old_P_bar = P_bar.copy()
        old_P_tilda = P_tilda.copy()
        old_wsr = float(full_comm_history[-1])

        old_state = evaluate_comm_only_state(
            K=K,
            w=w,
            d=d,
            tau=tau,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1,
            kappa_DAC0=kappa_DAC0,
            kappa_Th1=kappa_Th1,
            kappa_ADC1=kappa_ADC1,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            kappa_Th0=kappa_Th0,
            kappa_ADC0=kappa_ADC0,
            P_bar=old_P_bar,
            P_tilda=old_P_tilda,
            compute_sinr_exact_fn=compute_sinr_exact_fn,
            compute_wsr_exact_fn=compute_wsr_exact_fn,
        )

        gamma = old_state["gamma"]
        P_th = old_state["P_th"]
        P_adc = old_state["P_adc"]

        mu = update_auxiliary_fn(
            K,
            w,
            gamma,
            kappa_S,
            kappa_V1,
            kappa_K1,
            kappa_V0,
            kappa_K0,
            kappa_DAC1,
            kappa_DAC0,
            kappa_M1,
            kappa_M0,
            kappa_Th1,
            kappa_ADC1,
            old_P_bar,
            old_P_tilda,
            P_th,
            P_adc,
        )

        # Update P_bar using old P_tilda.
        P_bar_target, D_bar_info = update_P_bar_using_D_lambda(
            K=K,
            w=w,
            gamma=gamma,
            mu=mu,
            P_bar=old_P_bar,
            P_tilda=old_P_tilda,
            P_prime_fixed=P_prime_fixed,
            P_total_max=P_total_max,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1,
            kappa_DAC0=kappa_DAC0,
            kappa_Th1=kappa_Th1,
            kappa_ADC1=kappa_ADC1,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            kappa_Th0=kappa_Th0,
            kappa_ADC0=kappa_ADC0,
            eps_power=eps_power,
            eps_lambda=eps_lambda,
            lambda_mode=lambda_mode_bar,
            normalize_gradient=normalize_lambda_gradients,
        )

        # Update P_tilda using updated P_bar but old auxiliary variables.
        P_tilda_target, D_tilda_info = update_P_tilda_using_D_lambda(
            K=K,
            w=w,
            gamma=gamma,
            mu=mu,
            P_bar=P_bar_target,
            P_tilda=old_P_tilda,
            P_prime_fixed=P_prime_fixed,
            P_total_max=P_total_max,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            eps_power=eps_power,
            eps_lambda=eps_lambda,
            normalize_gradient=normalize_lambda_gradients,
        )

        eta = 1.0
        accepted = False

        if use_backtracking:
            while eta >= 1e-6:
                P_bar_candidate = old_P_bar + eta * (P_bar_target - old_P_bar)
                P_tilda_candidate = old_P_tilda + eta * (P_tilda_target - old_P_tilda)

                candidate_state = evaluate_comm_only_state(
                    K=K,
                    w=w,
                    d=d,
                    tau=tau,
                    kappa_S=kappa_S,
                    kappa_V1=kappa_V1,
                    kappa_K1=kappa_K1,
                    kappa_V0=kappa_V0,
                    kappa_K0=kappa_K0,
                    kappa_DAC1=kappa_DAC1,
                    kappa_DAC0=kappa_DAC0,
                    kappa_Th1=kappa_Th1,
                    kappa_ADC1=kappa_ADC1,
                    kappa_M1=kappa_M1,
                    kappa_M0=kappa_M0,
                    kappa_Th0=kappa_Th0,
                    kappa_ADC0=kappa_ADC0,
                    P_bar=P_bar_candidate,
                    P_tilda=P_tilda_candidate,
                    compute_sinr_exact_fn=compute_sinr_exact_fn,
                    compute_wsr_exact_fn=compute_wsr_exact_fn,
                )

                if candidate_state["comm_wsr"] >= old_wsr - 1e-12:
                    P_bar = P_bar_candidate
                    P_tilda = P_tilda_candidate
                    new_wsr = float(candidate_state["comm_wsr"])
                    accepted = True
                    break

                eta *= 0.5

            if not accepted:
                P_bar = old_P_bar
                P_tilda = old_P_tilda
                new_wsr = old_wsr
                eta = 0.0
        else:
            P_bar = P_bar_target
            P_tilda = P_tilda_target
            new_state = evaluate_comm_only_state(
                K=K,
                w=w,
                d=d,
                tau=tau,
                kappa_S=kappa_S,
                kappa_V1=kappa_V1,
                kappa_K1=kappa_K1,
                kappa_V0=kappa_V0,
                kappa_K0=kappa_K0,
                kappa_DAC1=kappa_DAC1,
                kappa_DAC0=kappa_DAC0,
                kappa_Th1=kappa_Th1,
                kappa_ADC1=kappa_ADC1,
                kappa_M1=kappa_M1,
                kappa_M0=kappa_M0,
                kappa_Th0=kappa_Th0,
                kappa_ADC0=kappa_ADC0,
                P_bar=P_bar,
                P_tilda=P_tilda,
                compute_sinr_exact_fn=compute_sinr_exact_fn,
                compute_wsr_exact_fn=compute_wsr_exact_fn,
            )
            new_wsr = float(new_state["comm_wsr"])
            accepted = True

        full_comm_history.append(float(new_wsr))
        comm_history.append(float(new_wsr))

        D_bar_aug_history.append(D_bar_info["D_aug"].copy())
        D_bar_update_history.append(D_bar_info["D_update"].copy())
        lambda_bar_history.append(D_bar_info["lambda"].copy())
        grad_bar_history.append(D_bar_info["gradient"].copy())
        grad_bar_raw_history.append(D_bar_info["gradient_raw"].copy())
        alpha_bar_history.append(D_bar_info["alpha"].copy())
        effective_alpha_bar_history.append(D_bar_info["effective_alpha"].copy())
        grad_bar_norm_scale_history.append(D_bar_info["gradient_normalization_scale"].copy())

        D_tilda_matrix_history.append(D_tilda_info["D_matrix"].copy())
        D_tilda_update_history.append(D_tilda_info["D_update"].copy())
        lambda_tilda_history.append(D_tilda_info["lambda"].copy())
        grad_tilda_history.append(D_tilda_info["gradient"].copy())
        grad_tilda_raw_history.append(D_tilda_info["gradient_raw"].copy())
        alpha_tilda_history.append(D_tilda_info["alpha"].copy())
        effective_alpha_tilda_history.append(D_tilda_info["effective_alpha"].copy())
        grad_tilda_norm_scale_history.append(D_tilda_info["gradient_normalization_scale"].copy())

        P_bar_history.append(P_bar.copy())
        P_tilda_history.append(P_tilda.copy())

        delta = float(new_wsr - old_wsr)

        if verbose:
            print(
                f"lambda-QT iter={t + 1:03d} | "
                f"comm={new_wsr:.8f} | "
                f"delta={delta:.3e} | "
                f"eta={eta:.3e} | "
                f"accepted={accepted}"
            )

        if print_step_sizes:
            every = max(1, int(step_print_every))
            should_print = (t == 0) or ((t + 1) % every == 0)
            if should_print:
                alpha_bar_now = np.asarray(D_bar_info["alpha"], dtype=float)
                alpha_tilda_now = np.asarray(D_tilda_info["alpha"], dtype=float)
                lambda_bar_now = np.asarray(D_bar_info["lambda"], dtype=float)
                lambda_tilda_now = np.asarray(D_tilda_info["lambda"], dtype=float)
                print(
                    f"[STEP TRACE] iter={t + 1:04d} | "
                    f"alpha_bar mean={alpha_bar_now.mean():.6e}, min={alpha_bar_now.min():.6e}, max={alpha_bar_now.max():.6e} | "
                    f"alpha_tilda mean={alpha_tilda_now.mean():.6e}, min={alpha_tilda_now.min():.6e}, max={alpha_tilda_now.max():.6e} | "
                    f"lambda_bar mean={lambda_bar_now.mean():.6e} | "
                    f"lambda_tilda mean={lambda_tilda_now.mean():.6e} | "
                    f"normalized={normalize_lambda_gradients} | "
                    f"eta={eta:.3e}"
                )

        if t > 0 and abs(delta) < epsilon:
            converged = True
            break

    final_state = evaluate_comm_only_state(
        K=K,
        w=w,
        d=d,
        tau=tau,
        kappa_S=kappa_S,
        kappa_V1=kappa_V1,
        kappa_K1=kappa_K1,
        kappa_V0=kappa_V0,
        kappa_K0=kappa_K0,
        kappa_DAC1=kappa_DAC1,
        kappa_DAC0=kappa_DAC0,
        kappa_Th1=kappa_Th1,
        kappa_ADC1=kappa_ADC1,
        kappa_M1=kappa_M1,
        kappa_M0=kappa_M0,
        kappa_Th0=kappa_Th0,
        kappa_ADC0=kappa_ADC0,
        P_bar=P_bar,
        P_tilda=P_tilda,
        compute_sinr_exact_fn=compute_sinr_exact_fn,
        compute_wsr_exact_fn=compute_wsr_exact_fn,
    )

    full_comm_history_arr = np.asarray(full_comm_history, dtype=float)

    result = {
        "method": "normalized_lambda_unfolding_gradient_comm_only" if normalize_lambda_gradients else "lambda_unfolding_gradient_comm_only",
        "normalize_lambda_gradients": bool(normalize_lambda_gradients),
        "initial_comm_wsr": float(full_comm_history_arr[0]),
        "final_comm_wsr": float(full_comm_history_arr[-1]),
        "comm_history": np.asarray(comm_history, dtype=float),
        "full_comm_history": full_comm_history_arr,
        "P_bar_initial": np.asarray(P_bar_init, dtype=float).reshape(K),
        "P_tilda_initial": np.asarray(P_tilda_init, dtype=float).reshape(K),
        "P_prime_fixed": P_prime_fixed,
        "P_bar_opt": P_bar,
        "P_tilda_opt": P_tilda,
        "P_prime_opt": P_prime_fixed,
        "gamma_opt": final_state["gamma"],
        "P_th_opt": final_state["P_th"],
        "P_adc_opt": final_state["P_adc"],
        "D_bar_aug_history": np.asarray(D_bar_aug_history, dtype=float),
        "D_bar_update_history": np.asarray(D_bar_update_history, dtype=float),
        "lambda_bar_history": np.asarray(lambda_bar_history, dtype=float),
        "grad_bar_history": np.asarray(grad_bar_history, dtype=float),
        "grad_bar_raw_history": np.asarray(grad_bar_raw_history, dtype=float),
        "alpha_bar_history": np.asarray(alpha_bar_history, dtype=float),
        "effective_alpha_bar_history": np.asarray(effective_alpha_bar_history, dtype=float),
        "grad_bar_norm_scale_history": np.asarray(grad_bar_norm_scale_history, dtype=float),
        "D_tilda_matrix_history": np.asarray(D_tilda_matrix_history, dtype=float),
        "D_tilda_update_history": np.asarray(D_tilda_update_history, dtype=float),
        "lambda_tilda_history": np.asarray(lambda_tilda_history, dtype=float),
        "grad_tilda_history": np.asarray(grad_tilda_history, dtype=float),
        "grad_tilda_raw_history": np.asarray(grad_tilda_raw_history, dtype=float),
        "alpha_tilda_history": np.asarray(alpha_tilda_history, dtype=float),
        "effective_alpha_tilda_history": np.asarray(effective_alpha_tilda_history, dtype=float),
        "grad_tilda_norm_scale_history": np.asarray(grad_tilda_norm_scale_history, dtype=float),
        "P_bar_history": np.asarray(P_bar_history, dtype=float),
        "P_tilda_history": np.asarray(P_tilda_history, dtype=float),
        "monotonic": bool(np.all(np.diff(full_comm_history_arr) >= -1e-10)),
        "converged": converged,
        "iterations": len(comm_history),
    }

    if print_step_sizes:
        print_lambda_step_size_summary(
            result,
            label="D/lambda QT alpha/lambda summary",
            last_n=step_print_last_n,
        )

    return result


# -----------------------------------------------------------------------------
# Public plotting API
# MAIN-FILE API: imported by the main file to plot classical/lambda/DU communication histories.
# -----------------------------------------------------------------------------
def plot_comm_only_method_comparison(
    *,
    classical_result: Optional[Dict[str, Any]] = None,
    lambda_qt_result: Optional[Dict[str, Any]] = None,
    du_history: Optional[Any] = None,
    title: str = "Communication-only: classical vs D/lambda QT vs unfolding",
    save_path: Optional[str] = None,
) -> None:
    """Plot classical, D/lambda QT, and optional DU convergence histories."""
    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")

    plt.figure(figsize=(8, 5))

    if classical_result is not None:
        plt.plot(
            np.asarray(classical_result["full_comm_history"], dtype=float),
            marker="o",
            linewidth=2,
            label="Classical closed-form QT",
        )

    if lambda_qt_result is not None:
        plt.plot(
            np.asarray(lambda_qt_result["full_comm_history"], dtype=float),
            marker="s",
            linewidth=2,
            label="D/lambda unfolding-gradient QT",
        )

    if du_history is not None:
        plt.plot(
            np.asarray(du_history, dtype=float).reshape(-1),
            marker="^",
            linewidth=2,
            label="Deep unfolding",
        )

    plt.xlabel("Iteration / layer")
    plt.ylabel("Communication WSR")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


# -----------------------------------------------------------------------------
# Reporting helper
# INTERNAL HELPER: used by run_classical_lambda_qt_comparison to print final comparison values.
# -----------------------------------------------------------------------------
def print_comm_only_comparison_summary(
    *,
    classical_result: Optional[Dict[str, Any]],
    lambda_qt_result: Dict[str, Any],
    du_history: Optional[Any] = None,
) -> None:
    """Print a compact numerical comparison."""
    print("\nCOMMUNICATION-ONLY COMPARISON")
    print("-" * 90)

    if classical_result is not None:
        print(f"Initial WSR                 : {classical_result['initial_comm_wsr']:.8f}")
        print(f"Classical closed-form final : {classical_result['final_comm_wsr']:.8f}")
        print(f"Classical iterations        : {classical_result['iterations']}")
        print(f"Classical monotonic         : {classical_result['monotonic']}")
    else:
        print(f"Initial WSR                 : {lambda_qt_result['initial_comm_wsr']:.8f}")

    print(f"D/lambda unfolding-grad final: {lambda_qt_result['final_comm_wsr']:.8f}")
    print(f"D/lambda QT iterations      : {lambda_qt_result['iterations']}")
    print(f"D/lambda QT monotonic       : {lambda_qt_result['monotonic']}")

    if du_history is not None:
        du_hist_np = np.asarray(du_history, dtype=float).reshape(-1)
        print(f"DU final                    : {du_hist_np[-1]:.8f}")


# -----------------------------------------------------------------------------
# Main comparison wrapper
# MAIN-FILE API: imported by the main file to compare classical FP/QT, lambda-QT and DU for communication mode.
# -----------------------------------------------------------------------------
def run_classical_lambda_qt_comparison(
    *,
    K: int,
    w: Any,
    d: float,
    tau: float,
    kappa_S: Any,
    kappa_V1: Any,
    kappa_K1: Any,
    kappa_V0: Any,
    kappa_K0: Any,
    kappa_DAC1: Any,
    kappa_DAC0: Any,
    kappa_Th1: Any,
    kappa_ADC1: Any,
    kappa_M1: Any,
    kappa_M0: Any,
    kappa_Th0: Any,
    kappa_ADC0: Any,
    Z_list: Any,
    P_bar_init: Any,
    P_tilda_init: Any,
    P_prime_fixed: Any,
    P_total_max: Optional[Any],
    max_iters: int = 200,
    epsilon: float = 1e-6,
    eps_power: float = 1e-12,
    eps_lambda: float = 1e-30,
    lambda_mode_bar: str = "actual",
    normalize_lambda_gradients: bool = True,
    use_backtracking: bool = True,
    du_history: Optional[Any] = None,
    save_plot_path: Optional[str] = "comm_only_classical_vs_lambda_qt_vs_du.png",
    verbose: bool = False,
    print_step_sizes: bool = False,
    step_print_every: int = 50,
    step_print_last_n: int = 10,
    step_size_save_path: Optional[str] = None,
    compute_sinr_exact_fn: Callable[..., float] | None = None,
    compute_wsr_exact_fn: Callable[..., float] | None = None,
    update_auxiliary_fn: Callable[..., Array] | None = None,
    classical_runner_fn: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Run existing classical communication-only optimizer and new D/lambda QT optimizer.

    Required callbacks:
        compute_sinr_exact_fn = compute_SINR_exact
        compute_wsr_exact_fn  = compute_WSR_exact
        update_auxiliary_fn   = update_auxiliary

    Optional callback:
        classical_runner_fn = run_communication_only_optimisation
    """
    classical_result = None

    if classical_runner_fn is not None:
        classical_result = classical_runner_fn(
            K=K,
            w=w,
            d=d,
            tau=tau,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1,
            kappa_DAC0=kappa_DAC0,
            kappa_Th1=kappa_Th1,
            kappa_ADC1=kappa_ADC1,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            kappa_Th0=kappa_Th0,
            kappa_ADC0=kappa_ADC0,
            Z_list=Z_list,
            P_bar_init=P_bar_init,
            P_tilda_init=P_tilda_init,
            P_prime_fixed=P_prime_fixed,
            P_total_max=P_total_max,
            max_iters=max_iters,
            epsilon=epsilon,
            eps_power=eps_power,
            verbose=verbose,
        )

    lambda_qt_result = run_communication_only_lambda_qt(
        K=K,
        w=w,
        d=d,
        tau=tau,
        kappa_S=kappa_S,
        kappa_V1=kappa_V1,
        kappa_K1=kappa_K1,
        kappa_V0=kappa_V0,
        kappa_K0=kappa_K0,
        kappa_DAC1=kappa_DAC1,
        kappa_DAC0=kappa_DAC0,
        kappa_Th1=kappa_Th1,
        kappa_ADC1=kappa_ADC1,
        kappa_M1=kappa_M1,
        kappa_M0=kappa_M0,
        kappa_Th0=kappa_Th0,
        kappa_ADC0=kappa_ADC0,
        P_bar_init=P_bar_init,
        P_tilda_init=P_tilda_init,
        P_prime_fixed=P_prime_fixed,
        P_total_max=P_total_max,
        max_iters=max_iters,
        epsilon=epsilon,
        eps_power=eps_power,
        eps_lambda=eps_lambda,
        lambda_mode_bar=lambda_mode_bar,
        normalize_lambda_gradients=normalize_lambda_gradients,
        use_backtracking=use_backtracking,
        verbose=verbose,
        print_step_sizes=print_step_sizes,
        step_print_every=step_print_every,
        step_print_last_n=step_print_last_n,
        compute_sinr_exact_fn=compute_sinr_exact_fn,
        compute_wsr_exact_fn=compute_wsr_exact_fn,
        update_auxiliary_fn=update_auxiliary_fn,
    )

    print_comm_only_comparison_summary(
        classical_result=classical_result,
        lambda_qt_result=lambda_qt_result,
        du_history=du_history,
    )

    if step_size_save_path is not None:
        save_lambda_step_size_summary(lambda_qt_result, step_size_save_path)

    if save_plot_path is not None:
        plot_comm_only_method_comparison(
            classical_result=classical_result,
            lambda_qt_result=lambda_qt_result,
            du_history=du_history,
            save_path=save_plot_path,
        )

    return {
        "classical": classical_result,
        "lambda_qt": lambda_qt_result,
        "du_history": None if du_history is None else np.asarray(du_history, dtype=float),
    }
