"""
Joint communication-sensing LDT + nonhomogeneous-QT / D-lambda optimizer.

This module extends the communication-only D/lambda-QT code to the joint case:
    P_bar[k]    : communication pilot/channel-estimation power
    P_tilda[k]  : communication data power
    P_prime[k]  : sensing power

Per-user constraint:
    P_bar[k] + P_tilda[k] + P_prime[k] <= P_total_max[k]

The communication P_bar and P_tilda D/lambda updates reuse the same LDT-correct
communication curvature used in Communication_Accelerated_QT.py.  The new part
is the sensing P_prime D/lambda update.

Sensing model:
    S_q = sum_k a_sens[q,k] P_prime[k]
    D_q = sum_k b_sens[q,k] P_prime[k] + n_sens[q]
    f_s = w_s * sum_q omega_q log2(1 + S_q/D_q)

LDT/QT auxiliary variables:
    gamma_s,q = S_q / D_q
    rho_s,q   = w_s * omega_q * (1 + gamma_s,q)
    y_s,q     = sqrt(rho_s,q * S_q) / (S_q + D_q)

For the Section-V-style sensing formulation, use the square-root variable
    x_prime[k] = sqrt(P_prime[k]).

The full target numerator is represented by
    A_q = diag(sqrt(a_sens[q,1]), ..., sqrt(a_sens[q,K])),
so ||A_q x_prime||^2 = S_q and all users' sensing powers are included in
target q's numerator.  The post-LDT denominator is represented by the
augmented matrix
    B_q = diag(sqrt(a_sens[q,1]+b_sens[q,1]), ...,
               sqrt(a_sens[q,K]+b_sens[q,K]), sqrt(n_sens[q])).

For fixed auxiliary variables, the nonhomogeneous-QT update uses
    D_prime[k] = sum_q y_s,q^2 * (a_sens[q,k] + b_sens[q,k]),
    linear_prime[k] = sum_q sqrt(rho_s,q) * sqrt(a_sens[q,k]) * y_vec[q,k],
where y_vec[q,k] = sqrt(rho_s,q) * sqrt(a_sens[q,k]) * x_prime[k] / (S_q+D_q).
The update is performed in x_prime, then squared back to P_prime.

The optimizer uses optional backtracking on the true joint objective for a
monotone accepted trajectory.

Normalized version:
    If normalize_lambda_gradients=True, each lambda-gradient block is scaled
    by the same max-gradient rule used in the communication accelerated QT /
    unfolding code before applying alpha = 1/lambda.

    For P_bar and P_tilda, the imported communication updates normalize the
    direct-power gradients as
        grad_norm = Pmax / (K * max(abs(grad_raw))) * grad_raw.

    For P_prime, the optimized variable is x_prime = sqrt(P_prime), so the
    default sensing normalization uses
        grad_x_norm = sqrt(Pmax) / (K * max(abs(grad_x_raw))) * grad_x_raw.
    Set prime_normalization_mode="power" to use Pmax scaling instead.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    # Preferred when this file is placed inside the Communication/ package.
    from Communication.Communication_Accelerated_QT import (  # type: ignore
        _as_power_vector,
        _flatten_history,
        _history_stats_line,
        clip_P_bar_joint,
        clip_P_tilda_joint,
        update_P_bar_using_D_lambda,
        update_P_tilda_using_D_lambda,
    )
except Exception:  # pragma: no cover
    # Fallback for running this file from the same folder as Communication_Accelerated_QT.py.
    from Communication_Accelerated_QT import (  # type: ignore
        _as_power_vector,
        _flatten_history,
        _history_stats_line,
        clip_P_bar_joint,
        clip_P_tilda_joint,
        update_P_bar_using_D_lambda,
        update_P_tilda_using_D_lambda,
    )

Array = np.ndarray


# -----------------------------------------------------------------------------
# Gradient normalization
# INTERNAL HELPER: normalizes direct-power gradients when power-domain scaling is requested.
# -----------------------------------------------------------------------------
def normalize_single_power_grad_np(
    grad: Any,
    P_total_max: Optional[Any],
    K: int,
    eps: float = 1e-12,
) -> tuple[Array, Array, float]:
    """
    Normalize a direct-power gradient block using the same rule as the
    normalized communication accelerated QT code:

        grad_norm = Pmax / (K * max(abs(grad_raw))) * grad_raw.

    Returns grad_norm, scale, and rho=max(abs(grad_raw)).
    """
    grad_np = np.asarray(grad, dtype=float).reshape(K)
    rho = float(max(np.max(np.abs(grad_np)), eps))

    Pmax = _as_power_vector(P_total_max, K)
    if Pmax is None:
        scale = np.ones(K, dtype=float)
        return grad_np.copy(), scale, rho

    scale = np.asarray(Pmax, dtype=float).reshape(K) / (rho * float(K))
    return scale * grad_np, scale, rho


# -----------------------------------------------------------------------------
# Gradient normalization
# INTERNAL HELPER: normalizes the x_prime = sqrt(P_prime) sensing gradient.
# -----------------------------------------------------------------------------
def normalize_sqrt_power_grad_np(
    grad: Any,
    P_total_max: Optional[Any],
    K: int,
    eps: float = 1e-12,
) -> tuple[Array, Array, float]:
    """
    Normalize a square-root-power gradient block.

    P_prime is updated through x_prime = sqrt(P_prime), so the natural
    unfolding-style scale for the x-gradient is sqrt(Pmax), not Pmax:

        grad_x_norm = sqrt(Pmax) / (K * max(abs(grad_x_raw))) * grad_x_raw.

    Returns grad_x_norm, scale, and rho=max(abs(grad_x_raw)).
    """
    grad_np = np.asarray(grad, dtype=float).reshape(K)
    rho = float(max(np.max(np.abs(grad_np)), eps))

    Pmax = _as_power_vector(P_total_max, K)
    if Pmax is None:
        scale = np.ones(K, dtype=float)
        return grad_np.copy(), scale, rho

    sqrt_pmax = np.sqrt(np.maximum(np.asarray(Pmax, dtype=float).reshape(K), eps))
    scale = sqrt_pmax / (rho * float(K))
    return scale * grad_np, scale, rho


# -----------------------------------------------------------------------------
# History helpers
# INTERNAL HELPER: safely copies arrays from intermediate D/lambda dictionaries for result logging.
# -----------------------------------------------------------------------------
def _copy_info_array(info: Dict[str, Any], key: str, fallback_key: Optional[str] = None) -> Array:
    """Read an array from a D/lambda info dict with a safe fallback."""
    if key in info:
        return np.asarray(info[key], dtype=float).copy()
    if fallback_key is not None and fallback_key in info:
        return np.asarray(info[fallback_key], dtype=float).copy()
    return np.asarray([], dtype=float)


# -----------------------------------------------------------------------------
# Power-constraint clipping
# INTERNAL HELPER: clips sensing power P_prime under the remaining per-user budget.
# -----------------------------------------------------------------------------
def clip_P_prime_joint(
    P_prime_new: Any,
    P_bar: Any,
    P_tilda: Any,
    P_total_max: Optional[Any],
    eps: float = 1e-12,
) -> Array:
    """Clip P_prime for fixed P_bar and P_tilda."""
    P_prime_new = np.asarray(P_prime_new, dtype=float).reshape(-1)
    if P_total_max is None:
        return np.maximum(P_prime_new, eps)

    K = len(P_prime_new)
    Pmax = _as_power_vector(P_total_max, K)
    assert Pmax is not None
    upper = np.maximum(
        Pmax - np.asarray(P_bar, dtype=float).reshape(K) - np.asarray(P_tilda, dtype=float).reshape(K) - eps,
        eps,
    )
    return np.clip(P_prime_new, eps, upper)


# -----------------------------------------------------------------------------
# Projection helper
# INTERNAL HELPER: low-level simplex projection used by the three-block joint projection.
# -----------------------------------------------------------------------------
def _project_to_simplex_nonnegative(v: Array, total: float) -> Array:
    """Project v onto {x >= 0, sum(x) <= total}."""
    v = np.asarray(v, dtype=float).reshape(-1)
    if np.sum(np.maximum(v, 0.0)) <= total:
        return np.maximum(v, 0.0)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - total))[0]
    if len(rho) == 0:
        theta = cssv[-1] / len(u)
    else:
        rho = rho[-1]
        theta = (cssv[rho] - total) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# -----------------------------------------------------------------------------
# Joint projection
# INTERNAL HELPER: projects P_bar, P_tilda and P_prime onto the feasible per-user budget set.
# -----------------------------------------------------------------------------
def project_joint_three_power_blocks(
    P_bar_raw: Any,
    P_tilda_raw: Any,
    P_prime_raw: Any,
    P_total_max: Optional[Any],
    eps: float = 1e-12,
) -> tuple[Array, Array, Array]:
    """Exact per-user projection onto P_bar,P_tilda,P_prime >= eps and sum <= Pmax."""
    P_bar_raw = np.asarray(P_bar_raw, dtype=float).reshape(-1)
    P_tilda_raw = np.asarray(P_tilda_raw, dtype=float).reshape(-1)
    P_prime_raw = np.asarray(P_prime_raw, dtype=float).reshape(-1)
    K = len(P_bar_raw)
    if P_total_max is None:
        return np.maximum(P_bar_raw, eps), np.maximum(P_tilda_raw, eps), np.maximum(P_prime_raw, eps)

    Pmax = _as_power_vector(P_total_max, K)
    assert Pmax is not None
    out_bar = np.zeros(K, dtype=float)
    out_tilda = np.zeros(K, dtype=float)
    out_prime = np.zeros(K, dtype=float)
    for k in range(K):
        v = np.array([P_bar_raw[k], P_tilda_raw[k], P_prime_raw[k]], dtype=float)
        v_eps = np.maximum(v, eps)
        if np.sum(v_eps) <= Pmax[k]:
            p = v_eps
        else:
            budget = max(Pmax[k] - 3.0 * eps, 0.0)
            p = _project_to_simplex_nonnegative(v - eps, budget) + eps
        out_bar[k], out_tilda[k], out_prime[k] = p
    return out_bar, out_tilda, out_prime


# -----------------------------------------------------------------------------
# Sensing evaluation
# INTERNAL HELPER: computes target-wise desired sensing signal S_q and denominator D_q.
# -----------------------------------------------------------------------------
def compute_sensing_S_D(P_prime: Any, a_sens: Any, b_sens: Any, n_sens: Any, eps: float = 1e-30) -> tuple[Array, Array]:
    """Return S_q and D_q for the sensing model."""
    P_prime = np.asarray(P_prime, dtype=float).reshape(-1)
    a_sens = np.asarray(a_sens, dtype=float)
    b_sens = np.asarray(b_sens, dtype=float)
    n_sens = np.asarray(n_sens, dtype=float).reshape(-1)
    S = np.maximum(a_sens @ P_prime, eps)
    D = np.maximum(b_sens @ P_prime + n_sens, eps)
    return S, D


# -----------------------------------------------------------------------------
# Sensing evaluation
# INTERNAL HELPER: evaluates weighted sensing WSR for the current P_prime.
# -----------------------------------------------------------------------------
def compute_sensing_wsr(
    P_prime: Any,
    a_sens: Any,
    b_sens: Any,
    n_sens: Any,
    target_weights: Optional[Any] = None,
    w_s: float = 1.0,
) -> float:
    """Weighted sensing objective contribution: w_s * sum_q omega_q log2(1+SINR_q)."""
    S, D = compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens)
    sinr = S / (D + 1e-30)
    if target_weights is None:
        target_weights = np.ones_like(sinr)
    omega = np.asarray(target_weights, dtype=float).reshape(-1)
    return float(float(w_s) * np.sum(omega * np.log2(1.0 + np.maximum(sinr, 0.0))))


# -----------------------------------------------------------------------------
# Sensing reporting
# INTERNAL HELPER: builds a compact sensing dictionary used in result summaries.
# -----------------------------------------------------------------------------
def compute_sensing_components(
    P_prime: Any,
    a_sens: Any,
    b_sens: Any,
    n_sens: Any,
    target_weights: Optional[Any] = None,
    w_s: float = 1.0,
) -> Dict[str, Any]:
    """Compact sensing components dictionary used by result summaries."""
    S, D = compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens)
    sinr = S / (D + 1e-30)
    if target_weights is None:
        target_weights = np.ones_like(sinr)
    omega = np.asarray(target_weights, dtype=float).reshape(-1)
    raw = omega * np.log2(1.0 + np.maximum(sinr, 0.0))
    return {
        "S": S,
        "D": D,
        "SINR": sinr,
        "SINR_dB": 10.0 * np.log10(np.maximum(sinr, 1e-30)),
        "rate": np.log2(1.0 + np.maximum(sinr, 0.0)),
        "omega": omega,
        "weighted_rate_no_ws": raw,
        "weighted_rate_with_ws": float(w_s) * raw,
        "weighted_wsr_no_ws": float(np.sum(raw)),
        "raw_wsr": float(np.sum(raw)),
        "wsr": float(float(w_s) * np.sum(raw)),
        "weighted_wsr": float(float(w_s) * np.sum(raw)),
    }


# -----------------------------------------------------------------------------
# Sensing D/lambda construction
# INTERNAL HELPER: builds D_prime, lambda_prime and the QT sensing gradient for P_prime.
# -----------------------------------------------------------------------------
def build_D_lambda_P_prime_sensing(
    *,
    P_prime: Any,
    a_sens: Any,
    b_sens: Any,
    n_sens: Any,
    target_weights: Optional[Any],
    w_s: float,
    eps_power: float = 1e-12,
    eps_lambda: float = 1e-30,
) -> Dict[str, Array]:
    """
    Build the Section-V-style sensing QT quantities for P_prime.

    Variable used by the QT update:
        x_prime[k] = sqrt(P_prime[k]).

    Full target-wise matrices:
        A_q = diag(sqrt(a_q1), ..., sqrt(a_qK))
        B_q = diag(sqrt(a_q1+b_q1), ..., sqrt(a_qK+b_qK), sqrt(n_q))

    Thus:
        ||A_q x_prime||^2 = S_q = sum_k a_qk P_prime[k]
        ||B_q [x_prime;1]||^2 = S_q + D_q

    The scalar y_s[q] stored here is ||y_vec[q]||, where
        y_vec[q,k] = sqrt(rho_q) sqrt(a_qk) x_prime[k] / (S_q + D_q).

    The actual optimized D matrix over x_prime is diagonal:
        D_prime[k] = sum_q y_s[q]^2 * (a_qk + b_qk).

    The dummy augmented coordinate has:
        D_dummy = sum_q y_s[q]^2 * n_q,
    but it is not used to update any power variable.
    """
    P_prime = np.asarray(P_prime, dtype=float).reshape(-1)
    a_sens = np.asarray(a_sens, dtype=float)
    b_sens = np.asarray(b_sens, dtype=float)
    n_sens = np.asarray(n_sens, dtype=float).reshape(-1)

    K = P_prime.size
    Q = a_sens.shape[0]

    if a_sens.shape != b_sens.shape:
        raise ValueError(f"a_sens and b_sens must have the same shape, got {a_sens.shape} and {b_sens.shape}.")
    if a_sens.shape[1] != K:
        raise ValueError(f"a_sens has K={a_sens.shape[1]} columns but P_prime has length {K}.")

    if target_weights is None:
        omega = np.ones(Q, dtype=float)
    else:
        omega = np.asarray(target_weights, dtype=float).reshape(Q)

    # x_prime is the actual QT variable.
    x_prime = np.sqrt(np.maximum(P_prime, eps_power))

    # Full target-wise numerator and denominator.
    S, D = compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens, eps=eps_power)
    H = S + D

    # LDT and effective QT weight.
    mu_s = S / (D + eps_power)
    rho_no_ldt = float(w_s) * omega
    rho = rho_no_ldt * (1.0 + mu_s)

    sqrt_rho = np.sqrt(np.maximum(rho, 0.0))
    sqrt_a = np.sqrt(np.maximum(a_sens, 0.0))

    # Full vector QT auxiliary for numerator ||A_q x||^2 / H_q:
    # y_vec[q,k] = sqrt(rho_q) sqrt(a_qk) x_k / H_q.
    y_vec = (sqrt_rho[:, None] * sqrt_a * x_prime[None, :]) / (H[:, None] + eps_power)

    # Scalar stored for compactness: y_s[q] = ||y_vec[q]|| = sqrt(rho_q S_q)/(S_q+D_q).
    y_s = np.linalg.norm(y_vec, axis=1)

    # Linear coefficient in the surrogate 2 linear_x[k] x_prime[k] - D_prime[k] x_prime[k]^2.
    linear_x = np.sum(sqrt_rho[:, None] * sqrt_a * y_vec, axis=0)

    # Actual D matrix for optimized square-root powers.
    D_prime = np.sum((y_s ** 2)[:, None] * (a_sens + b_sens), axis=0)
    D_matrix = np.diag(D_prime)

    # Augmented dummy coordinate D entry for the constant n_q.
    D_dummy = np.sum((y_s ** 2) * n_sens)
    D_aug_diag = np.concatenate([D_prime, np.array([D_dummy], dtype=float)])
    D_aug = np.diag(D_aug_diag)

    # Nonhomogeneous-QT / gradient-projection step in x_prime.
    lambda_prime = np.maximum(D_prime, eps_lambda)
    alpha_prime = 1.0 / lambda_prime
    grad_x = linear_x - D_prime * x_prime

    return {
        "S": S,
        "D": D,
        "H": H,
        "mu_s": mu_s,
        "rho": rho,
        "rho_no_ldt": rho_no_ldt,
        "x_prime": x_prime,
        "y_vec": y_vec,
        "y_s": y_s,
        "linear": linear_x,
        "gradient": grad_x,
        "D_update": D_prime,
        "D_matrix": D_matrix,
        "D_aug": D_aug,
        "D_aug_diag": D_aug_diag,
        "D_dummy": np.array([D_dummy], dtype=float),
        "lambda": lambda_prime,
        "alpha": alpha_prime,
    }

# -----------------------------------------------------------------------------
# Sensing block update
# INTERNAL HELPER: updates P_prime using the Section-V-style D/lambda sensing step.
# -----------------------------------------------------------------------------
def update_P_prime_using_D_lambda(
    *,
    P_prime: Any,
    P_bar: Any,
    P_tilda: Any,
    P_total_max: Optional[Any],
    a_sens: Any,
    b_sens: Any,
    n_sens: Any,
    target_weights: Optional[Any],
    w_s: float,
    eps_power: float = 1e-12,
    eps_lambda: float = 1e-30,
    normalize_gradient: bool = True,
    normalization_mode: str = "sqrt_power",
    lambda_step_power: float = 0.5,
) -> tuple[Array, Dict[str, Array]]:
    """
    Section-V-style D/lambda update for sensing power P_prime.

    The update is performed in x_prime = sqrt(P_prime):
        x_new = x_old + alpha * grad_x,
    then squared back to P_prime and clipped under the joint per-user budget.

    If normalize_gradient=True, grad_x is normalized before the lambda step.
    The default normalization_mode="sqrt_power" uses sqrt(Pmax), because the
    optimized coordinate is x_prime. Use normalization_mode="power" only when
    you intentionally want the exact direct-power scaling Pmax/(K*max|grad|).
    """
    P_prime = np.asarray(P_prime, dtype=float).reshape(-1)
    D_info = build_D_lambda_P_prime_sensing(
        P_prime=P_prime,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
        eps_power=eps_power,
        eps_lambda=eps_lambda,
    )

    x_old = D_info["x_prime"]
    grad_x_raw = np.asarray(D_info["gradient"], dtype=float).reshape(-1)
    K = len(grad_x_raw)

    if normalize_gradient:
        mode = str(normalization_mode).lower().strip()
        if mode in ["sqrt_power", "sqrt", "x", "x_prime"]:
            grad_x, grad_norm_scale_prime, grad_norm_rho_prime = normalize_sqrt_power_grad_np(
                grad_x_raw,
                P_total_max=P_total_max,
                K=K,
                eps=eps_power,
            )
        elif mode in ["power", "pmax", "direct_power"]:
            grad_x, grad_norm_scale_prime, grad_norm_rho_prime = normalize_single_power_grad_np(
                grad_x_raw,
                P_total_max=P_total_max,
                K=K,
                eps=eps_power,
            )
        else:
            raise ValueError(
                "normalization_mode must be 'sqrt_power' or 'power', "
                f"got {normalization_mode!r}."
            )
    else:
        grad_x = grad_x_raw.copy()
        grad_norm_scale_prime = np.ones(K, dtype=float)
        grad_norm_rho_prime = float(max(np.max(np.abs(grad_x_raw)), eps_power))

    # alpha_prime = np.asarray(D_info["alpha"], dtype=float).reshape(K)
    lambda_prime = np.asarray(D_info["lambda"], dtype=float).reshape(K)
    alpha_prime_original = 1.0 / np.maximum(lambda_prime, eps_lambda)
    alpha_prime = 1.0 / np.power(np.maximum(lambda_prime, eps_lambda), lambda_step_power)
    effective_alpha_prime = alpha_prime * grad_norm_scale_prime

    x_candidate = x_old + alpha_prime * grad_x
    x_candidate = np.maximum(x_candidate, np.sqrt(eps_power))

    P_prime_candidate = x_candidate ** 2
    P_prime_new = clip_P_prime_joint(
        P_prime_new=P_prime_candidate,
        P_bar=P_bar,
        P_tilda=P_tilda,
        P_total_max=P_total_max,
        eps=eps_power,
    )

    D_info["alpha_original"] = alpha_prime_original
    D_info["alpha"] = alpha_prime
    D_info["lambda_step_power"] = float(lambda_step_power)
    D_info["effective_alpha"] = effective_alpha_prime
    D_info["gradient_raw"] = grad_x_raw
    D_info["gradient"] = grad_x
    D_info["gradient_normalized"] = grad_x
    D_info["gradient_normalization_scale"] = grad_norm_scale_prime
    D_info["gradient_normalization_rho"] = grad_norm_rho_prime
    D_info["normalize_gradient"] = bool(normalize_gradient)
    D_info["gradient_normalization_mode"] = str(normalization_mode)
    # D_info["effective_alpha"] = effective_alpha_prime
    D_info["x_prime_candidate"] = x_candidate
    D_info["P_prime_candidate"] = P_prime_candidate
    return P_prime_new, D_info

# -----------------------------------------------------------------------------
# Joint state evaluation
# INTERNAL HELPER: evaluates communication, sensing and joint objective for current powers.
# -----------------------------------------------------------------------------
def evaluate_joint_lambda_state(
    *,
    K: int,
    w: Any,
    d: float,
    tau: float,
    w_c: float,
    w_s: float,
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
    P_prime: Any,
    a_sens: Any,
    b_sens: Any,
    n_sens: Any,
    target_weights: Optional[Any],
    compute_sinr_exact_fn: Callable[..., float],
    compute_wsr_exact_fn: Callable[..., float],
) -> Dict[str, Any]:
    """Evaluate communication, sensing and joint objective for current powers."""
    P_bar = np.asarray(P_bar, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda, dtype=float).reshape(K)
    P_prime = np.asarray(P_prime, dtype=float).reshape(K)
    w = np.asarray(w, dtype=float).reshape(K)

    kappa_Th1 = np.asarray(kappa_Th1, dtype=float).reshape(K)
    kappa_ADC1 = np.asarray(kappa_ADC1, dtype=float).reshape(K)
    kappa_Th0 = np.asarray(kappa_Th0, dtype=float).reshape(K)
    kappa_ADC0 = np.asarray(kappa_ADC0, dtype=float).reshape(K)

    P_th = P_bar * kappa_Th1 + kappa_Th0
    P_adc = P_bar * kappa_ADC1 + kappa_ADC0

    gamma = np.array([
        compute_sinr_exact_fn(
            k, K,
            kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1, kappa_ADC1,
            P_bar, P_tilda,
            P_th, P_adc,
        )
        for k in range(K)
    ], dtype=float)

    comm_wsr = float(compute_wsr_exact_fn(
        K, w,
        kappa_S,
        kappa_V1, kappa_K1,
        kappa_V0, kappa_K0,
        kappa_DAC1, kappa_DAC0,
        kappa_M1, kappa_M0,
        kappa_Th1, kappa_ADC1,
        P_bar, P_tilda,
        P_th, P_adc,
        d, tau,
    ))

    sensing_wsr = compute_sensing_wsr(P_prime, a_sens, b_sens, n_sens, target_weights, w_s=w_s)
    joint_obj = float(float(w_c) * comm_wsr + sensing_wsr)
    return {
        "joint_obj": joint_obj,
        "comm_wsr": comm_wsr,
        "comm_contribution": float(float(w_c) * comm_wsr),
        "sensing_wsr": sensing_wsr,
        "gamma": gamma,
        "P_th": P_th,
        "P_adc": P_adc,
    }


# -----------------------------------------------------------------------------
# Main optimizer
# MAIN-FILE API: imported by the main file to run joint communication-sensing lambda-QT.
# -----------------------------------------------------------------------------
def run_joint_lambda_qt(
    *,
    K: int,
    w: Any,
    d: float,
    tau: float,
    w_c: float,
    w_s: float,
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
    a_sens: Any,
    b_sens: Any,
    n_sens: Any,
    target_weights: Optional[Any],
    P_bar_init: Any,
    P_tilda_init: Any,
    P_prime_init: Any,
    P_total_max: Optional[Any] = None,
    max_iters: int = 200,
    epsilon: float = 1e-6,
    eps_power: float = 1e-12,
    eps_lambda: float = 1e-30,
    lambda_mode_bar: str = "actual",
    normalize_lambda_gradients: bool = True,
    prime_normalization_mode: str = "sqrt_power",
    lambda_step_power: float = 0.5,
    use_backtracking: bool = True,
    min_eta: float = 1e-6,
    verbose: bool = False,
    print_step_sizes: bool = False,
    step_print_every: int = 50,
    step_print_last_n: int = 10,
    compute_sinr_exact_fn: Callable[..., float] | None = None,
    compute_wsr_exact_fn: Callable[..., float] | None = None,
    update_auxiliary_fn: Callable[..., Array] | None = None,
) -> Dict[str, Any]:
    """
    Joint D/lambda-QT optimizer for P_bar, P_tilda and P_prime.

    If normalize_lambda_gradients=True, all three lambda-gradient blocks are
    normalized before applying alpha=1/lambda. P_bar/P_tilda use the normalized
    communication accelerated-QT update; P_prime uses x_prime=sqrt(P_prime)
    normalization by default.
    """
    if compute_sinr_exact_fn is None:
        raise ValueError("Pass compute_sinr_exact_fn=compute_SINR_exact from your main code.")
    if compute_wsr_exact_fn is None:
        raise ValueError("Pass compute_wsr_exact_fn=compute_WSR_exact from your main code.")
    if update_auxiliary_fn is None:
        raise ValueError("Pass update_auxiliary_fn=update_auxiliary from your main code.")

    P_bar = np.asarray(P_bar_init, dtype=float).reshape(K).copy()
    P_tilda = np.asarray(P_tilda_init, dtype=float).reshape(K).copy()
    P_prime = np.asarray(P_prime_init, dtype=float).reshape(K).copy()
    w = np.asarray(w, dtype=float).reshape(K)

    P_bar, P_tilda, P_prime = project_joint_three_power_blocks(
        P_bar, P_tilda, P_prime, P_total_max, eps=eps_power
    )

    initial_state = evaluate_joint_lambda_state(
        K=K, w=w, d=d, tau=tau, w_c=w_c, w_s=w_s,
        kappa_S=kappa_S, kappa_V1=kappa_V1, kappa_K1=kappa_K1,
        kappa_V0=kappa_V0, kappa_K0=kappa_K0,
        kappa_DAC1=kappa_DAC1, kappa_DAC0=kappa_DAC0,
        kappa_Th1=kappa_Th1, kappa_ADC1=kappa_ADC1,
        kappa_M1=kappa_M1, kappa_M0=kappa_M0,
        kappa_Th0=kappa_Th0, kappa_ADC0=kappa_ADC0,
        P_bar=P_bar, P_tilda=P_tilda, P_prime=P_prime,
        a_sens=a_sens, b_sens=b_sens, n_sens=n_sens,
        target_weights=target_weights,
        compute_sinr_exact_fn=compute_sinr_exact_fn,
        compute_wsr_exact_fn=compute_wsr_exact_fn,
    )

    full_joint_history = [float(initial_state["joint_obj"])]
    joint_history = []
    comm_history = []
    sensing_history = []

    D_bar_aug_history, D_bar_update_history, lambda_bar_history, alpha_bar_history, grad_bar_history = [], [], [], [], []
    grad_bar_raw_history, effective_alpha_bar_history, grad_bar_norm_scale_history = [], [], []

    D_tilda_matrix_history, D_tilda_update_history, lambda_tilda_history, alpha_tilda_history, grad_tilda_history = [], [], [], [], []
    grad_tilda_raw_history, effective_alpha_tilda_history, grad_tilda_norm_scale_history = [], [], []

    D_prime_update_history, lambda_prime_history, alpha_prime_history, grad_prime_history = [], [], [], []
    grad_prime_raw_history, effective_alpha_prime_history, grad_prime_norm_scale_history = [], [], []

    sensing_mu_history, sensing_y_history, sensing_S_history, sensing_D_history = [], [], [], []

    P_bar_history = [P_bar.copy()]
    P_tilda_history = [P_tilda.copy()]
    P_prime_history = [P_prime.copy()]

    converged = False

    for t in range(max_iters):
        old_P_bar = P_bar.copy()
        old_P_tilda = P_tilda.copy()
        old_P_prime = P_prime.copy()
        old_joint = float(full_joint_history[-1])

        old_state = evaluate_joint_lambda_state(
            K=K, w=w, d=d, tau=tau, w_c=w_c, w_s=w_s,
            kappa_S=kappa_S, kappa_V1=kappa_V1, kappa_K1=kappa_K1,
            kappa_V0=kappa_V0, kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1, kappa_DAC0=kappa_DAC0,
            kappa_Th1=kappa_Th1, kappa_ADC1=kappa_ADC1,
            kappa_M1=kappa_M1, kappa_M0=kappa_M0,
            kappa_Th0=kappa_Th0, kappa_ADC0=kappa_ADC0,
            P_bar=old_P_bar, P_tilda=old_P_tilda, P_prime=old_P_prime,
            a_sens=a_sens, b_sens=b_sens, n_sens=n_sens,
            target_weights=target_weights,
            compute_sinr_exact_fn=compute_sinr_exact_fn,
            compute_wsr_exact_fn=compute_wsr_exact_fn,
        )

        gamma = old_state["gamma"]
        P_th = old_state["P_th"]
        P_adc = old_state["P_adc"]

        # Communication auxiliary uses the contribution weight w_c * d/tau * w,
        # consistent with the joint classical optimizer in the main code.
        w_comm_eff = float(w_c) * (float(d) / float(tau)) * w

        mu = update_auxiliary_fn(
            K, w_comm_eff, gamma,
            kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1, kappa_ADC1,
            old_P_bar,
            old_P_tilda,
            P_th,
            P_adc,
        )

        # 1) P_bar update with old P_tilda and old P_prime.
        P_bar_target, D_bar_info = update_P_bar_using_D_lambda(
            K=K,
            w=w_comm_eff,
            gamma=gamma,
            mu=mu,
            P_bar=old_P_bar,
            P_tilda=old_P_tilda,
            P_prime_fixed=old_P_prime,
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
            lambda_step_power=lambda_step_power,
        )

        # 2) P_tilda update with updated P_bar and old P_prime.
        P_tilda_target, D_tilda_info = update_P_tilda_using_D_lambda(
            K=K,
            w=w_comm_eff,
            gamma=gamma,
            mu=mu,
            P_bar=P_bar_target,
            P_tilda=old_P_tilda,
            P_prime_fixed=old_P_prime,
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
            lambda_step_power=lambda_step_power,
        )

        # 3) P_prime update with updated communication powers.
        P_prime_target, D_prime_info = update_P_prime_using_D_lambda(
            P_prime=old_P_prime,
            P_bar=P_bar_target,
            P_tilda=P_tilda_target,
            P_total_max=P_total_max,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_s=w_s,
            eps_power=eps_power,
            eps_lambda=eps_lambda,
            normalize_gradient=normalize_lambda_gradients,
            normalization_mode=prime_normalization_mode,
            lambda_step_power=lambda_step_power,
        )

        eta = 1.0
        accepted = False

        if use_backtracking:
            while eta >= min_eta:
                cand_bar = old_P_bar + eta * (P_bar_target - old_P_bar)
                cand_tilda = old_P_tilda + eta * (P_tilda_target - old_P_tilda)
                cand_prime = old_P_prime + eta * (P_prime_target - old_P_prime)
                cand_bar, cand_tilda, cand_prime = project_joint_three_power_blocks(
                    cand_bar, cand_tilda, cand_prime, P_total_max, eps=eps_power
                )
                cand_state = evaluate_joint_lambda_state(
                    K=K, w=w, d=d, tau=tau, w_c=w_c, w_s=w_s,
                    kappa_S=kappa_S, kappa_V1=kappa_V1, kappa_K1=kappa_K1,
                    kappa_V0=kappa_V0, kappa_K0=kappa_K0,
                    kappa_DAC1=kappa_DAC1, kappa_DAC0=kappa_DAC0,
                    kappa_Th1=kappa_Th1, kappa_ADC1=kappa_ADC1,
                    kappa_M1=kappa_M1, kappa_M0=kappa_M0,
                    kappa_Th0=kappa_Th0, kappa_ADC0=kappa_ADC0,
                    P_bar=cand_bar, P_tilda=cand_tilda, P_prime=cand_prime,
                    a_sens=a_sens, b_sens=b_sens, n_sens=n_sens,
                    target_weights=target_weights,
                    compute_sinr_exact_fn=compute_sinr_exact_fn,
                    compute_wsr_exact_fn=compute_wsr_exact_fn,
                )
                if cand_state["joint_obj"] >= old_joint - 1e-12:
                    P_bar, P_tilda, P_prime = cand_bar, cand_tilda, cand_prime
                    new_state = cand_state
                    accepted = True
                    break
                eta *= 0.5

            if not accepted:
                P_bar, P_tilda, P_prime = old_P_bar, old_P_tilda, old_P_prime
                new_state = old_state
                eta = 0.0
        else:
            P_bar, P_tilda, P_prime = project_joint_three_power_blocks(
                P_bar_target, P_tilda_target, P_prime_target, P_total_max, eps=eps_power
            )
            new_state = evaluate_joint_lambda_state(
                K=K, w=w, d=d, tau=tau, w_c=w_c, w_s=w_s,
                kappa_S=kappa_S, kappa_V1=kappa_V1, kappa_K1=kappa_K1,
                kappa_V0=kappa_V0, kappa_K0=kappa_K0,
                kappa_DAC1=kappa_DAC1, kappa_DAC0=kappa_DAC0,
                kappa_Th1=kappa_Th1, kappa_ADC1=kappa_ADC1,
                kappa_M1=kappa_M1, kappa_M0=kappa_M0,
                kappa_Th0=kappa_Th0, kappa_ADC0=kappa_ADC0,
                P_bar=P_bar, P_tilda=P_tilda, P_prime=P_prime,
                a_sens=a_sens, b_sens=b_sens, n_sens=n_sens,
                target_weights=target_weights,
                compute_sinr_exact_fn=compute_sinr_exact_fn,
                compute_wsr_exact_fn=compute_wsr_exact_fn,
            )
            accepted = True

        full_joint_history.append(float(new_state["joint_obj"]))
        joint_history.append(float(new_state["joint_obj"]))
        comm_history.append(float(new_state["comm_wsr"]))
        sensing_history.append(float(new_state["sensing_wsr"]))

        D_bar_aug_history.append(D_bar_info["D_aug"].copy())
        D_bar_update_history.append(D_bar_info["D_update"].copy())
        lambda_bar_history.append(D_bar_info["lambda"].copy())
        grad_bar_history.append(D_bar_info["gradient"].copy())
        grad_bar_raw_history.append(_copy_info_array(D_bar_info, "gradient_raw", "gradient"))
        alpha_bar_history.append(D_bar_info["alpha"].copy())
        effective_alpha_bar_history.append(_copy_info_array(D_bar_info, "effective_alpha", "alpha"))
        grad_bar_norm_scale_history.append(
            _copy_info_array(D_bar_info, "gradient_normalization_scale")
            if "gradient_normalization_scale" in D_bar_info
            else np.ones_like(np.asarray(D_bar_info["alpha"], dtype=float))
        )

        D_tilda_matrix_history.append(D_tilda_info["D_matrix"].copy())
        D_tilda_update_history.append(D_tilda_info["D_update"].copy())
        lambda_tilda_history.append(D_tilda_info["lambda"].copy())
        grad_tilda_history.append(D_tilda_info["gradient"].copy())
        grad_tilda_raw_history.append(_copy_info_array(D_tilda_info, "gradient_raw", "gradient"))
        alpha_tilda_history.append(D_tilda_info["alpha"].copy())
        effective_alpha_tilda_history.append(_copy_info_array(D_tilda_info, "effective_alpha", "alpha"))
        grad_tilda_norm_scale_history.append(
            _copy_info_array(D_tilda_info, "gradient_normalization_scale")
            if "gradient_normalization_scale" in D_tilda_info
            else np.ones_like(np.asarray(D_tilda_info["alpha"], dtype=float))
        )

        D_prime_update_history.append(D_prime_info["D_update"].copy())
        lambda_prime_history.append(D_prime_info["lambda"].copy())
        grad_prime_history.append(D_prime_info["gradient"].copy())
        grad_prime_raw_history.append(_copy_info_array(D_prime_info, "gradient_raw", "gradient"))
        alpha_prime_history.append(D_prime_info["alpha"].copy())
        effective_alpha_prime_history.append(_copy_info_array(D_prime_info, "effective_alpha", "alpha"))
        grad_prime_norm_scale_history.append(
            _copy_info_array(D_prime_info, "gradient_normalization_scale")
            if "gradient_normalization_scale" in D_prime_info
            else np.ones_like(np.asarray(D_prime_info["alpha"], dtype=float))
        )
        sensing_mu_history.append(D_prime_info["mu_s"].copy())
        sensing_y_history.append(D_prime_info["y_s"].copy())
        sensing_S_history.append(D_prime_info["S"].copy())
        sensing_D_history.append(D_prime_info["D"].copy())

        P_bar_history.append(P_bar.copy())
        P_tilda_history.append(P_tilda.copy())
        P_prime_history.append(P_prime.copy())

        delta = float(new_state["joint_obj"] - old_joint)
        if verbose:
            print(
                f"joint-lambda iter={t + 1:03d} | joint={new_state['joint_obj']:.8f} | "
                f"comm={new_state['comm_wsr']:.8f} | sensing={new_state['sensing_wsr']:.8f} | "
                f"delta={delta:.3e} | eta={eta:.3e} | accepted={accepted}"
            )

        if print_step_sizes:
            every = max(1, int(step_print_every))
            if t == 0 or ((t + 1) % every == 0):
                print(
                    f"[JOINT STEP TRACE] iter={t + 1:04d} | "
                    f"alpha_bar mean={np.mean(D_bar_info['alpha']):.6e} | "
                    f"alpha_tilda mean={np.mean(D_tilda_info['alpha']):.6e} | "
                    f"alpha_prime mean={np.mean(D_prime_info['alpha']):.6e} | "
                    f"eff_alpha_prime mean={np.mean(D_prime_info.get('effective_alpha', D_prime_info['alpha'])):.6e} | "
                    f"lambda_prime mean={np.mean(D_prime_info['lambda']):.6e} | "
                    f"normalized={normalize_lambda_gradients} | eta={eta:.3e}"
                )

        if t > 0 and abs(delta) < epsilon:
            converged = True
            break

    final_state = evaluate_joint_lambda_state(
        K=K, w=w, d=d, tau=tau, w_c=w_c, w_s=w_s,
        kappa_S=kappa_S, kappa_V1=kappa_V1, kappa_K1=kappa_K1,
        kappa_V0=kappa_V0, kappa_K0=kappa_K0,
        kappa_DAC1=kappa_DAC1, kappa_DAC0=kappa_DAC0,
        kappa_Th1=kappa_Th1, kappa_ADC1=kappa_ADC1,
        kappa_M1=kappa_M1, kappa_M0=kappa_M0,
        kappa_Th0=kappa_Th0, kappa_ADC0=kappa_ADC0,
        P_bar=P_bar, P_tilda=P_tilda, P_prime=P_prime,
        a_sens=a_sens, b_sens=b_sens, n_sens=n_sens,
        target_weights=target_weights,
        compute_sinr_exact_fn=compute_sinr_exact_fn,
        compute_wsr_exact_fn=compute_wsr_exact_fn,
    )
    full_joint_history_arr = np.asarray(full_joint_history, dtype=float)

    result = {
        "method": "normalized_joint_lambda_unfolding_gradient_qt" if normalize_lambda_gradients else "joint_lambda_unfolding_gradient_qt",
        "normalize_lambda_gradients": bool(normalize_lambda_gradients),
        "prime_normalization_mode": str(prime_normalization_mode),
        "initial_joint_wsr": float(full_joint_history_arr[0]),
        "final_joint_wsr": float(full_joint_history_arr[-1]),
        "final_wsr": float(full_joint_history_arr[-1]),
        "joint_history": np.asarray(joint_history, dtype=float),
        "full_joint_history": full_joint_history_arr,
        "comm_history": np.asarray(comm_history, dtype=float),
        "sensing_history": np.asarray(sensing_history, dtype=float),
        "P_bar_initial": np.asarray(P_bar_init, dtype=float).reshape(K),
        "P_tilda_initial": np.asarray(P_tilda_init, dtype=float).reshape(K),
        "P_prime_initial": np.asarray(P_prime_init, dtype=float).reshape(K),
        "P_bar_opt": P_bar,
        "P_tilda_opt": P_tilda,
        "P_prime_opt": P_prime,
        "gamma_opt": final_state["gamma"],
        "P_th_opt": final_state["P_th"],
        "P_adc_opt": final_state["P_adc"],
        "sensing_final": compute_sensing_components(P_prime, a_sens, b_sens, n_sens, target_weights, w_s=w_s),
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
        "D_prime_update_history": np.asarray(D_prime_update_history, dtype=float),
        "lambda_prime_history": np.asarray(lambda_prime_history, dtype=float),
        "grad_prime_history": np.asarray(grad_prime_history, dtype=float),
        "grad_prime_raw_history": np.asarray(grad_prime_raw_history, dtype=float),
        "alpha_prime_history": np.asarray(alpha_prime_history, dtype=float),
        "effective_alpha_prime_history": np.asarray(effective_alpha_prime_history, dtype=float),
        "grad_prime_norm_scale_history": np.asarray(grad_prime_norm_scale_history, dtype=float),
        "sensing_mu_history": np.asarray(sensing_mu_history, dtype=float),
        "sensing_y_history": np.asarray(sensing_y_history, dtype=float),
        "sensing_S_history": np.asarray(sensing_S_history, dtype=float),
        "sensing_D_history": np.asarray(sensing_D_history, dtype=float),
        "P_bar_history": np.asarray(P_bar_history, dtype=float),
        "P_tilda_history": np.asarray(P_tilda_history, dtype=float),
        "P_prime_history": np.asarray(P_prime_history, dtype=float),
        "monotonic": bool(np.all(np.diff(full_joint_history_arr) >= -1e-10)),
        "converged": converged,
        "iterations": len(joint_history),
        "w_c": float(w_c),
        "w_s": float(w_s),
    }

    if print_step_sizes:
        print_joint_lambda_step_size_summary(result, last_n=step_print_last_n)

    return result


# -----------------------------------------------------------------------------
# Public reporting API
# MAIN-FILE API: imported by the main file to print joint alpha/lambda summaries.
# -----------------------------------------------------------------------------
def print_joint_lambda_step_size_summary(result: Dict[str, Any], *, last_n: int = 10) -> None:
    """Print alpha/lambda summaries for P_bar, P_tilda and P_prime."""
    alpha_bar = _flatten_history(result.get("alpha_bar_history", []))
    alpha_tilda = _flatten_history(result.get("alpha_tilda_history", []))
    alpha_prime = _flatten_history(result.get("alpha_prime_history", []))
    lambda_bar = _flatten_history(result.get("lambda_bar_history", []))
    lambda_tilda = _flatten_history(result.get("lambda_tilda_history", []))
    lambda_prime = _flatten_history(result.get("lambda_prime_history", []))
    eff_alpha_bar = _flatten_history(result.get("effective_alpha_bar_history", []))
    eff_alpha_tilda = _flatten_history(result.get("effective_alpha_tilda_history", []))
    eff_alpha_prime = _flatten_history(result.get("effective_alpha_prime_history", []))

    print("\n" + "=" * 110)
    print("Joint D/lambda QT alpha/lambda summary")
    print("=" * 110)
    print(_history_stats_line("alpha_bar", alpha_bar))
    print(_history_stats_line("alpha_tilda", alpha_tilda))
    print(_history_stats_line("alpha_prime", alpha_prime))
    if eff_alpha_bar.size:
        print(_history_stats_line("effective_alpha_bar", eff_alpha_bar))
    if eff_alpha_tilda.size:
        print(_history_stats_line("effective_alpha_tilda", eff_alpha_tilda))
    if eff_alpha_prime.size:
        print(_history_stats_line("effective_alpha_prime", eff_alpha_prime))
    print(_history_stats_line("lambda_bar", lambda_bar))
    print(_history_stats_line("lambda_tilda", lambda_tilda))
    print(_history_stats_line("lambda_prime", lambda_prime))

    if alpha_prime.size == 0:
        return
    n_iter = alpha_prime.shape[0]
    start = max(0, n_iter - int(last_n))
    print("\nLast joint step-size values, averaged over users:")
    print(
        f"{'iter':>6} | {'mean alpha_bar':>16} | {'mean alpha_tilda':>18} | {'mean alpha_prime':>18} | "
        f"{'mean eff_alpha_prime':>22} | {'mean lambda_prime':>20}"
    )
    print("-" * 119)
    for i in range(start, n_iter):
        eff_prime_mean = eff_alpha_prime[i].mean() if eff_alpha_prime.size else float("nan")
        print(
            f"{i + 1:6d} | {alpha_bar[i].mean():16.6e} | {alpha_tilda[i].mean():18.6e} | "
            f"{alpha_prime[i].mean():18.6e} | {eff_prime_mean:22.6e} | {lambda_prime[i].mean():20.6e}"
        )


# -----------------------------------------------------------------------------
# Public reporting API
# MAIN-FILE API: imported by the main file to save joint alpha/lambda summaries.
# -----------------------------------------------------------------------------
def save_joint_lambda_step_size_summary(result: Dict[str, Any], save_path: str) -> None:
    """Save per-iteration mean alpha/lambda values for all three joint blocks."""
    alpha_bar = _flatten_history(result.get("alpha_bar_history", []))
    alpha_tilda = _flatten_history(result.get("alpha_tilda_history", []))
    alpha_prime = _flatten_history(result.get("alpha_prime_history", []))
    lambda_bar = _flatten_history(result.get("lambda_bar_history", []))
    lambda_tilda = _flatten_history(result.get("lambda_tilda_history", []))
    lambda_prime = _flatten_history(result.get("lambda_prime_history", []))
    eff_alpha_bar = _flatten_history(result.get("effective_alpha_bar_history", []))
    eff_alpha_tilda = _flatten_history(result.get("effective_alpha_tilda_history", []))
    eff_alpha_prime = _flatten_history(result.get("effective_alpha_prime_history", []))
    if alpha_bar.size == 0 or alpha_tilda.size == 0 or alpha_prime.size == 0:
        raise ValueError("No joint alpha history found. Run joint_lambda before saving step sizes.")
    n_iter = alpha_bar.shape[0]
    if eff_alpha_bar.size == 0:
        eff_alpha_bar = alpha_bar
    if eff_alpha_tilda.size == 0:
        eff_alpha_tilda = alpha_tilda
    if eff_alpha_prime.size == 0:
        eff_alpha_prime = alpha_prime
    table = np.column_stack([
        np.arange(1, n_iter + 1),
        alpha_bar.mean(axis=1),
        alpha_tilda.mean(axis=1),
        alpha_prime.mean(axis=1),
        eff_alpha_bar.mean(axis=1),
        eff_alpha_tilda.mean(axis=1),
        eff_alpha_prime.mean(axis=1),
        lambda_bar.mean(axis=1),
        lambda_tilda.mean(axis=1),
        lambda_prime.mean(axis=1),
    ])
    np.savetxt(
        save_path,
        table,
        header=(
            "iter mean_alpha_bar mean_alpha_tilda mean_alpha_prime "
            "mean_effective_alpha_bar mean_effective_alpha_tilda mean_effective_alpha_prime "
            "mean_lambda_bar mean_lambda_tilda mean_lambda_prime"
        ),
        fmt=["%d", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e"],
    )
    print(f"[SAVE] joint lambda/D step-size summary: {save_path}")


# -----------------------------------------------------------------------------
# Public plotting API
# MAIN-FILE API: imported by the main file to plot joint classical/lambda/DU histories.
# -----------------------------------------------------------------------------
def plot_joint_method_comparison(
    *,
    initial_value: Optional[float] = None,
    classical_result: Optional[Dict[str, Any]] = None,
    lambda_qt_result: Optional[Dict[str, Any]] = None,
    du_history: Optional[Any] = None,
    title: str = "Joint: EPA vs classical vs D/lambda QT vs unfolding",
    save_path: Optional[str] = None,
) -> None:
    """Plot joint convergence histories."""
    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")

    plt.figure(figsize=(8, 5))
    if initial_value is not None:
        plt.plot([0], [float(initial_value)], marker="x", markersize=9, label="EPA / initial")
    if classical_result is not None:
        plt.plot(
            np.asarray(classical_result["full_joint_history"], dtype=float).reshape(-1),
            marker="o",
            linewidth=2,
            label="Joint classical FP/QT",
        )
    if lambda_qt_result is not None:
        plt.plot(
            np.asarray(lambda_qt_result["full_joint_history"], dtype=float).reshape(-1),
            marker="s",
            linewidth=2,
            label="Joint D/lambda-QT",
        )
    if du_history is not None and len(np.asarray(du_history).reshape(-1)) > 0:
        plt.plot(
            np.asarray(du_history, dtype=float).reshape(-1),
            marker="^",
            linewidth=2,
            label="Joint deep unfolding",
        )
    plt.xlabel("Iteration / layer")
    plt.ylabel("Joint objective")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
