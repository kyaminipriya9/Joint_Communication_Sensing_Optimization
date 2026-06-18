"""Joint communication-sensing FP/QT power optimization.

This script optimizes three per-user power blocks:
    P_bar[k]   : pilot / channel-estimation power
    P_tilda[k] : communication data power
    P_prime[k] : sensing power

The per-user constraint is
    P_bar[k] + P_tilda[k] + P_prime[k] <= Pmax.

The joint objective is
    w_c * WSR_comm + w_s * sum_q omega_q log2(1 + SINR_sens,q).
"""

import os
import time
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
from scipy.io import loadmat #type: ignore
from kappa import (compute_kappa_S, compute_kappa_V1, compute_kappa_V0,
                   compute_kappa_K1, compute_kappa_K0,
                   compute_kappa_DAC1, compute_kappa_DAC0,compute_kappa_Th1,compute_kappa_ADC1,
                   compute_g_lik)
from kappa_mui import build_kappa_MUI
from matrix_builders import (build_all_matrices, build_R_tilda_all, build_C_yy_all, build_G_list)

import torch  # type: ignore
import torch.optim as optim  # type: ignore

from unfolding_for_jointopt_latest import (
    JointCommSensingUnfoldingPGD,
    to_torch_1d,
    to_torch_2d,
    scalar_to_torch,
)

   
# Utility
def _slice_k(lst_2d, k):
    """1-D slice over panels for fixed user k."""
    return [lst_2d[l][k] for l in range(len(lst_2d))]


# Build every scalar kappa — called ONCE before the iteration loop
def build_all_kappas(K, alpha_list, tau, d,
                     U_list, V_list, U_q_list,V_q_list, Sigma_list,
                     R_list, h_bar_list,
                     C_h_ref_all, C_h_total_all,
                     sigma_d2, sigma_w2,D_Th_list, D_ADC_list,
                     A_list, Theta_list,
                     Z_list,
                     g_lik):
    """
    Returns
    -------
    kappa_S    : (K,)
    kappa_V1   : (K,)
    kappa_V0   : (K,)
    kappa_K1   : (K,)
    kappa_K0   : (K,)
    kappa_DAC1 : (K,)
    kappa_DAC0 : (K,)
    kappa_M1   : (K,K)
    kappa_M0   : (K,K)
    """
    kappa_S    = np.zeros(K)
    kappa_V1   = np.zeros(K)
    kappa_V0   = np.zeros(K)
    kappa_K1   = np.zeros(K)
    kappa_K0   = np.zeros(K)
    kappa_DAC1 = np.zeros(K)
    kappa_DAC0 = np.zeros(K)
    kappa_Th1 = np.zeros(K)
    kappa_ADC1 = np.zeros(K)
    kappa_Th0 = np.zeros(K)
    kappa_ADC0 = np.zeros(K)
    for k in range(K):
        U_k     = _slice_k(U_list,      k)
        V_k     = _slice_k(V_list,      k)
        Uq_k     = _slice_k(U_q_list,      k)
        Vq_k     = _slice_k(V_q_list,      k)
        Sigma_k = _slice_k(Sigma_list,  k)
        h_bar_k = _slice_k(h_bar_list,  k)
        R_k     = _slice_k(R_list,      k)
        C_ref_k = _slice_k(C_h_ref_all, k)

        kappa_S[k] = compute_kappa_S(
            alpha_list[k], tau, U_k, V_k, Sigma_k)

        kappa_V1[k] = compute_kappa_V1(
            alpha_list[k], tau,
            U_k, V_k, Sigma_k, R_k, h_bar_k, C_ref_k)

        kappa_V0[k] = compute_kappa_V0(
            alpha_list[k], tau, k,
            U_k, V_k, Sigma_k, R_k,
            h_bar_list,
            C_ref_k, C_h_total_all,
            sigma_d2, sigma_w2,
            Uq_k, Vq_k,
            Theta_list)

        kappa_K1[k] = compute_kappa_K1(
            alpha_list[k], tau,
            U_k, V_k, Sigma_k, C_ref_k)

        kappa_K0[k] = compute_kappa_K0(
            alpha_list[k], tau, k,
            U_k, V_k, Sigma_k, C_ref_k, C_h_total_all,
            sigma_d2, sigma_w2,
            Uq_k, Vq_k,
            Theta_list)

        kappa_DAC1[k] = compute_kappa_DAC1(
            k, K, alpha_list, tau, d,
            U_list, V_list,
            Sigma_list, C_h_ref_all,
            R_list, h_bar_list, sigma_d2)

        kappa_DAC0[k] = compute_kappa_DAC0(
            k, K, alpha_list, tau, d,
            U_list, V_list, C_h_total_all,
            A_list, Theta_list,
            sigma_d2, sigma_w2,
            g_lik)
        kappa_Th1[k] = compute_kappa_Th1(k, alpha_list,
                         sigma_w2, tau, d,
                        D_Th_list, C_h_total_all)
        kappa_ADC1[k] = compute_kappa_ADC1(k, alpha_list, tau, d,
                       D_ADC_list,
                       C_h_total_all)
    kappa_M1, kappa_M0 = build_kappa_MUI(
        K, alpha_list, Z_list,
        Sigma_list, C_h_ref_all, C_h_total_all,
        U_list, V_list, R_list, h_bar_list,
        sigma_d2, sigma_w2, tau,
        U_q_list, V_q_list,
            Theta_list)

    return (kappa_S,
            kappa_V1, kappa_V0,
            kappa_K1, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_Th1,kappa_ADC1,
            kappa_M1,  kappa_M0)

def build_kappa_Th0_ADC0(K, tau, d, sigma_w2,
                         sigma_d2,
                         G_list,
                         W_list,
                         A_list,
                         Theta_list,
                         C_h_total_all):
    """
    Computes the pilot-power-independent parts of P_th and P_adc.

    P_th,k  = P_bar[k] * kappa_Th1[k]  + kappa_Th0[k]
    P_adc,k = P_bar[k] * kappa_ADC1[k] + kappa_ADC0[k]

    This assumes G_lk is fixed because Cyy^{-1} is fixed.
    """

    L = len(A_list)

    kappa_Th0 = np.zeros(K)
    kappa_ADC0 = np.zeros(K)

    for l in range(L):
        A_l = A_list[l]
        Theta_l = Theta_list[l]
        N = A_l.shape[0]

        # Pilot-power-independent covariance part of the projected observation
        C0_l = np.zeros((N, N), dtype=complex)

        # DAC distortion contribution
        for i in range(K):
            C0_l += sigma_d2[i] * (
                A_l @ C_h_total_all[l][i] @ A_l.conj().T
            )

        # Thermal noise contribution
        C0_l += sigma_w2 * (A_l @ A_l.conj().T)

        # ADC quantization contribution
        C0_l += Theta_l

        # Projection length factor
        C0_l *= tau

        for k in range(K):
            G_lk = G_list[l][k]
            W_lk = W_list[l][k]

            # Independent part of S_lk = E[hhat hhat^H]
            S0_lk = G_lk @ C0_l @ G_lk.conj().T

            th_mat = (
                W_lk.conj().T
                @ A_l.conj().T
                @ A_l
                @ W_lk
                @ S0_lk
            )

            adc_mat = (
                W_lk.conj().T
                @ Theta_l
                @ W_lk
                @ S0_lk
            )

            kappa_Th0[k] += sigma_w2 * tau * d * np.real(np.trace(th_mat))
            kappa_ADC0[k] += tau * d * np.real(np.trace(adc_mat))

    return kappa_Th0, kappa_ADC0

def build_theta_list_for_power(A_list, C_h_total_all, alpha_list, P_tot_list, sigma_w2):
    Theta_list = []

    for l, A_l in enumerate(A_list):
        N = A_l.shape[0]
        I = np.eye(N, dtype=complex)

        S_full = np.zeros((N, N), dtype=complex)

        for i in range(len(alpha_list)):
            S_full += alpha_list[i] * P_tot_list[i] * C_h_total_all[l][i]

        S_full += sigma_w2 * I

        # MATLAB matching:
        # S_l = diag(diag(S_l_full))
        S_l = np.diag(np.real(np.diag(S_full))).astype(complex)

        Theta_l = A_l @ (I - A_l) @ S_l
        # Theta_l = 0.5 * (Theta_l + Theta_l.conj().T)

        Theta_list.append(Theta_l)

    return Theta_list
    
# Precompute P_th and P_adc for all users — called ONCE before iteration loop
def compute_Ak(k, kappa_S, P_tilda):
    return P_tilda[k] * kappa_S[k]


def compute_Bk(k, K, kappa_V1, kappa_K1, kappa_DAC1, kappa_M1, P_tilda):
    self_term = P_tilda[k] * (kappa_V1[k] + kappa_K1[k])
    mui_term  = sum(P_tilda[i] * kappa_M1[k, i]
                    for i in range(K) if i != k)
    return self_term + mui_term + kappa_DAC1[k]


def compute_Ck(k, K, kappa_V0, kappa_K0, kappa_DAC0, kappa_M0, P_tilda):
    self_term = P_tilda[k] * (kappa_V0[k] + kappa_K0[k])
    mui_term  = sum(P_tilda[i] * kappa_M0[k, i]
                    for i in range(K) if i != k)
    return self_term + mui_term + kappa_DAC0[k]


# SINR and WSR
def compute_SINR_exact(k, K,
                       kappa_S,
                       kappa_V1, kappa_K1,
                       kappa_V0, kappa_K0,
                       kappa_DAC1, kappa_DAC0,
                       kappa_M1, kappa_M0,
                       kappa_Th1,kappa_ADC1,
                       P_bar, P_tilda,P_th, P_adc):
    """
    SINR_k = A_k P̄_k / ( B_k P̄_k + C_k + P_th[k] + P_adc[k] )
    """
    Ak = compute_Ak(k, kappa_S, P_tilda)
    Bk = compute_Bk(k, K, kappa_V1, kappa_K1, kappa_DAC1, kappa_M1, P_tilda)
    Ck = compute_Ck(k, K, kappa_V0, kappa_K0, kappa_DAC0, kappa_M0, P_tilda)
    den = (
        Bk * P_bar[k]
        + Ck
        + P_th[k]
        + P_adc[k]
    )

    return (Ak * P_bar[k]) / (den + 1e-30)

def print_power_components(label, K,
                           kappa_S,
                           kappa_V1, kappa_K1,
                           kappa_V0, kappa_K0,
                           kappa_DAC1, kappa_DAC0,
                           kappa_M1, kappa_M0,
                           kappa_Th1, kappa_ADC1,
                           P_bar, P_tilda,
                           P_th, P_adc):
    print(f"\n Power Components: {label} ")
    print(
        f"{'k':>2} | {'PS':>12} | {'PV':>12} | {'PK':>12} | "
        f"{'PMUI':>12} | {'PDAC':>12} | {'Pth':>12} | {'Padc':>12} | {'SINR':>12}"
    )
    print("-" * 118)

    total = {
        "PS": 0.0, "PV": 0.0, "PK": 0.0, "PMUI": 0.0,
        "PDAC": 0.0, "Pth": 0.0, "Padc": 0.0
    }

    for k in range(K):
        PS = P_tilda[k] * P_bar[k] * kappa_S[k]

        PV = P_tilda[k] * P_bar[k] * kappa_V1[k] + P_tilda[k] * kappa_V0[k]

        PK = P_tilda[k] * P_bar[k] * kappa_K1[k] + P_tilda[k] * kappa_K0[k]

        PMUI = sum(
            P_tilda[i] * (P_bar[k] * kappa_M1[k, i] + kappa_M0[k, i])
            for i in range(K) if i != k
        )

        PDAC = P_bar[k] * kappa_DAC1[k] + kappa_DAC0[k]

        Pth = P_th[k]
        Padc = P_adc[k]

        den = PV + PK + PMUI + PDAC + Pth + Padc
        SINR = PS / (den + 1e-30)

        print(
            f"{k:2d} | {PS:12.4e} | {PV:12.4e} | {PK:12.4e} | "
            f"{PMUI:12.4e} | {PDAC:12.4e} | {Pth:12.4e} | {Padc:12.4e} | {SINR:12.4e}"
        )

        total["PS"] += PS
        total["PV"] += PV
        total["PK"] += PK
        total["PMUI"] += PMUI
        total["PDAC"] += PDAC
        total["Pth"] += Pth
        total["Padc"] += Padc

    print("-" * 118)
    print(
        f"{'SUM':>2} | {total['PS']:12.4e} | {total['PV']:12.4e} | {total['PK']:12.4e} | "
        f"{total['PMUI']:12.4e} | {total['PDAC']:12.4e} | {total['Pth']:12.4e} | "
        f"{total['Padc']:12.4e} | {'':>12}"
    )

def clip_P_bar_given_P_tilda(P_bar_new, P_tilda, P_total_max, eps=1e-12):
    if P_total_max is None:
        return np.maximum(P_bar_new, eps)

    Pmax = (
        P_total_max * np.ones_like(P_bar_new)
        if np.isscalar(P_total_max)
        else np.asarray(P_total_max)
    )
    upper = np.maximum(Pmax - P_tilda - eps, eps)
    return np.clip(P_bar_new, eps, upper)


def clip_P_tilda_given_P_bar(P_tilda_new, P_bar, P_total_max, eps=1e-12):
    if P_total_max is None:
        return np.maximum(P_tilda_new, eps)

    Pmax = (
        P_total_max * np.ones_like(P_tilda_new)
        if np.isscalar(P_total_max)
        else np.asarray(P_total_max)
    )
    upper = np.maximum(Pmax - P_bar - eps, eps)
    return np.clip(P_tilda_new, eps, upper)

def compute_WSR_exact(K, w,
                      kappa_S,
                      kappa_V1, kappa_K1,
                      kappa_V0, kappa_K0,
                      kappa_DAC1, kappa_DAC0,
                      kappa_M1, kappa_M0,
                      kappa_Th1,kappa_ADC1,
                      P_bar, P_tilda,P_th, P_adc,d,tau):         
    """WSR = Σ_k w_k log2(1 + SINR_k)"""
    wsr = 0.0
    for k in range(K):
        sinr = compute_SINR_exact(
            k, K, kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1, kappa_ADC1,
            P_bar, P_tilda,P_th, P_adc
        )
        wsr += w[k]  * np.log2(1.0 + max(sinr, 0.0))
    return wsr * (d/tau)


# Auxiliary variable μ_k
def update_auxiliary(K, w, gamma,
                     kappa_S,
                     kappa_V1, kappa_K1,
                     kappa_V0, kappa_K0,
                     kappa_DAC1, kappa_DAC0,
                     kappa_M1, kappa_M0,
                     kappa_Th1,kappa_ADC1, 
                     P_bar, P_tilda,P_th, P_adc):
    mu = np.zeros(K)
    for k in range(K):
        Ak = compute_Ak(k, kappa_S, P_tilda)
        Bk = compute_Bk(k, K, kappa_V1, kappa_K1,
                        kappa_DAC1, kappa_M1, P_tilda)
        Ck = compute_Ck(k, K, kappa_V0, kappa_K0,
                        kappa_DAC0, kappa_M0, P_tilda)
        numer = w[k] * (1.0 + gamma[k]) * Ak * P_bar[k]
        denom = (
            (Ak + Bk) * P_bar[k]
            + Ck
            + P_th[k]
            + P_adc[k]
        )
        mu[k] = np.sqrt(max(numer, 0.0)) / (denom + 1e-30)
    return mu
  
def compute_P_bar(K, P_tilda, w,
                    kappa_S, kappa_V1, kappa_K1,
                    kappa_M1, kappa_Th1,
                    kappa_DAC1, kappa_ADC1,Z_list,
                    gamma, mu):
    """
    Computes P̄_k* using P̃ (P_tilda) as per closed-form expression
    """
    P_bar_new = np.zeros(K)
    
    for k in range(K):
        auxillary_k = mu[k]
        gamma_k = gamma[k]
    # Numerator (uses P_tilda)
        numerator = w[k] * (1 + gamma_k) * P_tilda[k] * kappa_S[k]

        # self term (uses P_tilda)
        self_term = P_tilda[k] * (kappa_S[k] + kappa_V1[k] + kappa_K1[k])

        # Interference term (uses P_tilda for all i ≠ k)
        summation = 0.0
        for i in range(K):
            if i != k:
                summation     +=  P_tilda[i] * kappa_M1[k,i]
        # Constant hardware/noise terms (NOT multiplied by P_tilda)
        constant_terms = (
            kappa_Th1[k] +
            kappa_DAC1[k] +
            kappa_ADC1[k]
        )

        denominator = (auxillary_k ** 2) * (self_term + summation + constant_terms) ** 2
        P_bar_new[k] = numerator / (denominator + 1e-30)
    return P_bar_new
    
def compute_P_tilda(K, P_bar, w,
                      kappa_S,
                      kappa_V1, kappa_K1,
                      kappa_V0, kappa_K0,
                      kappa_M1, kappa_M0,
                      gamma, mu):

    """
    Computes P_tilda[k] using P_bar and given parameters
    auxillary: array of size K (since auxillary_i appears in summation)
    """
    P_tilda_new = np.zeros(K)
    for k in range(K):

        auxillary_k = mu[k]
        gamma_k = gamma[k]

        # Numerator
        numerator = (
            (auxillary_k ** 2) *
            w[k] *
            (1 + gamma_k) *
            kappa_S[k] *
            P_bar[k]
        )

        # Self term
        self_term = (
            P_bar[k] * (kappa_S[k] + kappa_V1[k] + kappa_K1[k]) +
            kappa_V0[k] + kappa_K0[k]
        )

        # Interference
        summation = 0.0

        for i in range(K):
            if i != k:
                summation += (
                    (mu[i] ** 2) *
                    (P_bar[i] * kappa_M1[i, k] + kappa_M0[i, k])
                )

        # Denominator
        denominator = auxillary_k ** 2 * self_term + summation
        
        P_tilda_new[k] = numerator / (denominator**2 + 1e-30)
    return P_tilda_new

def _as_power_vector(P_total_max, K):
    if P_total_max is None:
        return None
    if np.isscalar(P_total_max):
        return float(P_total_max) * np.ones(K, dtype=float)
    return np.asarray(P_total_max, dtype=float).reshape(K)

def _project_to_simplex_nonnegative(v, total):
    """
    Project vector v onto {x >= 0, sum(x) <= total}.
    """
    v = np.asarray(v, dtype=float)

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


def project_joint_three_power_blocks(
    P_bar_raw,
    P_tilda_raw,
    P_prime_raw,
    P_total_max,
    eps=1e-12,
):
    """
    Jointly project (P_bar, P_tilda, P_prime) for each user onto

        P_bar[k] >= eps,
        P_tilda[k] >= eps,
        P_prime[k] >= eps,
        P_bar[k] + P_tilda[k] + P_prime[k] <= P_total_max[k].

    This avoids sequential clipping bias.
    """
    P_bar_raw = np.asarray(P_bar_raw, dtype=float).reshape(-1)
    P_tilda_raw = np.asarray(P_tilda_raw, dtype=float).reshape(-1)
    P_prime_raw = np.asarray(P_prime_raw, dtype=float).reshape(-1)

    K = len(P_bar_raw)
    Pmax = _as_power_vector(P_total_max, K)

    if Pmax is None:
        return (
            np.maximum(P_bar_raw, eps),
            np.maximum(P_tilda_raw, eps),
            np.maximum(P_prime_raw, eps),
        )

    P_bar_proj = np.zeros(K)
    P_tilda_proj = np.zeros(K)
    P_prime_proj = np.zeros(K)

    for k in range(K):
        v = np.array(
            [P_bar_raw[k], P_tilda_raw[k], P_prime_raw[k]],
            dtype=float,
        )

        # First enforce minimum eps.
        v_eps = np.maximum(v, eps)

        if np.sum(v_eps) <= Pmax[k]:
            p = v_eps
        else:
            # Project after removing eps floor.
            budget = max(Pmax[k] - 3.0 * eps, 0.0)
            p0 = _project_to_simplex_nonnegative(v - eps, budget)
            p = p0 + eps

        P_bar_proj[k] = p[0]
        P_tilda_proj[k] = p[1]
        P_prime_proj[k] = p[2]

    return P_bar_proj, P_tilda_proj, P_prime_proj
def reset_violating_users_to_epa(
    P_bar_raw,
    P_tilda_raw,
    P_prime_raw,
    P_bar_epa,
    P_tilda_epa,
    P_prime_epa,
    P_total_max,
    eps=1e-12,
    verbose=False,
):
    """
    Check per-user power constraint.

    If user k violates:
        P_bar[k] + P_tilda[k] + P_prime[k] <= P_total_max[k]

    then reset that user's three powers to EPA values:
        P_bar_epa[k], P_tilda_epa[k], P_prime_epa[k]

    Otherwise keep the updated values.
    """

    P_bar_new = np.asarray(P_bar_raw, dtype=float).reshape(-1).copy()
    P_tilda_new = np.asarray(P_tilda_raw, dtype=float).reshape(-1).copy()
    P_prime_new = np.asarray(P_prime_raw, dtype=float).reshape(-1).copy()

    P_bar_epa = np.asarray(P_bar_epa, dtype=float).reshape(-1)
    P_tilda_epa = np.asarray(P_tilda_epa, dtype=float).reshape(-1)
    P_prime_epa = np.asarray(P_prime_epa, dtype=float).reshape(-1)

    K = len(P_bar_new)
    Pmax = _as_power_vector(P_total_max, K)

    if Pmax is None:
        return (
            np.maximum(P_bar_new, eps),
            np.maximum(P_tilda_new, eps),
            np.maximum(P_prime_new, eps),
            np.zeros(K, dtype=bool),
        )

    # Avoid negative/zero powers first
    P_bar_new = np.maximum(P_bar_new, eps)
    P_tilda_new = np.maximum(P_tilda_new, eps)
    P_prime_new = np.maximum(P_prime_new, eps)

    total_power = P_bar_new + P_tilda_new + P_prime_new

    violation_mask = total_power > (Pmax + 1e-10)

    if np.any(violation_mask):
        P_bar_new[violation_mask] = P_bar_epa[violation_mask]
        P_tilda_new[violation_mask] = P_tilda_epa[violation_mask]
        P_prime_new[violation_mask] = P_prime_epa[violation_mask]

    # Safety check: EPA itself must be feasible
    total_after = P_bar_new + P_tilda_new + P_prime_new
    if np.any(total_after > Pmax + 1e-10):
        raise RuntimeError(
            "EPA fallback is also violating the power constraint. "
            "Check init_delta/init_lambda or EPA definition."
        )

    if verbose and np.any(violation_mask):
        print(
            f"[EPA RESET] {np.sum(violation_mask)} users violated power constraint "
            f"and were reset to EPA values."
        )

    return P_bar_new, P_tilda_new, P_prime_new, violation_mask

def clip_P_bar_joint(P_bar_new, P_tilda, P_prime, P_total_max, eps=1e-12):
    P_bar_new = np.asarray(P_bar_new, dtype=float)
    if P_total_max is None:
        return np.maximum(P_bar_new, eps)
    Pmax = _as_power_vector(P_total_max, len(P_bar_new))
    upper = np.maximum(Pmax - np.asarray(P_tilda) - np.asarray(P_prime) - eps, eps)
    return np.clip(P_bar_new, eps, upper)


def clip_P_tilda_joint(P_tilda_new, P_bar, P_prime, P_total_max, eps=1e-12):
    P_tilda_new = np.asarray(P_tilda_new, dtype=float)
    if P_total_max is None:
        return np.maximum(P_tilda_new, eps)
    Pmax = _as_power_vector(P_total_max, len(P_tilda_new))
    upper = np.maximum(Pmax - np.asarray(P_bar) - np.asarray(P_prime) - eps, eps)
    return np.clip(P_tilda_new, eps, upper)


def clip_P_prime_joint(P_prime_new, P_bar, P_tilda, P_total_max, eps=1e-12):
    P_prime_new = np.asarray(P_prime_new, dtype=float)
    if P_total_max is None:
        return np.maximum(P_prime_new, eps)
    Pmax = _as_power_vector(P_total_max, len(P_prime_new))
    upper = np.maximum(Pmax - np.asarray(P_bar) - np.asarray(P_tilda) - eps, eps)
    return np.clip(P_prime_new, eps, upper)


# ============
# Sensing coefficient loading/building
# a_sens[q,k], b_sens[q,k], n_sens[q]
# ============

def _normalize_sensing_matrix_to_qk(X, K, name):
    """Accepts K x Tg or Tg x K and returns Tg x K."""
    X = np.asarray(X, dtype=float)
    X = np.squeeze(X)

    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D after squeeze; got shape {X.shape}")

    if X.shape[0] == K:      # K x Tg from MATLAB
        return X.T.copy()
    if X.shape[1] == K:      # Tg x K already
        return X.copy()

    raise ValueError(f"Cannot interpret {name} shape {X.shape} as K x Tg or Tg x K with K={K}")


def load_sensing_coefficients_from_mat(mat_file, setup_idx=0, K_use=None):
    """
    Prefer MATLAB's already computed sensing kappas, if present.

    Returns
    -------
    a_sens : shape Tg x K
        Desired sensing coefficients a_{qk}.
    b_sens : shape Tg x K
        Interference+clutter coefficients b_{qk}.
    n_sens : shape Tg
        Fixed sensing floor n_q.
    """
    mat = loadmat(mat_file, squeeze_me=False, struct_as_record=False)

    if K_use is None:
        if "UE" in mat:
            K_use = int(np.squeeze(mat["UE"]))
        elif "K" in mat:
            K_use = int(np.squeeze(mat["K"]))
        else:
            raise ValueError("K_use must be provided when UE/K is not found in mat file.")

    # Best case: Sensing_kappa_joint.Kappa_S, .Beta_dash, .Kappa_const
    if "Sensing_kappa_joint" in mat:
        s = mat["Sensing_kappa_joint"]

        s_flat = np.ravel(s, order="F")
        if setup_idx >= len(s_flat):
            raise IndexError(
                f"setup_idx={setup_idx} but Sensing_kappa_joint has only {len(s_flat)} entries"
            )

        s0 = s_flat[setup_idx]
        if hasattr(s0, "Kappa_S"):
            A = np.asarray(s0.Kappa_S)
            B = np.asarray(s0.Beta_dash)
            C = np.asarray(s0.Kappa_const)
        else:
            A = np.asarray(s0["Kappa_S"])
            B = np.asarray(s0["Beta_dash"])
            C = np.asarray(s0["Kappa_const"])

        a_sens = _normalize_sensing_matrix_to_qk(A, K_use, "Kappa_S")
        b_sens = _normalize_sensing_matrix_to_qk(B, K_use, "Beta_dash")
        n_sens = np.asarray(C, dtype=float).squeeze()
        return a_sens, b_sens, n_sens

    # Fallback names from your uploaded .mat file.
    if all(name in mat for name in ["A_coeff", "B_coeff", "B_const"]):
        a_sens = _normalize_sensing_matrix_to_qk(mat["A_coeff"], K_use, "A_coeff")
        b_sens = _normalize_sensing_matrix_to_qk(mat["B_coeff"], K_use, "B_coeff")
        n_sens = np.asarray(mat["B_const"], dtype=float).squeeze()
        return a_sens, b_sens, n_sens

    raise KeyError(
        "No sensing coefficients found. Expected Sensing_kappa_joint or "
        "A_coeff/B_coeff/B_const. Use build_sensing_coefficients_from_geometry(...) instead."
    )


def _col(x):
    return np.asarray(x, dtype=complex).reshape(-1, 1)


def _get_b_lq(b_target_all, l, q):
    return _col(b_target_all[:, l, q])


def _build_v_list_for_q(b_target_all, q, V_combiner_all=None, v_iter=None):
    M = b_target_all.shape[1]
    if V_combiner_all is None:
        return [_get_b_lq(b_target_all, l, q) for l in range(M)]
    return [_col(V_combiner_all[:, l, q, v_iter]) for l in range(M)]

def compute_Ps_coeff_sensing_proj(q, k, T,
                                  alpha_list,
                                  A_list,
                                  b_target_all,
                                  alpha_var_matrix_all,
                                  v_list):
    """
    MATLAB match:
        Kappa_S_acc(i,q) += alpha_i^2 * sigma_i,l,q^2 *
                            |v_lq^H A_l b_lq|^2

    Returns coefficient a_{q,k}, not multiplied by P'_k.
    """
    total = 0.0

    for l, A_l in enumerate(A_list):
        v_lq = v_list[l]
        b_lq = _get_b_lq(b_target_all, l, q)

        gain = np.abs((v_lq.conj().T @ A_l @ b_lq)[0, 0]) ** 2
        total += alpha_var_matrix_all[k, l, q] * gain

    return float(np.real((T ** 2) * (alpha_list[k] ** 2) * total))


def compute_Pinter_coeff_sensing_proj(q, k, T,
                                      alpha_list,
                                      A_list,
                                      b_target_all,
                                      alpha_var_matrix_all,
                                      v_list):
    """
    MATLAB match:
        Beta_dash_acc(i,q) += alpha_i^2 *
            sum_{t != q} sigma_i,l,t^2 |v_lq^H A_l b_lt|^2

    Returns inter-target coefficient for b_{q,k}, not multiplied by P'_k.
    """
    Tg = b_target_all.shape[2]
    total = 0.0

    for l, A_l in enumerate(A_list):
        v_lq = v_list[l]

        for t in range(Tg):
            if t == q:
                continue

            b_lt = _get_b_lq(b_target_all, l, t)
            gain = np.abs((v_lq.conj().T @ A_l @ b_lt)[0, 0]) ** 2
            total += alpha_var_matrix_all[k, l, t] * gain

    return float(np.real((T ** 2) * (alpha_list[k] ** 2) * total))


def compute_Pclutter_coeff_sensing_proj(q, T,
                                        alpha_list,
                                        A_list,
                                        v_list,
                                        H_direct_iter,
                                        iter_idx):
    """
    MATLAB match:
        Z_q = sum_l v_lq^H sum_i alpha_i A_l h_dir_li
        clutter_coeff_q = Tau^2 * |Z_q|^2

    Returns target-wise clutter coefficient for one MC iteration.
    """
    K = len(alpha_list)
    z_q = 0.0 + 0.0j

    for l, A_l in enumerate(A_list):
        v_lq = v_list[l]

        clutter_vector_sum_i = np.zeros((A_l.shape[0], 1), dtype=complex)

        for i in range(K):
            h_dir_li = _col(H_direct_iter[:, i, l, iter_idx])
            clutter_vector_sum_i += alpha_list[i] * (A_l @ h_dir_li)

        z_q += (v_lq.conj().T @ clutter_vector_sum_i)[0, 0]

    return float(np.real((T ** 2) * (np.abs(z_q) ** 2)))


def compute_PDAC_const_sensing_proj(q, T,
                                    sigma_d2,
                                    A_list,
                                    v_list,
                                    H_true_iter,
                                    iter_idx):
    """
    MATLAB match:
        W_i = sum_l v_lq^H A_l h_li
        P_DAC_q = Tau * sum_i sigma_DAC_i^2 |W_i|^2

    Returns fixed DAC floor for one target and one MC iteration.
    """
    K = len(sigma_d2)
    W_i = np.zeros(K, dtype=complex)

    for i in range(K):
        z_i = 0.0 + 0.0j

        for l, A_l in enumerate(A_list):
            v_lq = v_list[l]
            h_li = _col(H_true_iter[:, i, l, iter_idx])
            z_i += (v_lq.conj().T @ A_l @ h_li)[0, 0]

        W_i[i] = z_i

    return float(np.real(T * np.sum(sigma_d2 * (np.abs(W_i) ** 2))))


def compute_PTh_const_sensing_proj(q, T, sigma_w2, A_list, v_list):
    """
    MATLAB match:
        P_Th_q = Tau * Noise_var * sum_l v_lq^H A_l A_l^H v_lq
    """
    total = 0.0

    for l, A_l in enumerate(A_list):
        v_lq = v_list[l]
        val = v_lq.conj().T @ A_l @ A_l.conj().T @ v_lq
        total += np.real(val[0, 0])

    return float(np.real(T * sigma_w2 * total))


def compute_PADC_const_sensing_proj(q, T, Theta_list, v_list):
    """
    MATLAB match:
        P_ADC_q = Tau * sum_l v_lq^H Theta_l v_lq
    """
    total = 0.0

    for l, Theta_l in enumerate(Theta_list):
        v_lq = v_list[l]
        val = v_lq.conj().T @ Theta_l @ v_lq
        total += np.real(val[0, 0])

    return float(np.real(T * total))

def build_sensing_coefficients_matlab_matching(
    T,
    alpha_list,
    sigma_d2,
    A_list,
    Theta_list,
    b_target_all,
    alpha_var_matrix_all,
    V_combiner_all,
    H_direct_iter,
    H_true_iter,
    sigma_w2,
    return_parts=False,
):
    alpha_list = np.asarray(alpha_list, dtype=float).reshape(-1)
    sigma_d2 = np.asarray(sigma_d2, dtype=float).reshape(-1)
    alpha_var_matrix_all = np.asarray(alpha_var_matrix_all, dtype=float)

    if V_combiner_all is None:
        raise ValueError(
            "V_combiner_all is required for MATLAB-matching sensing coefficients. "
            "Without it, Python will fall back to MR sensing and will not match MATLAB."
        )

    if H_direct_iter is None:
        raise ValueError("H_direct_iter is required for MATLAB-matching clutter coefficient.")

    if H_true_iter is None:
        raise ValueError("H_true_iter is required for MATLAB-matching DAC coefficient.")

    K = len(alpha_list)
    Tg = b_target_all.shape[2]
    num_iter = V_combiner_all.shape[3]

    if H_direct_iter.shape[3] != num_iter:
        raise ValueError(
            f"H_direct_iter has {H_direct_iter.shape[3]} iterations, "
            f"but V_combiner_all has {num_iter}."
        )

    if H_true_iter.shape[3] != num_iter:
        raise ValueError(
            f"H_true_iter has {H_true_iter.shape[3]} iterations, "
            f"but V_combiner_all has {num_iter}."
        )

    a_sens_acc = np.zeros((Tg, K), dtype=float)
    inter_acc = np.zeros((Tg, K), dtype=float)
    clutter_acc = np.zeros(Tg, dtype=float)
    n_sens_acc = np.zeros(Tg, dtype=float)

    for iter_idx in range(num_iter):
        for q in range(Tg):
            v_list = build_sensing_v_list_from_V(
                V_combiner_all=V_combiner_all,
                q=q,
                iter_idx=iter_idx,
            )

            for k in range(K):
                a_sens_acc[q, k] += compute_Ps_coeff_sensing_proj(
                    q=q,
                    k=k,
                    T=T,
                    alpha_list=alpha_list,
                    A_list=A_list,
                    b_target_all=b_target_all,
                    alpha_var_matrix_all=alpha_var_matrix_all,
                    v_list=v_list,
                )

                inter_acc[q, k] += compute_Pinter_coeff_sensing_proj(
                    q=q,
                    k=k,
                    T=T,
                    alpha_list=alpha_list,
                    A_list=A_list,
                    b_target_all=b_target_all,
                    alpha_var_matrix_all=alpha_var_matrix_all,
                    v_list=v_list,
                )

            clutter_acc[q] += compute_Pclutter_coeff_sensing_proj(
                q=q,
                T=T,
                alpha_list=alpha_list,
                A_list=A_list,
                v_list=v_list,
                H_direct_iter=H_direct_iter,
                iter_idx=iter_idx,
            )

            pdac_q = compute_PDAC_const_sensing_proj(
                q=q,
                T=T,
                sigma_d2=sigma_d2,
                A_list=A_list,
                v_list=v_list,
                H_true_iter=H_true_iter,
                iter_idx=iter_idx,
            )

            pth_q = compute_PTh_const_sensing_proj(
                q=q,
                T=T,
                sigma_w2=sigma_w2,
                A_list=A_list,
                v_list=v_list,
            )

            padc_q = compute_PADC_const_sensing_proj(
                q=q,
                T=T,
                Theta_list=Theta_list,
                v_list=v_list,
            )

            n_sens_acc[q] += pdac_q + pth_q + padc_q

    a_sens = a_sens_acc / num_iter
    inter_coeff = inter_acc / num_iter
    clutter_coeff = clutter_acc / num_iter
    n_sens = n_sens_acc / num_iter

    # MATLAB does:
    #   Beta_dash_mat = P_inter_coeff + repmat(clutter_coeff_uniform, UE, 1)
    # Since Python uses Tg x K, repeat clutter over the K dimension.
    b_sens = inter_coeff + clutter_coeff[:, None]

    a_sens = np.real(a_sens)
    b_sens = np.real(b_sens)
    n_sens = np.real(n_sens)

    if return_parts:
        return a_sens, b_sens, n_sens, {
            "inter_coeff": inter_coeff,
            "clutter_coeff": clutter_coeff,
            "n_sens": n_sens,
        }

    return a_sens, b_sens, n_sens

def compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens):
    P_prime = np.asarray(P_prime, dtype=float).reshape(-1)
    S = np.maximum(np.asarray(a_sens, dtype=float) @ P_prime, 0.0)
    D = np.maximum(np.asarray(b_sens, dtype=float) @ P_prime + np.asarray(n_sens, dtype=float), 1e-30)
    return S, D


def compute_sensing_sinr(P_prime, a_sens, b_sens, n_sens):
    S, D = compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens)
    return S / (D + 1e-30)


def compute_sensing_wsr(P_prime, a_sens, b_sens, n_sens, target_weights=None, w_s=1.0):
    sinr = compute_sensing_sinr(P_prime, a_sens, b_sens, n_sens)
    Tg = len(sinr)
    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)
    rho = float(w_s) * np.asarray(target_weights, dtype=float).reshape(Tg)
    return float(np.sum(rho * np.log2(1.0 + np.maximum(sinr, 0.0))))


def update_sensing_mu(P_prime, a_sens, b_sens, n_sens):
    """LDT update: mu_q = SINR_sens,q."""
    return compute_sensing_sinr(P_prime, a_sens, b_sens, n_sens)


def update_sensing_y(P_prime, a_sens, b_sens, n_sens, mu_s, target_weights=None, w_s=1.0):
    """QT update: y_q = sqrt(rho_q(1+mu_q)S_q)/(S_q+D_q)."""
    S, D = compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens)
    Tg = len(S)
    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)
    rho = float(w_s) * np.asarray(target_weights, dtype=float).reshape(Tg)
    numerator = np.sqrt(np.maximum(rho * (1.0 + np.asarray(mu_s)) * S, 0.0))
    return numerator / (S + D + 1e-30)


def _sensing_coordinate_derivative(p_k, k, P_prime, a_sens, b_sens, mu_s, y_s, rho, eps=1e-30):
    P_tmp = np.asarray(P_prime, dtype=float).copy()
    P_tmp[k] = p_k
    S = np.maximum(np.asarray(a_sens) @ P_tmp, eps)

    term1 = (
        np.asarray(y_s)
        * np.sqrt(np.maximum(rho * (1.0 + np.asarray(mu_s)), 0.0))
        * np.asarray(a_sens)[:, k]
        / np.sqrt(S)
    )
    term2 = (np.asarray(y_s) ** 2) * (np.asarray(a_sens)[:, k] + np.asarray(b_sens)[:, k])
    return float(np.sum(term1 - term2))


def update_P_prime(
    P_prime,
    P_bar,
    P_tilda,
    a_sens,
    b_sens,
    n_sens,
    mu_s,
    y_s,
    target_weights=None,
    w_s=1.0,
    P_total_max=None,
    P_search_max=None,
    eps_power=1e-12,
    bisect_iters=60,
):
    """
    Coordinate-wise update of sensing power P'_k.

    For Tg>1, the derivative is a sum of multiple 1/sqrt(S_q) terms, so there
    is generally no single algebraic closed form. This routine solves the exact
    1-D stationarity condition for each coordinate by bisection; the derivative
    is monotone decreasing in P'_k.
    """
    P_new = np.asarray(P_prime, dtype=float).copy()
    K = len(P_new)
    Tg = np.asarray(a_sens).shape[0]

    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)
    rho = float(w_s) * np.asarray(target_weights, dtype=float).reshape(Tg)

    Pmax = _as_power_vector(P_total_max, K)
    Psearch = _as_power_vector(P_search_max, K)

    for k in range(K):
        if Pmax is None:
            # Raw sensing update. Do not enforce the joint budget here.
            # But use P_search_max as a finite bisection search cap to avoid huge raw values.
            if Psearch is not None:
                upper = max(float(Psearch[k]), eps_power)
            else:
                upper = max(float(P_new[k]), 1.0)
                while _sensing_coordinate_derivative(upper, k, P_new, a_sens, b_sens, mu_s, y_s, rho) > 0:
                    upper *= 2.0
                    if upper > 1e12:
                        break
        else:
            upper = Pmax[k] - P_bar[k] - P_tilda[k] - eps_power
            upper = max(float(upper), eps_power)

        lo = eps_power
        hi = upper

        d_lo = _sensing_coordinate_derivative(lo, k, P_new, a_sens, b_sens, mu_s, y_s, rho)
        d_hi = _sensing_coordinate_derivative(hi, k, P_new, a_sens, b_sens, mu_s, y_s, rho)

        if d_lo <= 0.0:
            P_new[k] = lo
            continue
        if d_hi >= 0.0:
            P_new[k] = hi
            continue

        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            d_mid = _sensing_coordinate_derivative(mid, k, P_new, a_sens, b_sens, mu_s, y_s, rho)
            if d_mid > 0.0:
                lo = mid
            else:
                hi = mid

        P_new[k] = 0.5 * (lo + hi)

    return np.maximum(P_new, eps_power)
    # return clip_P_prime_joint(P_new, P_bar, P_tilda, P_total_max, eps_power)

def update_P_prime_block_exact(
    P_prime,
    P_bar,
    P_tilda,
    a_sens,
    b_sens,
    n_sens,
    mu_s,
    y_s,
    target_weights,
    w_s,
    P_total_max,
    eps_power=1e-12,
    inner_iters=50,
    inner_tol=1e-8,
):
    P_new = np.asarray(P_prime, dtype=float).copy()

    for _ in range(inner_iters):
        P_old = P_new.copy()

        P_new = update_P_prime(
            P_prime=P_new,
            P_bar=P_bar,
            P_tilda=P_tilda,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            mu_s=mu_s,
            y_s=y_s,
            target_weights=target_weights,
            w_s=w_s,
            P_total_max=P_total_max,
            P_search_max=None,
            eps_power=eps_power,
        )

        if np.max(np.abs(P_new - P_old)) < inner_tol:
            break

    return P_new

def compute_joint_objective(
    comm_wsr,
    P_prime,
    a_sens,
    b_sens,
    n_sens,
    target_weights=None,
    w_c=1.0,
    w_s=1.0,
):
    sensing_wsr = compute_sensing_wsr(
        P_prime=P_prime,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
    )
    return float(w_c * comm_wsr + sensing_wsr), float(sensing_wsr)

def evaluate_joint_state(
    K, w,
    kappa_S,
    kappa_V1, kappa_K1,
    kappa_V0, kappa_K0,
    kappa_DAC1, kappa_DAC0,
    kappa_M1, kappa_M0,
    kappa_Th1, kappa_ADC1,
    kappa_Th0, kappa_ADC0,
    P_bar, P_tilda, P_prime,
    a_sens, b_sens, n_sens,
    target_weights,
    w_c, w_s,
    d, tau,
):
    """
    Evaluate communication WSR, sensing WSR, joint objective, gamma,
    P_th and P_adc for a given feasible power state.
    """

    P_bar = np.asarray(P_bar, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda, dtype=float).reshape(K)
    P_prime = np.asarray(P_prime, dtype=float).reshape(K)

    P_th = P_bar * kappa_Th1 + kappa_Th0
    P_adc = P_bar * kappa_ADC1 + kappa_ADC0

    gamma = np.array([
        compute_SINR_exact(
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
    ])

    comm_wsr = compute_WSR_exact(
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
    )

    joint_obj, sensing_wsr = compute_joint_objective(
        comm_wsr=comm_wsr,
        P_prime=P_prime,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_c=w_c,
        w_s=w_s,
    )

    return {
        "joint_obj": float(joint_obj),
        "comm_wsr": float(comm_wsr),
        "sensing_wsr": float(sensing_wsr),
        "gamma": gamma,
        "P_th": P_th,
        "P_adc": P_adc,
    }
def accept_three_power_update_with_backtracking(
    K, w,
    kappa_S,
    kappa_V1, kappa_K1,
    kappa_V0, kappa_K0,
    kappa_DAC1, kappa_DAC0,
    kappa_M1, kappa_M0,
    kappa_Th1, kappa_ADC1,
    kappa_Th0, kappa_ADC0,
    P_bar_old, P_tilda_old, P_prime_old,
    P_bar_target, P_tilda_target, P_prime_target,
    old_state,
    a_sens, b_sens, n_sens,
    target_weights,
    w_c, w_s,
    d, tau,
    min_eta=1e-4,
    accept_tol=1e-12,
):
    """
    Accept projected three-power update only if the joint objective does not decrease.
    Otherwise backtrack from old powers toward projected target.
    """

    eta = 1.0
    old_joint = float(old_state["joint_obj"])

    while eta >= min_eta:
        P_bar_candidate = P_bar_old + eta * (P_bar_target - P_bar_old)
        P_tilda_candidate = P_tilda_old + eta * (P_tilda_target - P_tilda_old)
        P_prime_candidate = P_prime_old + eta * (P_prime_target - P_prime_old)

        candidate_state = evaluate_joint_state(
            K=K,
            w=w,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1,
            kappa_DAC0=kappa_DAC0,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            kappa_Th1=kappa_Th1,
            kappa_ADC1=kappa_ADC1,
            kappa_Th0=kappa_Th0,
            kappa_ADC0=kappa_ADC0,
            P_bar=P_bar_candidate,
            P_tilda=P_tilda_candidate,
            P_prime=P_prime_candidate,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_c=w_c,
            w_s=w_s,
            d=d,
            tau=tau,
        )

        if candidate_state["joint_obj"] >= old_joint - accept_tol:
            return (
                P_bar_candidate,
                P_tilda_candidate,
                P_prime_candidate,
                candidate_state,
                eta,
                True,
            )

        eta *= 0.5

    # If no acceptable step is found, reject update and stay at old point.
    return (
        P_bar_old,
        P_tilda_old,
        P_prime_old,
        old_state,
        0.0,
        False,
    )

def compute_sensing_components_from_coeffs(
    P_prime,
    a_sens,
    b_sens,
    n_sens,
    target_weights=None,
    w_s=1.0,
    sensing_coeff_parts=None,
):
    P_prime = np.asarray(P_prime, dtype=float).reshape(-1)
    a_sens = np.asarray(a_sens, dtype=float)
    b_sens = np.asarray(b_sens, dtype=float)
    n_sens = np.asarray(n_sens, dtype=float).reshape(-1)

    S, D = compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens)

    # Default old combined denominator contribution
    combined_I_clutter = np.maximum(b_sens @ P_prime, 0.0)

    # New split: inter-target and clutter separately
    if sensing_coeff_parts is not None:
        inter_coeff = np.asarray(
            sensing_coeff_parts.get("inter_coeff", np.zeros_like(b_sens)),
            dtype=float,
        )

        clutter_coeff = np.asarray(
            sensing_coeff_parts.get("clutter_coeff", np.zeros(a_sens.shape[0])),
            dtype=float,
        ).reshape(-1)

        Pinter = np.maximum(inter_coeff @ P_prime, 0.0)

        # Because b_sens = inter_coeff + clutter_coeff[:, None]
        # so clutter contribution = clutter_coeff[q] * sum_k P_prime[k]
        Pclutter = np.maximum(clutter_coeff * np.sum(P_prime), 0.0)
    else:
        Pinter = combined_I_clutter
        Pclutter = np.zeros_like(Pinter)

    noise_floor = n_sens.copy()

    # For checking
    I_plus_clutter = Pinter + Pclutter
    D_split = I_plus_clutter + noise_floor

    sinr = S / (D + 1e-30)
    rate = np.log2(1.0 + np.maximum(sinr, 0.0))

    Tg = len(sinr)
    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)

    omega = np.asarray(target_weights, dtype=float).reshape(Tg)
    weighted_no_ws = omega * rate
    weighted_with_ws = float(w_s) * weighted_no_ws

    return {
        "S": S,
        "Pinter": Pinter,
        "Pclutter": Pclutter,
        "I": I_plus_clutter,
        "N": noise_floor,
        "D": D,
        "D_split": D_split,
        "SINR": sinr,
        "SINR_dB": 10.0 * np.log10(np.maximum(sinr, 1e-30)),
        "rate": rate,
        "omega": omega,
        "weighted_rate_no_ws": weighted_no_ws,
        "weighted_rate_with_ws": weighted_with_ws,
        "sum_se": float(np.sum(rate)),
        "weighted_wsr_no_ws": float(np.sum(weighted_no_ws)),
        "wsr": float(np.sum(weighted_with_ws)),
    }

# ============
# Replacement optimizer: joint BCO loop
# Requires your existing communication functions:
#   build_all_kappas, build_kappa_Th0_ADC0, compute_SINR_exact,
#   update_auxiliary, compute_P_bar, compute_P_tilda, compute_WSR_exact
# ============

def run_joint_wsr_optimisation(
    K, w, alpha_list,
    tau, d, sigma_w2, sigma_d2,
    U_list, V_list, Sigma_list,
    R_list, h_bar_list,
    C_h_ref_all, C_h_total_all, D_Th_list, D_ADC_list,
    A_list, Theta_list, Z_list, G_list,
    W_list, T,
    U_q_list, V_q_list, g_lik,
    a_sens, b_sens, n_sens,
    target_weights=None,
    w_c=1.0,
    w_s=1.0,
    P_bar_init=None, P_tilda_init=None, P_prime_init=None,
    max_iters=200, epsilon=1e-3,
    P_total_max=None,
    eps_power=1e-12,
    verbose=False,
):
    P_bar = np.ones(K) if P_bar_init is None else np.asarray(P_bar_init, float).reshape(K)
    P_tilda = np.ones(K) if P_tilda_init is None else np.asarray(P_tilda_init, float).reshape(K)
    P_prime = np.ones(K) if P_prime_init is None else np.asarray(P_prime_init, float).reshape(K)
    w = np.asarray(w, dtype=float).reshape(K)
    # w_comm_eff = float(w_c) * w
    w_comm_eff = float(w_c) * (d / tau) * w
    if target_weights is None:
        target_weights = np.ones(np.asarray(a_sens).shape[0], dtype=float)

    # Initial feasibility projection.
    # P_bar = clip_P_bar_joint(P_bar, P_tilda, P_prime, P_total_max, eps_power)
    # P_tilda = clip_P_tilda_joint(P_tilda, P_bar, P_prime, P_total_max, eps_power)
    # P_prime = clip_P_prime_joint(P_prime, P_bar, P_tilda, P_total_max, eps_power)
    # Initial joint feasibility projection.
    
    # Store EPA fallback values.
    # These are the initial feasible powers.
    P_bar_epa = P_bar.copy()
    P_tilda_epa = P_tilda.copy()
    P_prime_epa = P_prime.copy()
    if verbose:
        print("Building communication kappa coefficients …")

    (kappa_S,
     kappa_V1, kappa_V0,
     kappa_K1, kappa_K0,
     kappa_DAC1, kappa_DAC0, kappa_Th1, kappa_ADC1,
     kappa_M1, kappa_M0) = build_all_kappas(
        K, alpha_list, tau, d,
        U_list, V_list, U_q_list, V_q_list, Sigma_list,
        R_list, h_bar_list,
        C_h_ref_all, C_h_total_all,
        sigma_d2, sigma_w2, D_Th_list, D_ADC_list,
        A_list, Theta_list, Z_list,
        g_lik,
    )

    kappa_Th0, kappa_ADC0 = build_kappa_Th0_ADC0(
        K=K,
        tau=tau,
        d=d,
        sigma_w2=sigma_w2,
        sigma_d2=sigma_d2,
        G_list=G_list,
        W_list=W_list,
        A_list=A_list,
        Theta_list=Theta_list,
        C_h_total_all=C_h_total_all,
    )

    joint_history = []
    comm_history = []
    sensing_history = []
    wsr_prev = -np.inf

    start_time = time.perf_counter()
    converged = False
    convergence_iter = max_iters

    for t in range(max_iters):

        P_bar_old = P_bar.copy()
        P_tilda_old = P_tilda.copy()
        P_prime_old = P_prime.copy()

        old_state = evaluate_joint_state(
            K=K,
            w=w,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1,
            kappa_DAC0=kappa_DAC0,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            kappa_Th1=kappa_Th1,
            kappa_ADC1=kappa_ADC1,
            kappa_Th0=kappa_Th0,
            kappa_ADC0=kappa_ADC0,
            P_bar=P_bar_old,
            P_tilda=P_tilda_old,
            P_prime=P_prime_old,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_c=w_c,
            w_s=w_s,
            d=d,
            tau=tau,
        )

        gamma = old_state["gamma"]
        P_th = old_state["P_th"]
        P_adc = old_state["P_adc"]

        mu = update_auxiliary(
            K, w_comm_eff, gamma,
            kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1, kappa_ADC1,
            P_bar_old,
            P_tilda_old,
            P_th,
            P_adc,
        )

        mu_s = update_sensing_mu(
            P_prime_old,
            a_sens,
            b_sens,
            n_sens,
        )

        y_s = update_sensing_y(
            P_prime_old,
            a_sens,
            b_sens,
            n_sens,
            mu_s=mu_s,
            target_weights=target_weights,
            w_s=w_s,
        )
        #   order:
        #   1) P_bar
        #   2) P_tilda
        #   3) P_prime
        Pmax_vec = _as_power_vector(P_total_max, K)

        # === exact constrained P_bar block ===
        P_bar_raw = compute_P_bar(
            K, P_tilda_old, w_comm_eff,
            kappa_S, kappa_V1, kappa_K1,
            kappa_M1, kappa_Th1,
            kappa_DAC1, kappa_ADC1, Z_list,
            gamma, mu,
        )

        P_bar_upper = Pmax_vec - P_tilda_old - P_prime_old - eps_power
        P_bar_upper = np.maximum(P_bar_upper, eps_power)
        P_bar_new = np.clip(P_bar_raw, eps_power, P_bar_upper)

        # === exact constrained P_tilda block, using updated P_bar ===
        P_tilda_raw = compute_P_tilda(
            K, P_bar_new, w_comm_eff,
            kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_M1, kappa_M0,
            gamma, mu,
        )

        P_tilda_upper = Pmax_vec - P_bar_new - P_prime_old - eps_power
        P_tilda_upper = np.maximum(P_tilda_upper, eps_power)
        P_tilda_new = np.clip(P_tilda_raw, eps_power, P_tilda_upper)

        # === exact constrained P_prime block, using updated P_bar and P_tilda ===
        P_prime_new = update_P_prime_block_exact(
            P_prime=P_prime_old,
            P_bar=P_bar_new,
            P_tilda=P_tilda_new,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            mu_s=mu_s,
            y_s=y_s,
            target_weights=target_weights,
            w_s=w_s,
            P_total_max=P_total_max,
            eps_power=eps_power,
            inner_iters=50,
            inner_tol=1e-8,
        )

        P_bar_target = P_bar_new
        P_tilda_target = P_tilda_new
        P_prime_target = P_prime_new

        # ============================================================
        # New order:
        #   1) P_tilda
        #   2) P_bar
        #   3) P_prime
        # ============================================================

        # Pmax_vec = _as_power_vector(P_total_max, K)

        # # === exact constrained P_tilda block FIRST ===
        # # P_tilda is updated using old P_bar and old P_prime fixed.
        # P_tilda_raw = compute_P_tilda(
        #     K,
        #     P_bar_old,          # old P_bar, because P_bar has not been updated yet
        #     w_comm_eff,
        #     kappa_S,
        #     kappa_V1,
        #     kappa_K1,
        #     kappa_V0,
        #     kappa_K0,
        #     kappa_M1,
        #     kappa_M0,
        #     gamma,
        #     mu,
        # )

        # P_tilda_upper = Pmax_vec - P_bar_old - P_prime_old - eps_power
        # P_tilda_upper = np.maximum(P_tilda_upper, eps_power)

        # P_tilda_new = np.clip(
        #     P_tilda_raw,
        #     eps_power,
        #     P_tilda_upper,
        # )


        # # === exact constrained P_bar block SECOND ===
        # # P_bar is now updated using the NEW P_tilda and old P_prime fixed.
        # P_bar_raw = compute_P_bar(
        #     K,
        #     P_tilda_new,        # important: use updated P_tilda
        #     w_comm_eff,
        #     kappa_S,
        #     kappa_V1,
        #     kappa_K1,
        #     kappa_M1,
        #     kappa_Th1,
        #     kappa_DAC1,
        #     kappa_ADC1,
        #     Z_list,
        #     gamma,
        #     mu,
        # )

        # P_bar_upper = Pmax_vec - P_tilda_new - P_prime_old - eps_power
        # P_bar_upper = np.maximum(P_bar_upper, eps_power)

        # P_bar_new = np.clip(
        #     P_bar_raw,
        #     eps_power,
        #     P_bar_upper,
        # )


        # # === exact constrained P_prime block THIRD ===
        # # P_prime is updated using the NEW P_tilda and NEW P_bar.
        # P_prime_new = update_P_prime_block_exact(
        #     P_prime=P_prime_old,
        #     P_bar=P_bar_new,
        #     P_tilda=P_tilda_new,
        #     a_sens=a_sens,
        #     b_sens=b_sens,
        #     n_sens=n_sens,
        #     mu_s=mu_s,
        #     y_s=y_s,
        #     target_weights=target_weights,
        #     w_s=w_s,
        #     P_total_max=P_total_max,
        #     eps_power=eps_power,
        #     inner_iters=50,
        #     inner_tol=1e-8,
        # )

        # P_bar_target = P_bar_new
        # P_tilda_target = P_tilda_new
        # P_prime_target = P_prime_new

        # ============================================================
        # New order:
        #   1) P_tilda
        #   2) P_prime
        #   3) P_bar
        # ============================================================

        # Pmax_vec = _as_power_vector(P_total_max, K)

        # # ============================================================
        # # 1) exact constrained P_tilda block FIRST
        # #    P_tilda is updated using old P_bar and old P_prime fixed.
        # # ============================================================
        # P_tilda_raw = compute_P_tilda(
        #     K,
        #     P_bar_old,          # old P_bar because P_bar is not updated yet
        #     w_comm_eff,
        #     kappa_S,
        #     kappa_V1,
        #     kappa_K1,
        #     kappa_V0,
        #     kappa_K0,
        #     kappa_M1,
        #     kappa_M0,
        #     gamma,
        #     mu,
        # )

        # P_tilda_upper = Pmax_vec - P_bar_old - P_prime_old - eps_power
        # P_tilda_upper = np.maximum(P_tilda_upper, eps_power)

        # P_tilda_new = np.clip(
        #     P_tilda_raw,
        #     eps_power,
        #     P_tilda_upper,
        # )


        # # ============================================================
        # # 2) exact constrained P_prime block SECOND
        # #    P_prime is updated using old P_bar and updated P_tilda.
        # # ============================================================
        # P_prime_new = update_P_prime_block_exact(
        #     P_prime=P_prime_old,
        #     P_bar=P_bar_old,        # important: P_bar has not been updated yet
        #     P_tilda=P_tilda_new,    # use updated P_tilda
        #     a_sens=a_sens,
        #     b_sens=b_sens,
        #     n_sens=n_sens,
        #     mu_s=mu_s,
        #     y_s=y_s,
        #     target_weights=target_weights,
        #     w_s=w_s,
        #     P_total_max=P_total_max,
        #     eps_power=eps_power,
        #     inner_iters=50,
        #     inner_tol=1e-8,
        # )


        # # ============================================================
        # # 3) exact constrained P_bar block THIRD
        # #    P_bar is updated using updated P_tilda and updated P_prime.
        # # ============================================================
        # P_bar_raw = compute_P_bar(
        #     K,
        #     P_tilda_new,        # use updated P_tilda
        #     w_comm_eff,
        #     kappa_S,
        #     kappa_V1,
        #     kappa_K1,
        #     kappa_M1,
        #     kappa_Th1,
        #     kappa_DAC1,
        #     kappa_ADC1,
        #     Z_list,
        #     gamma,
        #     mu,
        # )

        # P_bar_upper = Pmax_vec - P_tilda_new - P_prime_new - eps_power
        # P_bar_upper = np.maximum(P_bar_upper, eps_power)

        # P_bar_new = np.clip(
        #     P_bar_raw,
        #     eps_power,
        #     P_bar_upper,
        # )


        # # ============================================================
        # # Final block targets for backtracking
        # # ============================================================
        # P_bar_target = P_bar_new
        # P_tilda_target = P_tilda_new
        # P_prime_target = P_prime_new

        (
            P_bar,
            P_tilda,
            P_prime,
            new_state,
            eta_used,
            accepted,
        ) = accept_three_power_update_with_backtracking(
            K=K,
            w=w,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1,
            kappa_DAC0=kappa_DAC0,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            kappa_Th1=kappa_Th1,
            kappa_ADC1=kappa_ADC1,
            kappa_Th0=kappa_Th0,
            kappa_ADC0=kappa_ADC0,
            P_bar_old=P_bar_old,
            P_tilda_old=P_tilda_old,
            P_prime_old=P_prime_old,
            P_bar_target=P_bar_target,
            P_tilda_target=P_tilda_target,
            P_prime_target=P_prime_target,
            old_state=old_state,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_c=w_c,
            w_s=w_s,
            d=d,
            tau=tau,
            min_eta=1e-4,
            accept_tol=1e-12,
        )

        # total_power = P_bar + P_tilda + P_prime
        # Pmax_vec = _as_power_vector(P_total_max, K)

        # if Pmax_vec is not None:
        #     max_violation = np.max(total_power - Pmax_vec)

        #     if max_violation > 1e-8:
        #         raise RuntimeError(
        #             f"Power constraint violated after iteration {t+1}: "
        #             f"max violation = {max_violation:.3e}"
        #        )
        joint_obj = new_state["joint_obj"]
        comm_wsr = new_state["comm_wsr"]
        sensing_wsr = new_state["sensing_wsr"]

        joint_history.append(joint_obj)
        comm_history.append(comm_wsr)
        sensing_history.append(sensing_wsr)

        delta = joint_obj - old_state["joint_obj"]

        if t > 0 and delta < -1e-10:
            print(f"Warning: joint objective decreased: {delta:.3e}")

        if verbose:
            print(
                f"iter={t+1:03d} | joint={joint_obj:.6f} | "
                f"comm={comm_wsr:.6f} | sensing={sensing_wsr:.6f} | "
                f"delta={delta:.3e} | eta={eta_used:.3e} | accepted={accepted}"
            )

        # Stop only when improvement is small and nonnegative
        if t > 0 and 0.0 <= delta < epsilon:
            converged = True
            convergence_iter = t + 1

            if verbose:
                print(
                    f"\nConverged at iteration {convergence_iter} "
                    f"(Δjoint = {delta:.2e})"
                )

            break

        wsr_prev = joint_obj

    elapsed_time_sec = time.perf_counter() - start_time

    if not converged:
        convergence_iter = len(joint_history)

    gamma = np.array([
        compute_SINR_exact(
            k, K, kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1, kappa_ADC1,
            P_bar, P_tilda, P_bar * kappa_Th1 + kappa_Th0, P_bar * kappa_ADC1 + kappa_ADC0,
        )
        for k in range(K)
    ])

    sensing_final = compute_sensing_components_from_coeffs(
        P_prime=P_prime,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
    )

    return {
        "joint_history": np.asarray(joint_history, dtype=float),
        "comm_history": np.asarray(comm_history, dtype=float),
        "sensing_history": np.asarray(sensing_history, dtype=float),
        "P_bar_opt": P_bar,
        "P_tilda_opt": P_tilda,
        "P_prime_opt": P_prime,
        "gamma_opt": gamma,
        "sensing_final": sensing_final,
        "converged": converged,
        "convergence_iter": convergence_iter,
        "elapsed_time_sec": elapsed_time_sec,
        "final_delta": float(joint_history[-1] - joint_history[-2]) if len(joint_history) > 1 else np.nan,
        "kappas": {
            "kappa_S": kappa_S,
            "kappa_V1": kappa_V1,
            "kappa_V0": kappa_V0,
            "kappa_K1": kappa_K1,
            "kappa_K0": kappa_K0,
            "kappa_DAC1": kappa_DAC1,
            "kappa_DAC0": kappa_DAC0,
            "kappa_Th1": kappa_Th1,
            "kappa_ADC1": kappa_ADC1,
            "kappa_M1": kappa_M1,
            "kappa_M0": kappa_M0,
            "kappa_Th0": kappa_Th0,
            "kappa_ADC0": kappa_ADC0,
        },
    }

def get_mat_var(mat, possible_names, required=True):
    for name in possible_names:
        if name in mat:
            return mat[name], name

    if not required:
        return None, None

    available = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(
        f"None of these variables were found: {possible_names}\n"
        f"Available variables are:\n{available}"
    )


def take_setup_last_dim(X, setup_idx, base_ndim, name):
    """
    If X has no setup dimension, return X.
    If X has setup dimension as last dimension, return X[..., setup_idx].
    """
    X = np.asarray(X)

    if X.ndim == base_ndim:
        return X

    if X.ndim == base_ndim + 1:
        return X[..., setup_idx]

    raise ValueError(
        f"{name} has shape {X.shape}. Expected {base_ndim}D or {base_ndim+1}D."
    )
def db2lin(db):
    return 10.0 ** (db / 10.0)


def lin2db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def hermitianize(X):
    return 0.5 * (X + X.conj().T)


def relerr(A, B):
    return np.linalg.norm(A - B, "fro") / (np.linalg.norm(A, "fro") + 1e-30)


def _slice_k(lst_2d, k):
    return [lst_2d[l][k] for l in range(len(lst_2d))]


def mat_4d_to_lk_list(X, K, M, hermitian=True):
    out = [[] for _ in range(M)]

    for l in range(M):
        for k in range(K):
            X_lk = np.asarray(X[:, :, k, l], dtype=complex)
            if hermitian:
                X_lk = hermitianize(X_lk)
            out[l].append(X_lk)

    return out


def mat_3d_ap_to_list(X, M, hermitian=True):
    out = []

    for l in range(M):
        X_l = np.asarray(X[:, :, l], dtype=complex)
        if hermitian:
            X_l = hermitianize(X_l)
        out.append(X_l)

    return out


def hbar_to_lk_list(Hbar, K, M):
    out = [[] for _ in range(M)]

    for l in range(M):
        for k in range(K):
            h_lk = np.asarray(Hbar[:, k, l], dtype=complex).reshape(-1, 1)
            out[l].append(h_lk)

    return out


def Z_to_list(Z_matrices, K):

    return [
        np.asarray(Z_matrices[:, :, k], dtype=complex)
        for k in range(K)
    ]


def build_W_list_MR(M, K, N):
    """
    MR combiner W_lk = I_N.
    """
    return [
        [np.eye(N, dtype=complex) for _ in range(K)]
        for _ in range(M)
    ]



# ============
# SENSING PROJECTION FUNCTIONS
# v_l,q = b_l,q  (MR sensing combiner)
# q = target index, not UE index
# ============

def _col(x):
    return np.asarray(x, dtype=complex).reshape(-1, 1)


def get_b_lq(b_target_all, l, q):
    """
    b_target_all has shape N x M x Tg.
    Returns b_{l,q} as N x 1.
    """
    return _col(b_target_all[:, l, q])


def build_sensing_v_list_MR(b_target_all, q):
    """
    MR sensing combiner:
        v_{l,q} = b_{l,q}
    """
    M = b_target_all.shape[1]
    return [get_b_lq(b_target_all, l, q) for l in range(M)]

def build_sensing_v_list_from_V(V_combiner_all, q, iter_idx):
    """
    V_combiner_all shape: N x M x Tg x num_iterations

    Returns:
        v_list[l] = v_{l,q} for MC iteration iter_idx
    """
    M = V_combiner_all.shape[1]
    return [_col(V_combiner_all[:, l, q, iter_idx]) for l in range(M)]

def estimate_sigma2_ilt_from_Cref(C_h_ref_all, b_target_all):
    """
    Estimate sigma^2_{i,l,t} from:

        C_h_ref[l][i] ≈ sum_t sigma2[i,l,t] b_{l,t} b_{l,t}^H

    Returns:
        sigma2_ilt with shape K x M x Tg
    """
    M = len(C_h_ref_all)
    K = len(C_h_ref_all[0])
    Tg = b_target_all.shape[2]

    sigma2_ilt = np.zeros((K, M, Tg), dtype=float)

    for l in range(M):
        Phi_cols = []
        for t in range(Tg):
            b_lt = get_b_lq(b_target_all, l, t)
            Bt = b_lt @ b_lt.conj().T
            Phi_cols.append(Bt.reshape(-1))

        Phi = np.stack(Phi_cols, axis=1)

        for i in range(K):
            Cref_li = np.asarray(C_h_ref_all[l][i], dtype=complex)
            c = Cref_li.reshape(-1)

            sol, *_ = np.linalg.lstsq(Phi, c, rcond=None)
            sol = np.maximum(np.real(sol), 0.0)

            sigma2_ilt[i, l, :] = sol

    return sigma2_ilt

    """
    P_S,q = P' T^2 sum_l sum_i alpha_i^2 sigma^2_{i,l,q}
            |v_lq^H A_l b_lq|^2
    """
def compute_Ps_sensing_proj(q, P_sense, T,
                            alpha_list, A_list,
                            b_target_all, sigma2_ilt,
                            v_list):
    K = len(alpha_list)
    M = len(A_list)

    total = 0.0

    for l in range(M):
        A_l = A_list[l]
        v_lq = v_list[l]
        b_lq = get_b_lq(b_target_all, l, q)

        gain = np.abs((v_lq.conj().T @ A_l @ b_lq)[0, 0]) ** 2

        for i in range(K):
            total += (alpha_list[i] ** 2) * sigma2_ilt[i, l, q] * gain

    return float(np.real(P_sense * (T ** 2) * total))


def compute_Pinter_sensing_proj(q, P_sense, T,
                                alpha_list, A_list,
                                b_target_all, sigma2_ilt,
                                v_list):
    """
    P_inter,q = P' T^2 sum_{t != q} sum_l sum_i alpha_i^2 sigma^2_{i,l,t}
                |v_lq^H A_l b_lt|^2
    """
    K = len(alpha_list)
    M = len(A_list)
    Tg = b_target_all.shape[2]

    total = 0.0

    for t in range(Tg):
        if t == q:
            continue

        for l in range(M):
            A_l = A_list[l]
            v_lq = v_list[l]
            b_lt = get_b_lq(b_target_all, l, t)

            gain = np.abs((v_lq.conj().T @ A_l @ b_lt)[0, 0]) ** 2

            for i in range(K):
                total += (alpha_list[i] ** 2) * sigma2_ilt[i, l, t] * gain

    return float(np.real(P_sense * (T ** 2) * total))


def compute_Pclutter_sensing_proj(q, P_sense, T,
                                  alpha_list, A_list,
                                  v_list,
                                  H_direct_iter=None,
                                  C_h_dir_all=None):
    """
    P_clutter,q = P' T^2 E | sum_l sum_i alpha_i v_lq^H A_l h_dir_li |^2

    If H_direct_iter is available, uses Monte-Carlo samples.
    Else uses covariance approximation.
    """
    K = len(alpha_list)
    M = len(A_list)

    if H_direct_iter is not None:
        num_samples = H_direct_iter.shape[3]
        acc = 0.0

        for s in range(num_samples):
            z = 0.0 + 0.0j

            for l in range(M):
                A_l = A_list[l]
                v_lq = v_list[l]

                for i in range(K):
                    h_dir_li = _col(H_direct_iter[:, i, l, s])
                    z += alpha_list[i] * (v_lq.conj().T @ A_l @ h_dir_li)[0, 0]

            acc += np.abs(z) ** 2

        expectation = acc / max(num_samples, 1)
        return float(np.real(P_sense * (T ** 2) * expectation))

    if C_h_dir_all is None:
        return np.nan

    total = 0.0

    for l in range(M):
        A_l = A_list[l]
        v_lq = v_list[l]

        for i in range(K):
            C_li = C_h_dir_all[l][i]
            val = v_lq.conj().T @ A_l @ C_li @ A_l.conj().T @ v_lq
            total += (alpha_list[i] ** 2) * np.real(val[0, 0])

    return float(np.real(P_sense * (T ** 2) * total))


def compute_PDAC_sensing_proj(q, T,
                              sigma_d2, A_list,
                              v_list,
                              H_true_iter=None,
                              C_h_total_all=None):
    """
    P_DAC,q = T sum_i sigma_DAC_i^2
              E | sum_l v_lq^H A_l h_li |^2

    If H_true_iter is available, uses Monte-Carlo samples.
    Else uses covariance approximation.
    """
    K = len(sigma_d2)
    M = len(A_list)

    total = 0.0

    if H_true_iter is not None:
        num_samples = H_true_iter.shape[3]

        for i in range(K):
            acc_i = 0.0

            for s in range(num_samples):
                z_i = 0.0 + 0.0j

                for l in range(M):
                    A_l = A_list[l]
                    v_lq = v_list[l]
                    h_li = _col(H_true_iter[:, i, l, s])

                    z_i += (v_lq.conj().T @ A_l @ h_li)[0, 0]

                acc_i += np.abs(z_i) ** 2

            expectation_i = acc_i / max(num_samples, 1)
            total += sigma_d2[i] * expectation_i

        return float(np.real(T * total))

    if C_h_total_all is None:
        return np.nan

    for i in range(K):
        acc_i = 0.0

        for l in range(M):
            A_l = A_list[l]
            v_lq = v_list[l]
            C_li = C_h_total_all[l][i]

            val = v_lq.conj().T @ A_l @ C_li @ A_l.conj().T @ v_lq
            acc_i += np.real(val[0, 0])

        total += sigma_d2[i] * acc_i

    return float(np.real(T * total))


def compute_PTh_sensing_proj(q, T, sigma_w2, A_list, v_list):
    """
    P_Th,q = T sigma_w^2 sum_l v_lq^H A_l A_l^H v_lq
    """
    total = 0.0

    for l in range(len(A_list)):
        A_l = A_list[l]
        v_lq = v_list[l]

        val = v_lq.conj().T @ A_l @ A_l.conj().T @ v_lq
        total += np.real(val[0, 0])

    return float(np.real(T * sigma_w2 * total))


def compute_PADC_sensing_proj(q, T, Theta_list, v_list):
    """
    P_ADC,q = T sum_l v_lq^H Theta_l v_lq
    """
    total = 0.0

    for l in range(len(Theta_list)):
        Theta_l = Theta_list[l]
        v_lq = v_list[l]

        val = v_lq.conj().T @ Theta_l @ v_lq
        total += np.real(val[0, 0])

    return float(np.real(T * total))


def compute_sensing_projection_components(P_sense, T,
                                           alpha_list,
                                           sigma_d2,
                                           A_list,
                                           Theta_list,
                                           b_target_all,
                                           C_h_ref_all,
                                           sigma_w2,
                                           H_direct_iter=None,
                                           H_true_iter=None,
                                           C_h_dir_all=None,
                                           C_h_total_all=None,
                                           alpha_var_matrix_all=None):
    """
    Computes sensing SINR components for all targets q = 0,...,Tg-1.
    """
    Tg = b_target_all.shape[2]

    if alpha_var_matrix_all is not None:
        sigma2_ilt = np.asarray(alpha_var_matrix_all, dtype=float)
    else:
        sigma2_ilt = estimate_sigma2_ilt_from_Cref(
            C_h_ref_all=C_h_ref_all,
            b_target_all=b_target_all
        )

    PS = np.zeros(Tg)
    Pinter = np.zeros(Tg)
    Pclutter = np.zeros(Tg)
    PDAC = np.zeros(Tg)
    PTh = np.zeros(Tg)
    PADC = np.zeros(Tg)
    SINR = np.zeros(Tg)

    for q in range(Tg):
        v_list = build_sensing_v_list_MR(b_target_all, q)

        PS[q] = compute_Ps_sensing_proj(
            q, P_sense, T,
            alpha_list, A_list,
            b_target_all, sigma2_ilt,
            v_list
        )

        Pinter[q] = compute_Pinter_sensing_proj(
            q, P_sense, T,
            alpha_list, A_list,
            b_target_all, sigma2_ilt,
            v_list
        )

        Pclutter[q] = compute_Pclutter_sensing_proj(
            q, P_sense, T,
            alpha_list, A_list,
            v_list,
            H_direct_iter=H_direct_iter,
            C_h_dir_all=C_h_dir_all
        )

        PDAC[q] = compute_PDAC_sensing_proj(
            q, T,
            sigma_d2,
            A_list,
            v_list,
            H_true_iter=H_true_iter,
            C_h_total_all=C_h_total_all
        )

        PTh[q] = compute_PTh_sensing_proj(
            q, T, sigma_w2, A_list, v_list
        )

        PADC[q] = compute_PADC_sensing_proj(
            q, T, Theta_list, v_list
        )

        den = Pinter[q] + Pclutter[q] + PDAC[q] + PTh[q] + PADC[q]
        SINR[q] = PS[q] / (den + 1e-30)

    return {
        "PS": PS,
        "Pinter": Pinter,
        "Pclutter": Pclutter,
        "PDAC": PDAC,
        "PTh": PTh,
        "PADC": PADC,
        "SINR": SINR,
        "SINR_dB": lin2db(SINR),
        "sigma2_ilt": sigma2_ilt,
    }

def compute_sensing_projection_components_with_Viter(
    P_sense, T,
    alpha_list,
    sigma_d2,
    A_list,
    Theta_list,
    b_target_all,
    C_h_ref_all,
    sigma_w2,
    H_direct_iter,
    H_true_iter,
    C_h_dir_all=None,
    C_h_total_all=None,
    alpha_var_matrix_all=None,
    V_combiner_all=None,
):
    """
    Sensing projection using clutter-aware V_combiner_all.

    V_combiner_all shape:
        N x M x Tg x num_iterations
    """

    Tg = b_target_all.shape[2]
    K = len(alpha_list)
    M = len(A_list)

    if alpha_var_matrix_all is not None:
        sigma2_ilt = np.asarray(alpha_var_matrix_all, dtype=float)
    else:
        sigma2_ilt = estimate_sigma2_ilt_from_Cref(
            C_h_ref_all=C_h_ref_all,
            b_target_all=b_target_all
        )

    num_iter = V_combiner_all.shape[3]

    PS = np.zeros(Tg)
    Pinter = np.zeros(Tg)
    Pclutter = np.zeros(Tg)
    PDAC = np.zeros(Tg)
    PTh = np.zeros(Tg)
    PADC = np.zeros(Tg)
    SINR = np.zeros(Tg)

    for q in range(Tg):

        PS_acc = 0.0
        Pinter_acc = 0.0
        Pclutter_acc = 0.0
        PDAC_acc = 0.0
        PTh_acc = 0.0
        PADC_acc = 0.0

        for n in range(num_iter):

            v_list = build_sensing_v_list_from_V(V_combiner_all, q, n)

            PS_acc += compute_Ps_sensing_proj(
                q, P_sense, T,
                alpha_list, A_list,
                b_target_all, sigma2_ilt,
                v_list
            )

            Pinter_acc += compute_Pinter_sensing_proj(
                q, P_sense, T,
                alpha_list, A_list,
                b_target_all, sigma2_ilt,
                v_list
            )

            # Clutter: use same H_direct_iter sample n as the V combiner
            z_clutter = 0.0 + 0.0j

            for l in range(M):
                A_l = A_list[l]
                v_lq = v_list[l]

                for i in range(K):
                    h_dir_li = _col(H_direct_iter[:, i, l, n])
                    z_clutter += alpha_list[i] * (
                        v_lq.conj().T @ A_l @ h_dir_li
                    )[0, 0]

            Pclutter_acc += P_sense * (T ** 2) * (np.abs(z_clutter) ** 2)

            # DAC distortion: use same H_true_iter sample n
            pdac_n = 0.0

            for i in range(K):
                z_i = 0.0 + 0.0j

                for l in range(M):
                    A_l = A_list[l]
                    v_lq = v_list[l]
                    h_li = _col(H_true_iter[:, i, l, n])

                    z_i += (v_lq.conj().T @ A_l @ h_li)[0, 0]

                pdac_n += sigma_d2[i] * (np.abs(z_i) ** 2)

            PDAC_acc += T * pdac_n

            PTh_acc += compute_PTh_sensing_proj(
                q, T, sigma_w2, A_list, v_list
            )

            PADC_acc += compute_PADC_sensing_proj(
                q, T, Theta_list, v_list
            )

        PS[q] = np.real(PS_acc / num_iter)
        Pinter[q] = np.real(Pinter_acc / num_iter)
        Pclutter[q] = np.real(Pclutter_acc / num_iter)
        PDAC[q] = np.real(PDAC_acc / num_iter)
        PTh[q] = np.real(PTh_acc / num_iter)
        PADC[q] = np.real(PADC_acc / num_iter)

        den = Pinter[q] + Pclutter[q] + PDAC[q] + PTh[q] + PADC[q]
        SINR[q] = PS[q] / (den + 1e-30)

    sensing_rate = np.log2(1.0 + SINR)

    return {
        "PS": PS,
        "Pinter": Pinter,
        "Pclutter": Pclutter,
        "PDAC": PDAC,
        "PTh": PTh,
        "PADC": PADC,
        "SINR": SINR,
        "SINR_dB": lin2db(SINR),
        "rate": sensing_rate,
        "sum_se": float(np.sum(sensing_rate)),
        "wsr": float(np.sum(sensing_rate)),
        "sigma2_ilt": sigma2_ilt,
    }

def print_sensing_projection_components(label, sensing):
    print(f"\nSensing Projection Components: {label}")
    print(
        f"{'target':>6} | {'PS':>12} | {'Pinter':>12} | {'Pclutter':>12} | "
        f"{'PDAC':>12} | {'PTh':>12} | {'PADC':>12} | {'SINR':>12}"
    )
    print("-" * 105)

    Tg = len(sensing["SINR"])

    for q in range(Tg):
        print(
            f"{q:6d} | "
            f"{sensing['PS'][q]:12.4e} | "
            f"{sensing['Pinter'][q]:12.4e} | "
            f"{sensing['Pclutter'][q]:12.4e} | "
            f"{sensing['PDAC'][q]:12.4e} | "
            f"{sensing['PTh'][q]:12.4e} | "
            f"{sensing['PADC'][q]:12.4e} | "
            f"{sensing['SINR'][q]:12.4e}"
        )

def load_matlab_direct_values(mat_file, setup_idx=0, K_use=8, M_use=16, N_use=16):
    mat = loadmat(mat_file)

    A_l_all, A_name = get_mat_var(
        mat,
        ["A_l_allsetups", "A_l_all"]
    )

    C_h_all_CF, C_h_all_name = get_mat_var(
        mat,
        ["C_h_all_CF_allsetups", "C_h_all_CF"]
    )

    C_h_dir_all_CF, C_h_dir_name = get_mat_var(
        mat,
        ["C_h_dir_all_CF_allsetups", "C_h_dir_all_CF"]
    )

    C_h_ref_all_CF, C_h_ref_name = get_mat_var(
        mat,
        ["C_h_ref_all_CF_allsetups", "C_h_ref_all_CF"]
    )

    # Theta_l_all_CF, Theta_name = get_mat_var(
    #     mat,
    #     ["Theta_l_all_CF_allsetups", "Theta_l_all_CF"]
    # )

    Z_matrices, Z_name = get_mat_var(
        mat,
        ["Z_matrices_allsetups", "Z_matrices"]
    )

    alpha_dac_all, alpha_name = get_mat_var(
        mat,
        ["alpha_dac_allsetups", "alpha_dac_all"]
    )
    alpha_var_matrix_all, alpha_var_name = get_mat_var(
        mat,
        ["alpha_var_matrix_allsetups", "alpha_var_matrix_all"],
        required=False
    )
    # Sensing-only variables. They are optional so the classical WSR code still runs
    # with old datasets that do not contain sensing samples/target steering vectors.
    b_target_all, b_target_name = get_mat_var(
        mat,
        ["b_target_allsetups", "b_target_all"],
        required=False
    )

    H_direct_iter, H_direct_name = get_mat_var(
        mat,
        ["H_direct_iter_allsetups", "H_direct_iter"],
        required=False
    )

    H_true_iter, H_true_name = get_mat_var(
        mat,
        ["H_true_iter_allsetups", "H_true_iter"],
        required=False
    )

    V_combiner_all, V_combiner_name = get_mat_var(
        mat,
        ["V_combiner_all", "V_combiner_allsetups"],
        required=False
    )

    R_mat, R_name = get_mat_var(
        mat,
        ["R_allsetups", "R"],
        required=False
    )

    h_bar_mat, hbar_name = get_mat_var(
        mat,
        ["h_bar_allsetups", "h_bar", "h_bar_max"],
        required=False
    )

    if R_mat is None or h_bar_mat is None:
        raise KeyError(
            "Dataset is missing R and/or h_bar. "
            "Your current kappa/G functions require R and h_bar. "
            "Please add R_allsetups with shape N×N×K×M×setups "
            "and h_bar_allsetups with shape N×K×M×setups."
        )

    # Select setup if setup dimension exists
    A_l_all = take_setup_last_dim(A_l_all, setup_idx, 3, A_name)
    C_h_all_CF = take_setup_last_dim(C_h_all_CF, setup_idx, 4, C_h_all_name)
    C_h_dir_all_CF = take_setup_last_dim(C_h_dir_all_CF, setup_idx, 4, C_h_dir_name)
    C_h_ref_all_CF = take_setup_last_dim(C_h_ref_all_CF, setup_idx, 4, C_h_ref_name)
    # Theta_l_all_CF = take_setup_last_dim(Theta_l_all_CF, setup_idx, 3, Theta_name)
    Z_matrices = take_setup_last_dim(Z_matrices, setup_idx, 3, Z_name)
    R_mat = take_setup_last_dim(R_mat, setup_idx, 4, R_name)
    h_bar_mat = take_setup_last_dim(h_bar_mat, setup_idx, 3, hbar_name)

    if b_target_all is not None:
        b_target_all = take_setup_last_dim(b_target_all, setup_idx, 3, b_target_name)

    if H_direct_iter is not None:
        H_direct_iter = take_setup_last_dim(H_direct_iter, setup_idx, 4, H_direct_name)

    if H_true_iter is not None:
        H_true_iter = take_setup_last_dim(H_true_iter, setup_idx, 4, H_true_name)
    
    if V_combiner_all is not None:
        V_combiner_all = take_setup_last_dim(
            V_combiner_all, setup_idx, 4, V_combiner_name
        )
        
    if alpha_var_matrix_all is not None:
        alpha_var_matrix_all = take_setup_last_dim(
            alpha_var_matrix_all, setup_idx, 3, alpha_var_name
        )
    N = N_use
    M = M_use
    K = K_use

    tau = Z_matrices.shape[0]
    d = Z_matrices.shape[1]

    A_l_all = A_l_all[:N, :N, :M]
    C_h_all_CF = C_h_all_CF[:N, :N, :K, :M]
    C_h_dir_all_CF = C_h_dir_all_CF[:N, :N, :K, :M]
    C_h_ref_all_CF = C_h_ref_all_CF[:N, :N, :K, :M]
    # Theta_l_all_CF = Theta_l_all_CF[:N, :N, :M]
    Z_matrices = Z_matrices[:, :, :K]
    R_mat = R_mat[:N, :N, :K, :M]
    h_bar_mat = h_bar_mat[:N, :K, :M]

    if b_target_all is not None:
        b_target_all = np.asarray(b_target_all[:N, :M, :], dtype=complex)

    if H_direct_iter is not None:
        H_direct_iter = np.asarray(H_direct_iter[:N, :K, :M, :], dtype=complex)

    if H_true_iter is not None:
        H_true_iter = np.asarray(H_true_iter[:N, :K, :M, :], dtype=complex)

    if V_combiner_all is not None:
        V_combiner_all = np.asarray(
            V_combiner_all[:N, :M, :, :],
            dtype=complex
        )

    if alpha_var_matrix_all is not None:
        alpha_var_matrix_all = np.asarray(
            alpha_var_matrix_all[:K, :M, :],
            dtype=float
        )
    # alpha can be K×setups or K×1 or K
    alpha_dac_all = np.asarray(alpha_dac_all)

    if alpha_dac_all.ndim == 2 and alpha_dac_all.shape[1] > 1:
        alpha_list = np.real(alpha_dac_all[:K, setup_idx]).reshape(-1)
    else:
        alpha_list = np.real(alpha_dac_all).reshape(-1)[:K]

    A_list = mat_3d_ap_to_list(
        A_l_all,
        M,
        # hermitian=True,
        hermitian=False,
    )

    h_bar_list = hbar_to_lk_list(
        h_bar_mat,
        K,
        M,
    )

    R_list = mat_4d_to_lk_list(
        R_mat,
        K,
        M,
        # hermitian=True,
        hermitian=False,
    )

    Sigma_list = mat_4d_to_lk_list(
        C_h_dir_all_CF,
        K,
        M,
        # hermitian=True,
        hermitian=False,
    )

    C_h_ref_all = mat_4d_to_lk_list(
        C_h_ref_all_CF,
        K,
        M,
        # hermitian=True,
        hermitian=False,
    )

    C_h_total_all = mat_4d_to_lk_list(
        C_h_all_CF,
        K,
        M,
        # hermitian=True,
        hermitian=False,
    )

    # Theta_list = mat_3d_ap_to_list(
    #     Theta_l_all_CF,
    #     M,
    #     hermitian=True,
    # )

    Z_list = Z_to_list(
        Z_matrices,
        K,
    )

    return {
        "mat": mat,
        "N": N,
        "K": K,
        "M": M,
        "tau": tau,
        "d": d,
        "setup_idx": setup_idx,
        "alpha_list": alpha_list,
        "A_list": A_list,
        "h_bar_list": h_bar_list,
        "R_list": R_list,
        "Sigma_list": Sigma_list,
        "C_h_ref_all": C_h_ref_all,
        "C_h_total_all": C_h_total_all,
        # "Theta_list": Theta_list,
        "Z_list": Z_list,

        # Sensing-projection data
        "alpha_var_matrix_all": alpha_var_matrix_all,
        "b_target_all": b_target_all,
        "H_direct_iter": H_direct_iter,
        "H_true_iter": H_true_iter,
        "V_combiner_all": V_combiner_all,
    }



# ============
# Pretty-print helpers for joint optimization outputs
# ============

def _fmt_float(x, width=13, prec=6):
    try:
        x = float(np.real(x))
    except Exception:
        return f"{str(x):>{width}}"
    if np.isnan(x):
        return f"{'nan':>{width}}"
    if np.isinf(x):
        return f"{str(x):>{width}}"
    if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
        return f"{x:{width}.{prec}e}"
    return f"{x:{width}.{prec}f}"


def print_rule(width=118, char="-"):
    print(char * width)


def print_heading(title, width=118):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_key_value_table(title, rows, width=118):
    print_heading(title, width)
    key_w = max(18, min(40, max(len(str(k)) for k, _ in rows) if rows else 18))
    print(f"{'Item':<{key_w}} | {'Value':>24}")
    print_rule(key_w + 3 + 24)
    for key, val in rows:
        if isinstance(val, (float, int, np.floating, np.integer)):
            val_str = _fmt_float(val, width=24, prec=6)
        else:
            val_str = f"{str(val):>24}"
        print(f"{str(key):<{key_w}} | {val_str}")


def print_power_split_table(Pmax_db, Pmax_lin, P_bar_init, P_tilda_init, P_prime_init, P_total_max):
    used = P_bar_init + P_tilda_init + P_prime_init
    slack = np.asarray(P_total_max if not np.isscalar(P_total_max) else P_total_max * np.ones_like(used)) - used
    rows = [
        ("Pmax dB", Pmax_db),
        ("Pmax linear", Pmax_lin),
        ("init P_bar/user", float(P_bar_init[0])),
        ("init P_tilda/user", float(P_tilda_init[0])),
        ("init P_prime/user", float(P_prime_init[0])),
        ("init total/user", float(used[0])),
        ("init slack/user", float(slack[0])),
    ]
    print_key_value_table("Initial Per-User Power Split", rows)


def print_power_allocation_table(label, P_bar, P_tilda, P_prime, P_total_max, max_rows=None):
    P_bar = np.asarray(P_bar, dtype=float).reshape(-1)
    P_tilda = np.asarray(P_tilda, dtype=float).reshape(-1)
    P_prime = np.asarray(P_prime, dtype=float).reshape(-1)
    K = len(P_bar)
    if np.isscalar(P_total_max):
        Pmax = float(P_total_max) * np.ones(K)
    else:
        Pmax = np.asarray(P_total_max, dtype=float).reshape(K)
    total = P_bar + P_tilda + P_prime
    slack = Pmax - total

    print_heading(label)
    print(
        f"{'k':>3} | {'P_bar':>13} | {'P_tilda':>13} | {'P_prime':>13} | "
        f"{'total':>13} | {'Pmax':>13} | {'slack':>13}"
    )
    print_rule(95)
    nshow = K if max_rows is None else min(K, max_rows)
    for k in range(nshow):
        print(
            f"{k:3d} | {_fmt_float(P_bar[k])} | {_fmt_float(P_tilda[k])} | "
            f"{_fmt_float(P_prime[k])} | {_fmt_float(total[k])} | "
            f"{_fmt_float(Pmax[k])} | {_fmt_float(slack[k])}"
        )
    if nshow < K:
        print(f"... ({K - nshow} more users omitted)")
    print_rule(95)
    print(
        f"{'SUM':>3} | {_fmt_float(np.sum(P_bar))} | {_fmt_float(np.sum(P_tilda))} | "
        f"{_fmt_float(np.sum(P_prime))} | {_fmt_float(np.sum(total))} | "
        f"{_fmt_float(np.sum(Pmax))} | {_fmt_float(np.sum(slack))}"
    )


def print_sensing_coeff_table(label, sensing, target_weights=None, w_s=1.0):
    if sensing is None:
        return
    sinr = np.asarray(sensing.get("SINR", []), dtype=float).reshape(-1)
    if len(sinr) == 0:
        return
    rate = np.asarray(sensing.get("rate", np.log2(1.0 + np.maximum(sinr, 0.0))), dtype=float).reshape(-1)
    sinr_db = np.asarray(sensing.get("SINR_dB", lin2db(sinr)), dtype=float).reshape(-1)
    S = np.asarray(sensing.get("S", np.full_like(sinr, np.nan)), dtype=float).reshape(-1)
    D = np.asarray(sensing.get("D", np.full_like(sinr, np.nan)), dtype=float).reshape(-1)
    Tg = len(sinr)
    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)
    omega = np.asarray(target_weights, dtype=float).reshape(Tg)
    weighted_rate = float(w_s) * omega * rate

    print_heading(label)
    print(
        f"{'q':>3} | {'S_q':>13} | {'D_q':>13} | {'SINR':>13} | "
        f"{'SINR(dB)':>13} | {'rate':>13} | {'omega':>13} | {'w_s*omega*rate':>17}"
    )
    print_rule(111)
    for q in range(Tg):
        print(
            f"{q:3d} | {_fmt_float(S[q])} | {_fmt_float(D[q])} | {_fmt_float(sinr[q])} | "
            f"{_fmt_float(sinr_db[q])} | {_fmt_float(rate[q])} | {_fmt_float(omega[q])} | "
            f"{_fmt_float(weighted_rate[q], width=17)}"
        )
    print_rule(111)
    print(
        f"{'SUM':>3} | {'':>13} | {'':>13} | {'':>13} | {'':>13} | "
        f"{_fmt_float(np.sum(rate))} | {'':>13} | {_fmt_float(np.sum(weighted_rate), width=17)}"
    )



def print_sensing_before_after_table(label, sensing_before, sensing_after, target_weights=None, w_s=1.0):
    """Print target-wise sensing performance before and after an optimization."""
    if sensing_before is None or sensing_after is None:
        return

    sinr_b = np.asarray(sensing_before.get("SINR", []), dtype=float).reshape(-1)
    sinr_a = np.asarray(sensing_after.get("SINR", []), dtype=float).reshape(-1)
    if len(sinr_b) == 0 or len(sinr_a) == 0:
        return
    if len(sinr_b) != len(sinr_a):
        raise ValueError("Before/after sensing arrays have different numbers of targets.")

    Tg = len(sinr_b)
    rate_b = np.asarray(sensing_before.get("rate", np.log2(1.0 + np.maximum(sinr_b, 0.0))), dtype=float).reshape(Tg)
    rate_a = np.asarray(sensing_after.get("rate", np.log2(1.0 + np.maximum(sinr_a, 0.0))), dtype=float).reshape(Tg)
    sinr_db_b = np.asarray(sensing_before.get("SINR_dB", lin2db(sinr_b)), dtype=float).reshape(Tg)
    sinr_db_a = np.asarray(sensing_after.get("SINR_dB", lin2db(sinr_a)), dtype=float).reshape(Tg)

    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)
    omega = np.asarray(target_weights, dtype=float).reshape(Tg)
    weighted_b = float(w_s) * omega * rate_b
    weighted_a = float(w_s) * omega * rate_a

    print_heading(label)
    print(
        f"{'q':>3} | {'SINR before':>13} | {'SINR after':>13} | {'SINR dB before':>15} | "
        f"{'SINR dB after':>14} | {'rate before':>13} | {'rate after':>13} | "
        f"{'weighted before':>16} | {'weighted after':>15} | {'gain':>13}"
    )
    print_rule(147)
    for q in range(Tg):
        gain_q = weighted_a[q] - weighted_b[q]
        print(
            f"{q:3d} | {_fmt_float(sinr_b[q])} | {_fmt_float(sinr_a[q])} | "
            f"{_fmt_float(sinr_db_b[q], width=15)} | {_fmt_float(sinr_db_a[q], width=14)} | "
            f"{_fmt_float(rate_b[q])} | {_fmt_float(rate_a[q])} | "
            f"{_fmt_float(weighted_b[q], width=16)} | {_fmt_float(weighted_a[q], width=15)} | "
            f"{_fmt_float(gain_q)}"
        )
    print_rule(147)
    print(
        f"{'SUM':>3} | {'':>13} | {'':>13} | {'':>15} | {'':>14} | "
        f"{_fmt_float(np.sum(rate_b))} | {_fmt_float(np.sum(rate_a))} | "
        f"{_fmt_float(np.sum(weighted_b), width=16)} | {_fmt_float(np.sum(weighted_a), width=15)} | "
        f"{_fmt_float(np.sum(weighted_a) - np.sum(weighted_b))}"
    )


def print_weight_table(w_c, w_s, target_weights):
    """Print how communication and sensing weights enter the joint objective."""
    target_weights = np.asarray(target_weights, dtype=float).reshape(-1)
    rows = [
        ("communication weight w_c", float(w_c)),
        ("sensing weight w_s", float(w_s)),
        ("number of targets", len(target_weights)),
        ("sum target weights", float(np.sum(target_weights))),
        ("min omega_q", float(np.min(target_weights))),
        ("max omega_q", float(np.max(target_weights))),
        ("objective", "w_c*WSR_comm + w_s*sum_q omega_q*R_sens,q"),
    ]
    print_key_value_table("Objective Weights Used by This Run", rows)


# ============
# Clean report helpers: one before table + one after table per power
# ============

def _sum_power(x):
    return float(np.sum(np.asarray(x, dtype=float)))


def _objective_row_values(comm_wsr, sensing_raw_no_ws, sensing_contribution, joint_obj, w_c, w_s):
    return [
        ("Communication", float(comm_wsr), float(w_c), float(w_c) * float(comm_wsr)),
        ("Sensing", float(sensing_raw_no_ws), float(w_s), float(sensing_contribution)),
        ("Joint", np.nan, np.nan, float(joint_obj)),
    ]


def print_objective_table(label, comm_wsr, sensing_raw_no_ws, sensing_contribution, joint_obj, w_c, w_s):
    """Print one compact objective table with comm, sensing, and joint values."""
    print_heading(label, width=118)
    print(
        f"{'Component':<15} | {'Raw value':>15} | {'Weight':>10} | {'Objective contribution':>24}"
    )
    print_rule(72)
    for name, raw, weight, contribution in _objective_row_values(
        comm_wsr, sensing_raw_no_ws, sensing_contribution, joint_obj, w_c, w_s
    ):
        raw_str = "" if np.isnan(raw) else _fmt_float(raw, width=15)
        weight_str = "" if np.isnan(weight) else _fmt_float(weight, width=10)
        print(
            f"{name:<15} | {raw_str:>15} | {weight_str:>10} | "
            f"{_fmt_float(contribution, width=24)}"
        )


def print_power_context_heading(result):
    """Main heading showing power, initial powers, optimized powers, w_c, and w_s."""
    Pmax_db = result.get("Pmax_db", np.nan)
    setup_idx = result.get("setup_idx", "")
    w_c = result.get("w_c", np.nan)
    w_s = result.get("w_s", np.nan)

    Pbar_i = result["P_bar_initial"]
    Pt_i = result["P_tilda_initial"]
    Pp_i = result["P_prime_initial"]
    Pbar_o = result["P_bar_opt"]
    Pt_o = result["P_tilda_opt"]
    Pp_o = result["P_prime_opt"]

    print_heading(
        f"Pmax = {Pmax_db:.1f} dB | setup = {setup_idx} | w_c = {w_c:.4g} | w_s = {w_s:.4g}",
        width=118,
    )
    print(
        f"{'Power state':<14} | {'sum P_bar':>13} | {'sum P_tilda':>13} | "
        f"{'sum P_prime':>13} | {'sum total':>13}"
    )
    print_rule(78)
    init_total = _sum_power(Pbar_i) + _sum_power(Pt_i) + _sum_power(Pp_i)
    opt_total = _sum_power(Pbar_o) + _sum_power(Pt_o) + _sum_power(Pp_o)
    print(
        f"{'Initial':<14} | {_fmt_float(_sum_power(Pbar_i))} | {_fmt_float(_sum_power(Pt_i))} | "
        f"{_fmt_float(_sum_power(Pp_i))} | {_fmt_float(init_total)}"
    )
    print(
        f"{'Optimized':<14} | {_fmt_float(_sum_power(Pbar_o))} | {_fmt_float(_sum_power(Pt_o))} | "
        f"{_fmt_float(_sum_power(Pp_o))} | {_fmt_float(opt_total)}"
    )


def print_comm_components_from_result(label, result, state="initial"):
    """Print communication power components using stored kappas and powers."""
    kappas = result["comm_kappas"]
    if state == "initial":
        P_bar = result["P_bar_initial"]
        P_tilda = result["P_tilda_initial"]
        P_th = result["P_th_initial"]
        P_adc = result["P_adc_initial"]
    elif state == "optimized":
        P_bar = result["P_bar_opt"]
        P_tilda = result["P_tilda_opt"]
        P_th = result["P_th_opt"]
        P_adc = result["P_adc_opt"]
    else:
        raise ValueError("state must be 'initial' or 'optimized'")

    print_power_components(
        label=label,
        K=len(P_bar),
        kappa_S=kappas["kappa_S"],
        kappa_V1=kappas["kappa_V1"],
        kappa_K1=kappas["kappa_K1"],
        kappa_V0=kappas["kappa_V0"],
        kappa_K0=kappas["kappa_K0"],
        kappa_DAC1=kappas["kappa_DAC1"],
        kappa_DAC0=kappas["kappa_DAC0"],
        kappa_M1=kappas["kappa_M1"],
        kappa_M0=kappas["kappa_M0"],
        kappa_Th1=kappas["kappa_Th1"],
        kappa_ADC1=kappas["kappa_ADC1"],
        P_bar=P_bar,
        P_tilda=P_tilda,
        P_th=P_th,
        P_adc=P_adc,
    )


def print_sensing_power_components(label, sensing, w_s=1.0):
    """Print sensing power components with interference and clutter separated."""
    if sensing is None:
        return

    S = np.asarray(sensing.get("S", []), dtype=float).reshape(-1)
    if len(S) == 0:
        return

    Pinter = np.asarray(
        sensing.get("Pinter", np.full_like(S, np.nan)),
        dtype=float,
    ).reshape(-1)

    Pclutter = np.asarray(
        sensing.get("Pclutter", np.full_like(S, np.nan)),
        dtype=float,
    ).reshape(-1)

    N_floor = np.asarray(
        sensing.get("N", np.full_like(S, np.nan)),
        dtype=float,
    ).reshape(-1)

    D = np.asarray(
        sensing.get("D", Pinter + Pclutter + N_floor),
        dtype=float,
    ).reshape(-1)

    sinr = np.asarray(
        sensing.get("SINR", S / (D + 1e-30)),
        dtype=float,
    ).reshape(-1)

    # sinr_db = np.asarray(
    #     sensing.get("SINR_dB", lin2db(sinr)),
    #     dtype=float,
    # ).reshape(-1)

    rate = np.asarray(
        sensing.get("rate", np.log2(1.0 + np.maximum(sinr, 0.0))),
        dtype=float,
    ).reshape(-1)

    print_heading(label, width=150)

    print(
        f"{'q':>3} | {'S desired':>13} | {'Pinter':>13} | {'Pclutter':>13} | "
        f"{'noise floor':>13} | {'D total':>13} | {'SINR':>13} | "
        f"{'rate':>13}"
    )
    print_rule(150)

    for q in range(len(S)):
        print(
            f"{q:3d} | "
            f"{_fmt_float(S[q])} | "
            f"{_fmt_float(Pinter[q])} | "
            f"{_fmt_float(Pclutter[q])} | "
            f"{_fmt_float(N_floor[q])} | "
            f"{_fmt_float(D[q])} | "
            f"{_fmt_float(sinr[q])} | "
            f"{_fmt_float(rate[q])} | "
        )

    print_rule(150)
    print(
        f"{'SUM':>3} | "
        f"{_fmt_float(np.sum(S))} | "
        f"{_fmt_float(np.sum(Pinter))} | "
        f"{_fmt_float(np.sum(Pclutter))} | "
        f"{_fmt_float(np.sum(N_floor))} | "
        f"{_fmt_float(np.sum(D))} | "
        f"{'':>13} | "
        f"{_fmt_float(np.sum(rate))} | "
    )
def print_full_power_report(result, print_components=True):
    """Print requested per-dB report: initial table, optimized table, and component tables."""
    w_c = float(result["w_c"])
    w_s = float(result["w_s"])
    initial_sensing = result["initial_sensing_components"]
    final_sensing = result["sensing_projection"]

    print_power_context_heading(result)

    print_objective_table(
        label="BEFORE OPTIMIZATION: objective values",
        comm_wsr=result["initial_comm_wsr"],
        sensing_raw_no_ws=initial_sensing.get("weighted_wsr_no_ws", 0.0),
        sensing_contribution=result["initial_sensing_wsr"],
        joint_obj=result["initial_joint_wsr"],
        w_c=w_c,
        w_s=w_s,
    )

    print_objective_table(
        label="AFTER OPTIMIZATION: objective values",
        comm_wsr=result["final_comm_wsr"],
        sensing_raw_no_ws=final_sensing.get("weighted_wsr_no_ws", 0.0),
        sensing_contribution=result["final_sensing_wsr"],
        joint_obj=result["final_wsr"],
        w_c=w_c,
        w_s=w_s,
    )

    if print_components:
        print_comm_components_from_result(
            label=f"COMMUNICATION components BEFORE optimization @ {result['Pmax_db']:.1f} dB",
            result=result,
            state="initial",
        )
        print_sensing_power_components(
            label=f"SENSING components BEFORE optimization @ {result['Pmax_db']:.1f} dB",
            sensing=initial_sensing,
            w_s=w_s,
        )
        print_comm_components_from_result(
            label=f"COMMUNICATION components AFTER optimization @ {result['Pmax_db']:.1f} dB",
            result=result,
            state="optimized",
        )
        print_sensing_power_components(
            label=f"SENSING components AFTER optimization @ {result['Pmax_db']:.1f} dB",
            sensing=final_sensing,
            w_s=w_s,
        )

def run_communication_only_optimisation(
    K,
    w,
    d,
    tau,
    kappa_S,
    kappa_V1,
    kappa_K1,
    kappa_V0,
    kappa_K0,
    kappa_DAC1,
    kappa_DAC0,
    kappa_Th1,
    kappa_ADC1,
    kappa_M1,
    kappa_M0,
    kappa_Th0,
    kappa_ADC0,
    Z_list,
    P_bar_init,
    P_tilda_init,
    P_prime_fixed,
    P_total_max=None,
    max_iters=200,
    epsilon=1e-6,
    eps_power=1e-12,
    verbose=False,
):
    """
    Classical communication-only FP/QT optimization.

    Optimized:
        P_bar
        P_tilda

    Fixed:
        P_prime

    Constraint:
        P_bar[k] + P_tilda[k] + P_prime_fixed[k] <= P_total_max[k]
    """

    P_bar = np.asarray(P_bar_init, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda_init, dtype=float).reshape(K)
    P_prime_fixed = np.asarray(P_prime_fixed, dtype=float).reshape(K)
    w = np.asarray(w, dtype=float).reshape(K)

    # Initial feasibility clipping with fixed sensing power.
    P_bar = clip_P_bar_joint(
        P_bar,
        P_tilda=P_tilda,
        P_prime=P_prime_fixed,
        P_total_max=P_total_max,
        eps=eps_power,
    )

    P_tilda = clip_P_tilda_joint(
        P_tilda,
        P_bar=P_bar,
        P_prime=P_prime_fixed,
        P_total_max=P_total_max,
        eps=eps_power,
    )

    P_th = P_bar * kappa_Th1 + kappa_Th0
    P_adc = P_bar * kappa_ADC1 + kappa_ADC0

    initial_comm_wsr = compute_WSR_exact(
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

    comm_history = []
    wsr_prev = initial_comm_wsr

    for t in range(max_iters):

        P_th = P_bar * kappa_Th1 + kappa_Th0
        P_adc = P_bar * kappa_ADC1 + kappa_ADC0

        gamma = np.array([
            compute_SINR_exact(
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
        ])

        mu = update_auxiliary(
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
            P_bar,
            P_tilda,
            P_th,
            P_adc,
        )

        # Update P_bar only, with P_prime fixed.
        P_bar = compute_P_bar(
            K,
            P_tilda,
            w,
            kappa_S,
            kappa_V1,
            kappa_K1,
            kappa_M1,
            kappa_Th1,
            kappa_DAC1,
            kappa_ADC1,
            Z_list,
            gamma,
            mu,
        )

        P_bar = clip_P_bar_joint(
            P_bar,
            P_tilda=P_tilda,
            P_prime=P_prime_fixed,
            P_total_max=P_total_max,
            eps=eps_power,
        )

        # Update P_tilda only, with P_prime fixed.
        P_tilda = compute_P_tilda(
            K,
            P_bar,
            w,
            kappa_S,
            kappa_V1,
            kappa_K1,
            kappa_V0,
            kappa_K0,
            kappa_M1,
            kappa_M0,
            gamma,
            mu,
        )

        P_tilda = clip_P_tilda_joint(
            P_tilda,
            P_bar=P_bar,
            P_prime=P_prime_fixed,
            P_total_max=P_total_max,
            eps=eps_power,
        )

        P_th = P_bar * kappa_Th1 + kappa_Th0
        P_adc = P_bar * kappa_ADC1 + kappa_ADC0

        comm_wsr = compute_WSR_exact(
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

        comm_history.append(float(comm_wsr))
        delta = float(comm_wsr - wsr_prev)

        if t > 0 and delta < -1e-10:
            print(f"Warning: communication-only objective decreased: {delta:.3e}")

        if verbose:
            print(
                f"comm-only iter={t+1:03d} | "
                f"comm={comm_wsr:.6f} | delta={delta:.3e}"
            )

        if t > 0 and abs(delta) < epsilon:
            break

        wsr_prev = float(comm_wsr)

    full_comm_history = (
        np.concatenate([[initial_comm_wsr], np.asarray(comm_history, dtype=float)])
        if len(comm_history)
        else np.asarray([initial_comm_wsr], dtype=float)
    )

    monotonic = bool(np.all(np.diff(full_comm_history) >= -1e-10))

    return {
        "initial_comm_wsr": float(initial_comm_wsr),
        "final_comm_wsr": float(full_comm_history[-1]),
        "comm_history": np.asarray(comm_history, dtype=float),
        "full_comm_history": full_comm_history,
        "P_bar_initial": np.asarray(P_bar_init, dtype=float).reshape(K),
        "P_tilda_initial": np.asarray(P_tilda_init, dtype=float).reshape(K),
        "P_prime_fixed": P_prime_fixed,
        "P_bar_opt": P_bar,
        "P_tilda_opt": P_tilda,
        "P_prime_opt": P_prime_fixed,
        "P_th_opt": P_th,
        "P_adc_opt": P_adc,
        "gamma_opt": gamma,
        "monotonic": monotonic,
        "iterations": len(comm_history),
    }
def run_sensing_only_optimisation(
    P_prime_init,
    P_bar_fixed,
    P_tilda_fixed,
    a_sens,
    b_sens,
    n_sens,
    target_weights=None,
    w_s=1.0,
    P_total_max=None,
    max_iters=200,
    epsilon=1e-6,
    eps_power=1e-12,
    verbose=False,
):
    """Optimize only sensing power P_prime while keeping P_bar and P_tilda fixed."""
    P_prime = np.asarray(P_prime_init, dtype=float).reshape(-1)
    P_bar_fixed = np.asarray(P_bar_fixed, dtype=float).reshape(-1)
    P_tilda_fixed = np.asarray(P_tilda_fixed, dtype=float).reshape(-1)
    K = len(P_prime)
    Tg = np.asarray(a_sens).shape[0]

    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float).reshape(Tg)

    P_prime = clip_P_prime_joint(
        P_prime,
        P_bar=P_bar_fixed,
        P_tilda=P_tilda_fixed,
        P_total_max=P_total_max,
        eps=eps_power,
    )

    sensing_initial = compute_sensing_components_from_coeffs(
        P_prime=P_prime,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
    )
    initial_sensing_wsr = float(sensing_initial["wsr"])

    sensing_history = []
    wsr_prev = initial_sensing_wsr

    for t in range(max_iters):
        mu_s = update_sensing_mu(P_prime, a_sens, b_sens, n_sens)
        y_s = update_sensing_y(
            P_prime,
            a_sens,
            b_sens,
            n_sens,
            mu_s=mu_s,
            target_weights=target_weights,
            w_s=w_s,
        )
        P_prime = update_P_prime(
            P_prime=P_prime,
            P_bar=P_bar_fixed,
            P_tilda=P_tilda_fixed,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            mu_s=mu_s,
            y_s=y_s,
            target_weights=target_weights,
            w_s=w_s,
            P_total_max=P_total_max,
            eps_power=eps_power,
        )

        sensing_wsr = compute_sensing_wsr(
            P_prime=P_prime,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_s=w_s,
        )
        sensing_history.append(float(sensing_wsr))
        delta = float(sensing_wsr - wsr_prev)

        if t > 0 and delta < -1e-10:
            print(f"Warning: sensing-only objective decreased: {delta:.3e}")

        if verbose:
            print(f"sensing-only iter={t+1:03d} | sensing={sensing_wsr:.6f} | delta={delta:.3e}")

        if t > 0 and abs(delta) < epsilon:
            break
        wsr_prev = float(sensing_wsr)

    sensing_final = compute_sensing_components_from_coeffs(
        P_prime=P_prime,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
    )
    sensing_history = np.asarray(sensing_history, dtype=float)
    full_sensing_history = np.concatenate([[initial_sensing_wsr], sensing_history]) if len(sensing_history) else np.asarray([initial_sensing_wsr])
    monotonic = bool(np.all(np.diff(full_sensing_history) >= -1e-10)) if len(full_sensing_history) > 1 else True

    return {
        "initial_sensing_wsr": initial_sensing_wsr,
        "final_sensing_wsr": float(sensing_final["wsr"]),
        "sensing_history": sensing_history,
        "full_sensing_history": full_sensing_history,
        "P_prime_initial": np.asarray(P_prime_init, dtype=float).reshape(K),
        "P_prime_opt": P_prime,
        "P_prime_sum": float(np.sum(P_prime)),
        "P_prime_mean": float(np.mean(P_prime)),
        "P_bar_fixed": P_bar_fixed,
        "P_tilda_fixed": P_tilda_fixed,
        "sensing_initial": sensing_initial,
        "sensing_final": sensing_final,
        "monotonic": monotonic,
        "iterations": len(sensing_history),
    }


def print_joint_summary_table(label, result):
    rows = [
        ("Pmax dB", result.get("Pmax_db", np.nan)),
        ("setup", result.get("setup_idx", "")),
        ("initial joint", result.get("initial_joint_wsr", result.get("initial_wsr", np.nan))),
        ("final joint", result.get("final_wsr", np.nan)),
        ("joint gain", result.get("final_wsr", np.nan) - result.get("initial_joint_wsr", result.get("initial_wsr", np.nan))),
        ("initial comm", result.get("initial_comm_wsr", np.nan)),
        ("final comm", result.get("final_comm_wsr", np.nan)),
        ("initial sensing", result.get("initial_sensing_wsr", np.nan)),
        ("final sensing", result.get("final_sensing_wsr", np.nan)),
        ("iterations", len(result.get("wsr_history", []))),
        ("monotonic", result.get("monotonic", False)),
        ("sum P_prime", result.get("P_prime_sum", np.nan)),
        ("mean P_prime", result.get("P_prime_mean", np.nan)),
    ]
    print_key_value_table(label, rows)




def pad_with_last_value(histories):
    """Pad 1-D histories to equal length by repeating each history's last value."""
    histories = [np.asarray(h, dtype=float).reshape(-1) for h in histories]
    if not histories:
        return np.zeros((0, 0), dtype=float)
    max_len = max(len(h) for h in histories)
    padded = np.zeros((len(histories), max_len), dtype=float)
    for i, h in enumerate(histories):
        if len(h) == 0:
            padded[i, :] = np.nan
            continue
        padded[i, :len(h)] = h
        padded[i, len(h):] = h[-1]
    return padded

def get_target_weights_from_mat(mat, Tg):
    """Load target weights omega_q from the .mat file when available; otherwise use ones."""
    for name in ["omega", "omega_q", "target_weights", "omega_s", "omega_target"]:
        if name in mat:
            omega = np.asarray(mat[name], dtype=float).squeeze().reshape(-1)
            if omega.size == 1:
                return float(omega[0]) * np.ones(Tg, dtype=float)
            if omega.size != Tg:
                raise ValueError(
                    f"{name} has length {omega.size}, but number of sensing targets Tg={Tg}."
                )
            return omega.astype(float)
    return np.ones(Tg, dtype=float)


def run_one_setup_power(
    mat_file,
    Pmax_db,
    model=None,
    train_unfolding=False,
    run_classical=True,
    print_components=True,
    setup_idx=0,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-6,
    verbose=True,
    check_sensing=False,
    print_sensing=True,
    init_delta=0.50,
    init_lambda=0.50,
    print_power_tables=True,
    run_sensing_only=True,
    classical_mode="joint",
    run_deep_unfolding=False,
    w_c_user=None,
    w_s_user=None,
):
    """Run joint communication-sensing FP/QT optimization for one setup and one power.

    Optimized variables:
        P_bar[k]   : communication pilot power
        P_tilda[k] : communication data power
        P_prime[k] : sensing power

    Constraint:
        P_bar[k] + P_tilda[k] + P_prime[k] <= Pmax_lin, for each user k.
    """

    data = load_matlab_direct_values(
        mat_file,
        setup_idx=setup_idx,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
    )

    N = data["N"]
    K = data["K"]
    M = data["M"]
    tau = data["tau"]
    d = data["d"]

    alpha_list = data["alpha_list"]
    A_list = data["A_list"]
    h_bar_list = data["h_bar_list"]
    R_list = data["R_list"]
    Sigma_list = data["Sigma_list"]
    C_h_ref_all = data["C_h_ref_all"]
    C_h_total_all = data["C_h_total_all"]
    Z_list = data["Z_list"]
    alpha_var_matrix_all = data["alpha_var_matrix_all"]
    b_target_all = data["b_target_all"]
    H_direct_iter = data["H_direct_iter"]
    H_true_iter = data["H_true_iter"]
    V_combiner_all = data["V_combiner_all"]
    mat = data["mat"]

    # a_sens, b_sens, n_sens = load_sensing_coefficients_from_mat(
    #     mat_file=mat_file,
    #     setup_idx=setup_idx,
    #     K_use=K,
    # )

    # ------------------------------------------------------------
    # Objective weights
    # If w_c_user / w_s_user are given, use them directly from Python.
    # Otherwise fall back to the .mat file.
    # ------------------------------------------------------------
    if w_c_user is None:
        w_c = float(np.squeeze(mat["w_c"])) if "w_c" in mat else 1.0
    else:
        w_c = float(w_c_user)

    if w_s_user is None:
        w_s = float(np.squeeze(mat["w_s"])) if "w_s" in mat else 1.0
    else:
        w_s = float(w_s_user)
    if verbose:
        print(f"[WEIGHT DEBUG] Using w_c = {w_c:.4f}, w_s = {w_s:.4f}")

    Pmax_lin = db2lin(Pmax_db)
    P_total_max = Pmax_lin

    # ------------------------------------------------------------
    # User-defined joint initialization
    # This is the actual optimization starting point.
    # It must satisfy P_bar + P_tilda + P_prime <= Pmax.
    # ------------------------------------------------------------
    delta0 = float(init_delta)      # sensing fraction
    lambda0 = float(init_lambda)    # pilot fraction inside communication power

    if not (0.0 <= delta0 <= 1.0):
        raise ValueError(f"init_delta must be in [0,1], got {delta0}")

    if not (0.0 <= lambda0 <= 1.0):
        raise ValueError(f"init_lambda must be in [0,1], got {lambda0}")

    P_prime_init = delta0 * Pmax_lin * np.ones(K)

    P_comm_init = (1.0 - delta0) * Pmax_lin

    P_bar_init = lambda0 * P_comm_init * np.ones(K)

    P_tilda_init = (1.0 - lambda0) * P_comm_init * np.ones(K)

    # ------------------------------------------------------------
    # MATLAB-style fixed hardware / estimation reference
    # Constraint uses 10, but fixed hardware/estimation uses 20.
    # ------------------------------------------------------------
    P_comm_ref_hw = 0.5 * Pmax_lin
    P_sense_ref_hw = 0.5 * Pmax_lin

    P_bar_ref = 0.5 * P_comm_ref_hw * np.ones(K)
    P_tilda_ref = 0.5 * P_comm_ref_hw * np.ones(K)
    P_prime_ref = P_sense_ref_hw * np.ones(K)

    P_tot_list = P_bar_ref + P_tilda_ref + P_prime_ref

    # Optional scalar diagnostic.
    # Use MATLAB-style sensing reference for sensing-projection diagnostic.
    P_sense = float(P_prime_ref[0])
    P_comm_max = float(P_comm_ref_hw)

    if classical_mode not in ["joint", "comm_only", "sens_only"]:
        raise ValueError(
            f"classical_mode must be 'joint', 'comm_only', or 'sens_only', got {classical_mode}"
        )
    if verbose or print_power_tables:
        print_power_split_table(
            Pmax_db=Pmax_db,
            Pmax_lin=Pmax_lin,
            P_bar_init=P_bar_init,
            P_tilda_init=P_tilda_init,
            P_prime_init=P_prime_init,
            P_total_max=P_total_max,
        )
        # target_weights is loaded after sensing coefficients are built.

    Theta_list = build_theta_list_for_power(
        A_list=A_list,
        C_h_total_all=C_h_total_all,
        alpha_list=alpha_list,
        P_tot_list=P_tot_list,
        sigma_w2=sigma_w2,
    )

    sigma_d2 = alpha_list * (1.0 - alpha_list) * P_tot_list
    w = np.ones(K)
    W_list = build_W_list_MR(M, K, N)

    # ------------------------------------------------------------
    # Build sensing coefficients from Python geometry formula
    # instead of loading MATLAB sensing coefficients.
    # ------------------------------------------------------------
    if b_target_all is None:
        raise KeyError(
            "Cannot build sensing coefficients from geometry because b_target_all is missing."
        )

    if alpha_var_matrix_all is None:
        raise KeyError(
            "Cannot build sensing coefficients from geometry because alpha_var_matrix_all is missing."
        )

    a_sens, b_sens, n_sens, sensing_coeff_parts = build_sensing_coefficients_matlab_matching(
        T=tau,
        alpha_list=alpha_list,
        sigma_d2=sigma_d2,
        A_list=A_list,
        Theta_list=Theta_list,
        b_target_all=b_target_all,
        alpha_var_matrix_all=alpha_var_matrix_all,
        V_combiner_all=V_combiner_all,
        H_direct_iter=H_direct_iter,
        H_true_iter=H_true_iter,
        sigma_w2=sigma_w2,
        return_parts=True,
    )

    sensing_coeff_source = "python_matlab_matching_mc"

    target_weights = get_target_weights_from_mat(mat, a_sens.shape[0])
    if verbose:
        print("sensing_coeff_source =", sensing_coeff_source)
        print("sum a_sens per target =", np.sum(a_sens, axis=1))
        print("sum b_sens per target =", np.sum(b_sens, axis=1))
        print("n_sens =", n_sens)
    if verbose or print_power_tables:
        print_weight_table(w_c=w_c, w_s=w_s, target_weights=target_weights)
    # Optional diagnostic only: this is a scalar pre-optimization sensing projection.
    # The optimized sensing result later comes from P_prime_opt and coefficient model.
    if check_sensing and b_target_all is not None and print_sensing:
        if V_combiner_all is not None:
            sensing_diag = compute_sensing_projection_components_with_Viter(
                P_sense=P_sense,
                T=tau,
                alpha_list=alpha_list,
                sigma_d2=sigma_d2,
                A_list=A_list,
                Theta_list=Theta_list,
                b_target_all=b_target_all,
                C_h_ref_all=C_h_ref_all,
                sigma_w2=sigma_w2,
                H_direct_iter=H_direct_iter,
                H_true_iter=H_true_iter,
                C_h_dir_all=Sigma_list,
                C_h_total_all=C_h_total_all,
                alpha_var_matrix_all=alpha_var_matrix_all,
                V_combiner_all=V_combiner_all,
            )
        else:
            sensing_diag = compute_sensing_projection_components(
                P_sense=P_sense,
                T=tau,
                alpha_list=alpha_list,
                sigma_d2=sigma_d2,
                A_list=A_list,
                Theta_list=Theta_list,
                b_target_all=b_target_all,
                C_h_ref_all=C_h_ref_all,
                sigma_w2=sigma_w2,
                H_direct_iter=H_direct_iter,
                H_true_iter=H_true_iter,
                C_h_dir_all=Sigma_list,
                C_h_total_all=C_h_total_all,
                alpha_var_matrix_all=alpha_var_matrix_all,
            )
        print_sensing_projection_components(
            label=f"INITIAL scalar sensing diagnostic @ Pmax={Pmax_db} dB, setup={setup_idx}, P_sense={P_sense:.4e}",
            sensing=sensing_diag,
        )
    elif check_sensing and b_target_all is None:
        print("[SENSING] b_target_all not found; skipped optional scalar sensing diagnostic.")

    R_tilda_all = build_R_tilda_all(R_list, h_bar_list)

    C_yy_all = build_C_yy_all(
        alpha_list=alpha_list,
        P_bar_ref=P_bar_ref,
        tau=tau,
        A_list=A_list,
        R_tilda_all=R_tilda_all,
        C_h_total_all=C_h_total_all,
        sigma_d2=sigma_d2,
        sigma_w2=sigma_w2,
        Theta_list=Theta_list,
    )

    G_list = build_G_list(
        alpha_list=alpha_list,
        P_bar_ref=P_bar_ref,
        tau=tau,
        A_list=A_list,
        R_tilda_all=R_tilda_all,
        C_yy_all=C_yy_all,
    )

    U_list, V_list, U_q_list, V_q_list, _, D_Th_list, D_ADC_list = build_all_matrices(
        A_list,
        G_list,
        W_list,
        R_list,
        h_bar_list,
        Theta_list,
    )

    # Build g_lik.
    g_lik = np.zeros((M, K, K))
    for l in range(M):
        for i in range(K):
            for k in range(K):
                g_lik[l, i, k] = compute_g_lik(
                    l, i, k,
                    U_list,
                    V_list,
                    Sigma_list,
                    R_list,
                    C_h_total_all,
                    C_h_ref_all,
                    h_bar_list,
                    T=tau,
                )

    # Kappas needed for initial objective and optional component tables.
    (
        kappa_S,
        kappa_V1, kappa_V0,
        kappa_K1, kappa_K0,
        kappa_DAC1, kappa_DAC0,
        kappa_Th1, kappa_ADC1,
        kappa_M1, kappa_M0,
    ) = build_all_kappas(
        K, alpha_list, tau, d,
        U_list, V_list,
        U_q_list, V_q_list,
        Sigma_list,
        R_list, h_bar_list,
        C_h_ref_all, C_h_total_all,
        sigma_d2, sigma_w2,
        D_Th_list, D_ADC_list,
        A_list, Theta_list,
        Z_list,
        g_lik,
    )

    kappa_Th0, kappa_ADC0 = build_kappa_Th0_ADC0(
        K=K,
        tau=tau,
        d=d,
        sigma_w2=sigma_w2,
        sigma_d2=sigma_d2,
        G_list=G_list,
        W_list=W_list,
        A_list=A_list,
        Theta_list=Theta_list,
        C_h_total_all=C_h_total_all,
    )

    # P_bar_before = P_bar_init.copy()
    # P_tilda_before = P_tilda_init.copy()
    # P_prime_before = P_prime_init.copy()
    
    P_bar_before, P_tilda_before, P_prime_before = project_joint_three_power_blocks(
        P_bar_raw=P_bar_init,
        P_tilda_raw=P_tilda_init,
        P_prime_raw=P_prime_init,
        P_total_max=P_total_max,
        eps=1e-12,
    )

    P_bar_init = P_bar_before.copy()
    P_tilda_init = P_tilda_before.copy()
    P_prime_init = P_prime_before.copy()
    
    # P_bar_before = P_bar_ref.copy()
    # P_tilda_before = P_tilda_ref.copy()
    # P_prime_before = P_prime_ref.copy()
    P_th_before = P_bar_before * kappa_Th1 + kappa_Th0
    P_adc_before = P_bar_before * kappa_ADC1 + kappa_ADC0

    initial_comm_wsr = float(compute_WSR_exact(
        K, w,
        kappa_S,
        kappa_V1, kappa_K1,
        kappa_V0, kappa_K0,
        kappa_DAC1, kappa_DAC0,
        kappa_M1, kappa_M0,
        kappa_Th1, kappa_ADC1,
        P_bar_before,
        P_tilda_before,
        P_th_before,
        P_adc_before,
        d,
        tau,
    ))

    initial_sensing_wsr = compute_sensing_wsr(
        P_prime=P_prime_before,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
    )

    initial_joint_wsr = float(w_c * initial_comm_wsr + initial_sensing_wsr)
    initial_wsr = initial_joint_wsr

    initial_sensing_components = compute_sensing_components_from_coeffs(
        P_prime=P_prime_before,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
        sensing_coeff_parts=sensing_coeff_parts,
    )
    comm_only_result = None
    sensing_only_result = None

    if classical_mode == "comm_only":
        comm_only_result = run_communication_only_optimisation(
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
            P_prime_fixed=P_prime_init,
            P_total_max=P_total_max,
            max_iters=max_iters,
            epsilon=epsilon,
            eps_power=1e-12,
            verbose=False,
        )

        print("\n[COMMUNICATION-ONLY CLASSICAL RESULT]")
        print(f"Initial communication WSR = {comm_only_result['initial_comm_wsr']:.6f}")
        print(f"Final communication WSR   = {comm_only_result['final_comm_wsr']:.6f}")
        print(f"Iterations                = {comm_only_result['iterations']}")
        print(f"Monotonic                 = {comm_only_result['monotonic']}")

        print_power_allocation_table(
            label=f"Communication-Only Optimized Power Allocation @ {Pmax_db} dB",
            P_bar=comm_only_result["P_bar_opt"],
            P_tilda=comm_only_result["P_tilda_opt"],
            P_prime=comm_only_result["P_prime_opt"],
            P_total_max=P_total_max,
        )

        return {
            "mode": "comm_only",
            "Pmax_db": Pmax_db,
            "setup_idx": setup_idx,
            "comm_only_result": comm_only_result,
            "P_bar_initial": P_bar_init,
            "P_tilda_initial": P_tilda_init,
            "P_prime_initial": P_prime_init,
            "P_bar_opt": comm_only_result["P_bar_opt"],
            "P_tilda_opt": comm_only_result["P_tilda_opt"],
            "P_prime_opt": comm_only_result["P_prime_opt"],
            "initial_comm_wsr": comm_only_result["initial_comm_wsr"],
            "final_comm_wsr": comm_only_result["final_comm_wsr"],
            "comm_history": comm_only_result["comm_history"],
            "full_comm_history": comm_only_result["full_comm_history"],
            "P_total_max": P_total_max,
            "Pmax_lin": Pmax_lin,
        }

    if classical_mode == "sens_only":
        sensing_only_result = run_sensing_only_optimisation(
            P_prime_init=P_prime_init,
            P_bar_fixed=P_bar_init,
            P_tilda_fixed=P_tilda_init,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_s=w_s,
            P_total_max=P_total_max,
            max_iters=max_iters,
            epsilon=epsilon,
            eps_power=1e-12,
            verbose=False,
        )

        print("\n[SENSING-ONLY CLASSICAL RESULT]")
        print(f"Initial sensing WSR = {sensing_only_result['initial_sensing_wsr']:.6f}")
        print(f"Final sensing WSR   = {sensing_only_result['final_sensing_wsr']:.6f}")
        print(f"Iterations          = {sensing_only_result['iterations']}")
        print(f"Monotonic           = {sensing_only_result['monotonic']}")

        print_sensing_before_after_table(
            label=f"Sensing-Only Optimization: Before vs After @ Pmax={Pmax_db} dB, setup={setup_idx}",
            sensing_before=sensing_only_result["sensing_initial"],
            sensing_after=sensing_only_result["sensing_final"],
            target_weights=target_weights,
            w_s=w_s,
        )

        print_power_allocation_table(
            label=f"Sensing-Only Optimized Power Allocation @ {Pmax_db} dB",
            P_bar=P_bar_init,
            P_tilda=P_tilda_init,
            P_prime=sensing_only_result["P_prime_opt"],
            P_total_max=P_total_max,
        )

        return {
            "mode": "sens_only",
            "Pmax_db": Pmax_db,
            "setup_idx": setup_idx,
            "sensing_only_result": sensing_only_result,
            "P_bar_initial": P_bar_init,
            "P_tilda_initial": P_tilda_init,
            "P_prime_initial": P_prime_init,
            "P_bar_opt": P_bar_init,
            "P_tilda_opt": P_tilda_init,
            "P_prime_opt": sensing_only_result["P_prime_opt"],
            "initial_sensing_wsr": sensing_only_result["initial_sensing_wsr"],
            "final_sensing_wsr": sensing_only_result["final_sensing_wsr"],
            "sensing_history": sensing_only_result["sensing_history"],
            "full_sensing_history": sensing_only_result["full_sensing_history"],
            "sensing_initial": sensing_only_result["sensing_initial"],
            "sensing_final": sensing_only_result["sensing_final"],
            "P_total_max": P_total_max,
            "Pmax_lin": Pmax_lin,
        }
    # ============
    # Deep unfolding model: joint communication + sensing
    # ============
    if run_deep_unfolding:
        device = next(model.parameters()).device if model is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if model is None:
            model = JointCommSensingUnfoldingPGD(
                K=K,
                num_layers=20,
                num_pgd_steps=20,
                init_step_bar=0.1,
                init_step_tilda=0.1,
                init_step_prime=0.1,
                enforce_full_power=False,
            ).to(device).double()
        else:
            model = model.to(device).double()

        P_bar_init_t = to_torch_1d(P_bar_init, device)
        P_tilda_init_t = to_torch_1d(P_tilda_init, device)
        P_prime_init_t = to_torch_1d(P_prime_init, device)
        w_t = to_torch_1d(w, device)
        Pmax_t = scalar_to_torch(P_total_max, device)

        ctx = torch.enable_grad() if train_unfolding else torch.no_grad()
        model.train(mode=bool(train_unfolding))

        with ctx:
            du_out = model(
                P_bar_init=P_bar_init_t,
                P_tilda_init=P_tilda_init_t,
                P_prime_init=P_prime_init_t,
                w=w_t,
                P_total_max=Pmax_t,
                d=d,
                tau=tau,
                kappa_S=to_torch_1d(kappa_S, device),
                kappa_V1=to_torch_1d(kappa_V1, device),
                kappa_K1=to_torch_1d(kappa_K1, device),
                kappa_V0=to_torch_1d(kappa_V0, device),
                kappa_K0=to_torch_1d(kappa_K0, device),
                kappa_DAC1=to_torch_1d(kappa_DAC1, device),
                kappa_DAC0=to_torch_1d(kappa_DAC0, device),
                kappa_Th1=to_torch_1d(kappa_Th1, device),
                kappa_ADC1=to_torch_1d(kappa_ADC1, device),
                kappa_Th0=to_torch_1d(kappa_Th0, device),
                kappa_ADC0=to_torch_1d(kappa_ADC0, device),
                kappa_M1=to_torch_2d(kappa_M1, device),
                kappa_M0=to_torch_2d(kappa_M0, device),
                a_sens=to_torch_2d(a_sens, device),
                b_sens=to_torch_2d(b_sens, device),
                n_sens=to_torch_1d(n_sens, device),
                target_weights=to_torch_1d(target_weights, device),
                w_c=w_c,
                w_s=w_s,
                return_history=True,
                debug_trace=True,
                debug_layer=0,
            )

        deep_final_joint_wsr_tensor = du_out["joint_obj"].mean()
        deep_final_comm_wsr_tensor = du_out["comm_wsr"].mean()
        deep_final_sensing_raw_tensor = du_out["sensing_wsr"].mean()
        deep_final_sensing_wsr_tensor = float(w_s) * deep_final_sensing_raw_tensor

        deep_final_joint_wsr = deep_final_joint_wsr_tensor.detach().cpu().item()
        deep_final_comm_wsr = deep_final_comm_wsr_tensor.detach().cpu().item()
        deep_final_sensing_wsr = deep_final_sensing_wsr_tensor.detach().cpu().item()
        deep_final_sensing_raw = deep_final_sensing_raw_tensor.detach().cpu().item()

        deep_joint_history_tensor = du_out["joint_history"].mean(dim=1)
        deep_comm_history_tensor = du_out["comm_history"].mean(dim=1)
        deep_sensing_raw_history_tensor = du_out["sensing_history"].mean(dim=1)
        deep_sensing_history_tensor = float(w_s) * deep_sensing_raw_history_tensor

        deep_joint_history = deep_joint_history_tensor.detach().cpu().numpy()
        deep_comm_history = deep_comm_history_tensor.detach().cpu().numpy()
        deep_sensing_history = deep_sensing_history_tensor.detach().cpu().numpy()

        deep_P_bar = du_out["P_bar"].detach().cpu().numpy().reshape(-1)
        deep_P_tilda = du_out["P_tilda"].detach().cpu().numpy().reshape(-1)
        deep_P_prime = du_out["P_prime"].detach().cpu().numpy().reshape(-1)
        deep_comm_sinr = du_out["comm_sinr"].detach().cpu().numpy().reshape(-1)
        deep_sensing_sinr = du_out["sensing_sinr"].detach().cpu().numpy().reshape(-1)

        deep_sensing_components = compute_sensing_components_from_coeffs(
            P_prime=deep_P_prime,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_s=w_s,
            sensing_coeff_parts=sensing_coeff_parts,
        )

        if verbose:
            print(
                f"[Joint Deep Unfolding] Pmax={Pmax_db:>6.1f} dB | "
                f"joint={deep_final_joint_wsr:.6f} | "
                f"comm={deep_final_comm_wsr:.6f} | "
                f"sensing={deep_final_sensing_wsr:.6f}"
            )

        if not run_classical:
            P_th_deep = deep_P_bar * kappa_Th1 + kappa_Th0
            P_adc_deep = deep_P_bar * kappa_ADC1 + kappa_ADC0
            return {
                "Pmax_db": Pmax_db,
                "setup_idx": setup_idx,
                "initial_wsr": float(initial_joint_wsr),
                "initial_joint_wsr": float(initial_joint_wsr),
                "initial_comm_wsr": float(initial_comm_wsr),
                "initial_sensing_wsr": float(initial_sensing_wsr),
                "final_wsr": np.nan,
                "wsr_history": np.array([]),
                "full_joint_history": np.asarray([initial_joint_wsr]),
                "P_bar_opt": None,
                "P_tilda_opt": None,
                "P_prime_opt": None,
                "gamma_opt": None,
                "P_bar_initial": P_bar_init,
                "P_tilda_initial": P_tilda_init,
                "P_prime_initial": P_prime_init,
                "monotonic": None,
                "comm_only_result": comm_only_result,
                "comm_history": np.array([]),
                "sensing_history": np.array([]),
                "joint_history": np.array([]),
                "final_comm_wsr": np.nan,
                "final_sensing_wsr": np.nan,
                "sensing_projection": None,
                "initial_sensing_components": initial_sensing_components,
                "sensing_only_result": None,
                "sensing_coeff_source": sensing_coeff_source,
                "target_weights": target_weights,
                "a_sens": a_sens,
                "b_sens": b_sens,
                "n_sens": n_sens,
                "w_c": w_c,
                "w_s": w_s,
                "Pmax_lin": Pmax_lin,
                "P_total_max": P_total_max,
                "K": K,
                "M": M,
                "N": N,
                "tau": tau,
                "d": d,
                "comm_kappas": {
                    "kappa_S": kappa_S, "kappa_V1": kappa_V1, "kappa_V0": kappa_V0,
                    "kappa_K1": kappa_K1, "kappa_K0": kappa_K0,
                    "kappa_DAC1": kappa_DAC1, "kappa_DAC0": kappa_DAC0,
                    "kappa_Th1": kappa_Th1, "kappa_ADC1": kappa_ADC1,
                    "kappa_M1": kappa_M1, "kappa_M0": kappa_M0,
                    "kappa_Th0": kappa_Th0, "kappa_ADC0": kappa_ADC0,
                },
                "P_th_initial": P_th_before,
                "P_adc_initial": P_adc_before,
                "P_th_opt": P_th_deep,
                "P_adc_opt": P_adc_deep,
                "deep_final_joint_wsr": deep_final_joint_wsr,
                "deep_final_joint_wsr_tensor": deep_final_joint_wsr_tensor,
                "deep_final_comm_wsr": deep_final_comm_wsr,
                "deep_final_sensing_wsr": deep_final_sensing_wsr,
                "deep_final_sensing_raw": deep_final_sensing_raw,
                "deep_joint_history": deep_joint_history,
                "deep_joint_history_tensor": deep_joint_history_tensor,
                "deep_comm_history": deep_comm_history,
                "deep_sensing_history": deep_sensing_history,
                "deep_P_bar": deep_P_bar,
                "deep_P_tilda": deep_P_tilda,
                "deep_P_prime": deep_P_prime,
                "deep_comm_sinr": deep_comm_sinr,
                "deep_sensing_sinr": deep_sensing_sinr,
                "deep_sensing_components": deep_sensing_components,
            }
    else:
        deep_final_joint_wsr = np.nan
        deep_final_comm_wsr = np.nan
        deep_final_sensing_wsr = np.nan
        deep_final_sensing_raw = np.nan

        deep_final_joint_wsr_tensor = None
        deep_joint_history_tensor = None

        deep_joint_history = np.array([])
        deep_comm_history = np.array([])
        deep_sensing_history = np.array([])

        deep_P_bar = np.full(K, np.nan)
        deep_P_tilda = np.full(K, np.nan)
        deep_P_prime = np.full(K, np.nan)

        deep_comm_sinr = np.full(K, np.nan)
        deep_sensing_sinr = np.array([])
        deep_sensing_components = None
        sensing_only_result = None
    if run_sensing_only:
        sensing_only_result = run_sensing_only_optimisation(
            P_prime_init=P_prime_init,
            P_bar_fixed=P_bar_init,
            P_tilda_fixed=P_tilda_init,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_s=w_s,
            P_total_max=P_total_max,
            max_iters=max_iters,
            epsilon=epsilon,
            eps_power=1e-12,
            verbose=False,
        )

    if print_components:
        print_power_components(
            label=f"BEFORE optimization @ {Pmax_db} dB",
            K=K,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1,
            kappa_DAC0=kappa_DAC0,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            kappa_Th1=kappa_Th1,
            kappa_ADC1=kappa_ADC1,
            P_bar=P_bar_before,
            P_tilda=P_tilda_before,
            P_th=P_th_before,
            P_adc=P_adc_before,
        )

    joint_result = run_joint_wsr_optimisation(
        K=K,
        w=w,
        alpha_list=alpha_list,
        tau=tau,
        d=d,
        sigma_w2=sigma_w2,
        sigma_d2=sigma_d2,
        U_list=U_list,
        V_list=V_list,
        Sigma_list=Sigma_list,
        R_list=R_list,
        h_bar_list=h_bar_list,
        C_h_ref_all=C_h_ref_all,
        C_h_total_all=C_h_total_all,
        D_Th_list=D_Th_list,
        D_ADC_list=D_ADC_list,
        A_list=A_list,
        Theta_list=Theta_list,
        Z_list=Z_list,
        G_list=G_list,
        W_list=W_list,
        T=tau,
        U_q_list=U_q_list,
        V_q_list=V_q_list,
        g_lik=g_lik,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_c=w_c,
        w_s=w_s,
        P_bar_init=P_bar_init,
        P_tilda_init=P_tilda_init,
        P_prime_init=P_prime_init,
        P_total_max=P_total_max,
        eps_power=1e-12,
        max_iters=max_iters,
        epsilon=epsilon,
        verbose=False,
    )
    convergence_iter = joint_result["convergence_iter"]
    converged = joint_result["converged"]
    elapsed_time_sec = joint_result["elapsed_time_sec"]
    final_delta = joint_result["final_delta"]
    wsr_history = np.asarray(joint_result["joint_history"], dtype=float)
    comm_history = np.asarray(joint_result["comm_history"], dtype=float)
    sensing_history = np.asarray(joint_result["sensing_history"], dtype=float)

    P_bar_opt = joint_result["P_bar_opt"]
    P_tilda_opt = joint_result["P_tilda_opt"]
    P_prime_opt = joint_result["P_prime_opt"]
    gamma_opt = joint_result["gamma_opt"]
    sensing_projection = joint_result["sensing_final"]

    sensing_projection = compute_sensing_components_from_coeffs(
        P_prime=P_prime_opt,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
        sensing_coeff_parts=sensing_coeff_parts,
    )
    P_th_after = P_bar_opt * kappa_Th1 + kappa_Th0
    P_adc_after = P_bar_opt * kappa_ADC1 + kappa_ADC0

    if print_components:
        print_power_components(
            label=f"AFTER optimization @ {Pmax_db} dB",
            K=K,
            kappa_S=kappa_S,
            kappa_V1=kappa_V1,
            kappa_K1=kappa_K1,
            kappa_V0=kappa_V0,
            kappa_K0=kappa_K0,
            kappa_DAC1=kappa_DAC1,
            kappa_DAC0=kappa_DAC0,
            kappa_M1=kappa_M1,
            kappa_M0=kappa_M0,
            kappa_Th1=kappa_Th1,
            kappa_ADC1=kappa_ADC1,
            P_bar=P_bar_opt,
            P_tilda=P_tilda_opt,
            P_th=P_th_after,
            P_adc=P_adc_after,
        )

    final_wsr = float(wsr_history[-1]) if len(wsr_history) else np.nan
    full_joint_history = np.concatenate([[initial_joint_wsr], wsr_history]) if len(wsr_history) else np.asarray([initial_joint_wsr])
    monotonic = bool(np.all(np.diff(full_joint_history) >= -1e-10)) if len(full_joint_history) > 1 else True

    result = {
        "Pmax_db": Pmax_db,
        "setup_idx": setup_idx,
        "initial_wsr": float(initial_wsr),
        "final_wsr": final_wsr,
        "wsr_history": wsr_history,
        "full_joint_history": full_joint_history,
        "P_bar_opt": P_bar_opt,
        "P_tilda_opt": P_tilda_opt,
        "P_prime_opt": P_prime_opt,
        "gamma_opt": gamma_opt,
        "P_bar_initial": P_bar_init,
        "P_tilda_initial": P_tilda_init,
        "P_prime_initial": P_prime_init,
        "monotonic": monotonic,
        "comm_history": comm_history,
        "sensing_history": sensing_history,
        "joint_history": wsr_history,
        "initial_comm_wsr": initial_comm_wsr,
        "initial_sensing_wsr": initial_sensing_wsr,
        "initial_joint_wsr": initial_joint_wsr,
        "final_comm_wsr": float(comm_history[-1]) if len(comm_history) else np.nan,
        "final_sensing_wsr": float(sensing_history[-1]) if len(sensing_history) else np.nan,
        "P_prime_sum": float(np.sum(P_prime_opt)),
        "P_prime_mean": float(np.mean(P_prime_opt)),
        "P_prime_min": float(np.min(P_prime_opt)),
        "P_prime_max": float(np.max(P_prime_opt)),
        "init_delta": delta0,
        "init_lambda": lambda0,
        "sensing_fraction": delta0,
        "sensing_projection": sensing_projection,
        "initial_sensing_components": initial_sensing_components,
        "sensing_only_result": sensing_only_result,
        "sensing_coeff_source": sensing_coeff_source,
        "target_weights": target_weights,
        "a_sens": a_sens,
        "b_sens": b_sens,
        "n_sens": n_sens,
        "w_c": w_c,
        "w_s": w_s,
        "Pmax_lin": Pmax_lin,
        "P_total_max": P_total_max,
        "K": K,
        "M": M,
        "N": N,
        "convergence_iter": convergence_iter,
        "converged": converged,
        "elapsed_time_sec": elapsed_time_sec,
        "final_delta": final_delta,
        "comm_kappas": {
            "kappa_S": kappa_S,
            "kappa_V1": kappa_V1,
            "kappa_V0": kappa_V0,
            "kappa_K1": kappa_K1,
            "kappa_K0": kappa_K0,
            "kappa_DAC1": kappa_DAC1,
            "kappa_DAC0": kappa_DAC0,
            "kappa_Th1": kappa_Th1,
            "kappa_ADC1": kappa_ADC1,
            "kappa_M1": kappa_M1,
            "kappa_M0": kappa_M0,
            "kappa_Th0": kappa_Th0,
            "kappa_ADC0": kappa_ADC0,
        },
        "P_th_initial": P_th_before,
        "P_adc_initial": P_adc_before,
        "P_th_opt": P_th_after,
        "P_adc_opt": P_adc_after,
        "tau": tau,
        "d": d,

        # Deep unfolding results
        "deep_final_joint_wsr": deep_final_joint_wsr,
        "deep_final_joint_wsr_tensor": deep_final_joint_wsr_tensor,
        "deep_final_comm_wsr": deep_final_comm_wsr,
        "deep_final_sensing_wsr": deep_final_sensing_wsr,
        "deep_final_sensing_raw": deep_final_sensing_raw,
        "deep_joint_history": deep_joint_history,
        "deep_joint_history_tensor": deep_joint_history_tensor,
        "deep_comm_history": deep_comm_history,
        "deep_sensing_history": deep_sensing_history,
        "deep_P_bar": deep_P_bar,
        "deep_P_tilda": deep_P_tilda,
        "deep_P_prime": deep_P_prime,
        "deep_comm_sinr": deep_comm_sinr,
        "deep_sensing_sinr": deep_sensing_sinr,
        "deep_sensing_components": deep_sensing_components,
    }

    if verbose:
        print_joint_summary_table("Joint Optimization Summary", result)

    if print_power_tables:
        print_power_allocation_table(
            label=f"Optimized Per-User Power Allocation @ {Pmax_db} dB",
            P_bar=P_bar_opt,
            P_tilda=P_tilda_opt,
            P_prime=P_prime_opt,
            P_total_max=P_total_max,
        )

    if print_sensing:
        if sensing_only_result is not None:
            print_sensing_before_after_table(
                label=f"Sensing-Only Optimization: Before vs After @ Pmax={Pmax_db} dB, setup={setup_idx}",
                sensing_before=initial_sensing_components,
                sensing_after=sensing_only_result["sensing_final"],
                target_weights=target_weights,
                w_s=w_s,
            )
            print_power_allocation_table(
                label=f"Sensing-Only Optimized Power Allocation @ {Pmax_db} dB",
                P_bar=P_bar_before,
                P_tilda=P_tilda_before,
                P_prime=sensing_only_result["P_prime_opt"],
                P_total_max=P_total_max,
            )

        print_sensing_before_after_table(
            label=f"Joint Optimization: Sensing Before vs After @ Pmax={Pmax_db} dB, setup={setup_idx}",
            sensing_before=initial_sensing_components,
            sensing_after=sensing_projection,
            target_weights=target_weights,
            w_s=w_s,
        )

    return result

def make_shuffled_batches(train_setups, batch_size, shuffle_batches=True, rng=None):
    """Create epoch-wise batches. If shuffle_batches=True, batches change every epoch."""
    setups = np.asarray(train_setups, dtype=int).copy()
    if shuffle_batches:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(setups)
    return [setups[i:i + batch_size] for i in range(0, len(setups), batch_size)]


def train_joint_unfolding_model(
    mat_file,
    model,
    optimizer,
    train_setups,
    train_power_list,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    num_epochs=15,
    batch_size=10,
    init_delta=0.50,
    init_lambda=0.50,
    shuffle_batches=True,
    seed=42,
):
    """Train DU step sizes correctly, with optional epoch-wise shuffled batches.

    Notes:
      - Uses only `train_setups`; testing must use disjoint `test_setups`.
      - Computes logs separately for each Pmax in `train_power_list`.
      - Performs one optimizer step per (Pmax, batch), as in the original design.
    """
    train_setups = np.asarray(train_setups, dtype=int)
    train_power_list = np.asarray(train_power_list, dtype=float)
    rng = np.random.default_rng(seed)

    train_joint_log = {float(Pdb): [] for Pdb in train_power_list}
    train_layer_joint_log = {float(Pdb): [] for Pdb in train_power_list}

    print("\n" + "=" * 90)
    print(f"Training Joint Deep Unfolding for Pmax values = {train_power_list}")
    print(f"shuffle_batches={shuffle_batches}, batch_size={batch_size}, seed={seed}")
    print("=" * 90)

    for epoch in range(num_epochs):
        train_batches = make_shuffled_batches(
            train_setups=train_setups,
            batch_size=batch_size,
            shuffle_batches=shuffle_batches,
            rng=rng,
        )

        epoch_power_means = []

        for Pdb in train_power_list:
            Pdb = float(Pdb)
            power_batch_final_joints = []
            power_batch_histories = []

            for batch_id, setup_batch in enumerate(train_batches):
                batch_mean_wsr, batch_layer_history_mean = train_one_batch(
                    mat_file=mat_file,
                    model=model,
                    optimizer=optimizer,
                    setup_batch=setup_batch,
                    Pmax_db=Pdb,
                    K_use=K_use,
                    M_use=M_use,
                    N_use=N_use,
                    sigma_w2=sigma_w2,
                    init_delta=init_delta,
                    init_lambda=init_lambda,
                )

                power_batch_final_joints.append(batch_mean_wsr)
                power_batch_histories.append(batch_layer_history_mean)

                setup_list_str = ",".join(map(str, setup_batch.tolist()))
                print(
                    f"P={Pdb:>6.1f} dB | "
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Batch {batch_id + 1}/{len(train_batches)} | "
                    f"Setups [{setup_list_str}] | "
                    f"Batch mean final Joint DU = {batch_mean_wsr:.6f}"
                )

            power_epoch_mean = float(np.mean(power_batch_final_joints))
            power_epoch_history = np.mean(np.stack(power_batch_histories, axis=0), axis=0)

            train_joint_log[Pdb].append(power_epoch_mean)
            train_layer_joint_log[Pdb].append(power_epoch_history)
            epoch_power_means.append(power_epoch_mean)

            print(
                f"[Epoch Power Summary] Epoch {epoch + 1}/{num_epochs} | "
                f"P={Pdb:>6.1f} dB | "
                f"Mean final Joint DU over {len(train_batches)} batches = {power_epoch_mean:.6f}"
            )

        print(
            f"[Epoch Summary] Epoch {epoch + 1}/{num_epochs} | "
            f"Mean over powers = {float(np.mean(epoch_power_means)):.6f}"
        )

    return {
        "train_joint_log": train_joint_log,
        "train_layer_joint_log": train_layer_joint_log,
    }


def get_learned_step_sizes(model):
    """Return learned PGD step-size arrays for Pbar, Ptilda, and Pprime."""
    with torch.no_grad():
        return {
            "Pbar": model.step_Pbar.detach().cpu().numpy().reshape(-1),
            "Ptilda": model.step_Ptilda.detach().cpu().numpy().reshape(-1),
            "Pprime": model.step_Pprime.detach().cpu().numpy().reshape(-1),
        }


def print_learned_step_sizes(model, max_rows=None):
    """Print layer-wise learned step sizes and summary statistics."""
    steps = get_learned_step_sizes(model)
    L = len(steps["Pbar"])
    nshow = L if max_rows is None else min(L, max_rows)

    print("\n" + "=" * 90)
    print("LEARNED STEP SIZES")
    print("=" * 90)
    print(f"{'layer':>5} | {'step_Pbar':>13} | {'step_Ptilda':>13} | {'step_Pprime':>13}")
    print("-" * 54)
    for i in range(nshow):
        print(
            f"{i:5d} | "
            f"{steps['Pbar'][i]:13.6e} | "
            f"{steps['Ptilda'][i]:13.6e} | "
            f"{steps['Pprime'][i]:13.6e}"
        )
    if nshow < L:
        print(f"... {L - nshow} layers omitted")
    print("-" * 54)
    for name, arr in steps.items():
        print(
            f"{name:<7} mean={np.mean(arr):.6e}, "
            f"min={np.min(arr):.6e}, max={np.max(arr):.6e}"
        )
    return steps


def plot_classical_du_convergence(joint_mean_history, du_mean_history, Pdb, out_dir="."):
    """Plot classical convergence and DU layer trajectory in the same figure."""
    os.makedirs(out_dir, exist_ok=True)
    classical = np.asarray(joint_mean_history, dtype=float).reshape(-1)
    du = np.asarray(du_mean_history, dtype=float).reshape(-1)

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(np.arange(len(classical)), classical, marker="o", linewidth=2, label="Classical FP/QT")
    plt.plot(np.arange(len(du)), du, marker="s", linewidth=2, label="Deep Unfolding")
    plt.xlabel("Classical iteration / DU layer")
    plt.ylabel("Mean joint objective")
    plt.title(f"Convergence: Classical vs Deep Unfolding @ {Pdb:g} dB")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"convergence_classical_vs_du_{Pdb:g}dB.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_power_sweep_initial_classical_du(power_list, initial_joint, classical_joint, du_joint, out_dir="."):
    """Plot EPA before optimization, classical optimization, and DU optimization vs power."""
    os.makedirs(out_dir, exist_ok=True)
    power_list = np.asarray(power_list, dtype=float).reshape(-1)
    initial_joint = np.asarray(initial_joint, dtype=float).reshape(-1)
    classical_joint = np.asarray(classical_joint, dtype=float).reshape(-1)
    du_joint = np.asarray(du_joint, dtype=float).reshape(-1)

    plt.figure(figsize=(8, 5.6))
    plt.plot(power_list, initial_joint, "--s", linewidth=2, label="EPA before optimization")
    plt.plot(power_list, classical_joint, "-o", linewidth=2, label="Classical FP/QT optimization")
    plt.plot(power_list, du_joint, "-^", linewidth=2, label="Deep unfolding optimization")
    plt.xlabel("Transmit power $P_{max}$ (dB)")
    plt.ylabel("Mean joint objective")
    plt.title("Power sweep: EPA vs Classical vs Deep Unfolding")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "power_sweep_epa_classical_du.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def _vec_to_str(x, precision=6):
    x = np.asarray(x, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.{precision}e}" for v in x) + "]"


# def export_debug_trace_to_word(debug_trace, config, out_path="pgd_debug_trace.docx"):
#     """Export one-layer, k-PGD-step power trace to Word.

#     Expected debug_trace item format:
#       {
#         'layer': int,
#         'pgd_step': int,
#         'step_Pbar': float,
#         'step_Ptilda': float,
#         'step_Pprime': float,
#         'P_bar_before': array,
#         'P_tilda_before': array,
#         'P_prime_before': array,
#         'P_bar_after_update': array,
#         'P_tilda_after_update': array,
#         'P_prime_after_update': array,
#         'P_bar_before_projection': array,
#         'P_tilda_before_projection': array,
#         'P_prime_before_projection': array,
#         'P_bar_after_projection': array,
#         'P_tilda_after_projection': array,
#         'P_prime_after_projection': array,
#       }
#     """
#     from docx import Document
#     from docx.shared import Inches, Pt

#     doc = Document()
#     section = doc.sections[0]
#     section.top_margin = Inches(0.55)
#     section.bottom_margin = Inches(0.55)
#     section.left_margin = Inches(0.55)
#     section.right_margin = Inches(0.55)

#     styles = doc.styles
#     styles["Normal"].font.name = "Arial"
#     styles["Normal"].font.size = Pt(9)

#     doc.add_heading("PGD Power Update Debug Trace", level=0)
#     doc.add_paragraph(
#         "This report records the power vectors before update, after update, before projection, "
#         "and after projection for the selected unfolding layer and its PGD steps."
#     )

#     doc.add_heading("Configuration", level=1)
#     cfg_table = doc.add_table(rows=1, cols=2)
#     cfg_table.style = "Table Grid"
#     cfg_table.rows[0].cells[0].text = "Item"
#     cfg_table.rows[0].cells[1].text = "Value"
#     for key, value in config.items():
#         cells = cfg_table.add_row().cells
#         cells[0].text = str(key)
#         cells[1].text = str(value)

#     if not debug_trace:
#         doc.add_heading("Trace", level=1)
#         doc.add_paragraph("No debug trace was provided.")
#         doc.save(out_path)
#         return out_path

#     doc.add_heading("Learned step sizes used in this trace", level=1)
#     step_table = doc.add_table(rows=1, cols=5)
#     step_table.style = "Table Grid"
#     headers = ["Layer", "PGD step", "step_Pbar", "step_Ptilda", "step_Pprime"]
#     for c, h in enumerate(headers):
#         step_table.rows[0].cells[c].text = h
#     for row in debug_trace:
#         cells = step_table.add_row().cells
#         cells[0].text = str(row.get("layer", ""))
#         cells[1].text = str(row.get("pgd_step", ""))
#         cells[2].text = f"{float(row.get('step_Pbar', np.nan)):.6e}"
#         cells[3].text = f"{float(row.get('step_Ptilda', np.nan)):.6e}"
#         cells[4].text = f"{float(row.get('step_Pprime', np.nan)):.6e}"

#     doc.add_heading("Power vectors", level=1)
#     for row in debug_trace:
#         doc.add_heading(f"Layer {row.get('layer')} - PGD step {row.get('pgd_step')}", level=2)
#         tbl = doc.add_table(rows=1, cols=4)
#         tbl.style = "Table Grid"
#         hdr = ["Stage", "P_bar", "P_tilda", "P_prime"]
#         for c, h in enumerate(hdr):
#             tbl.rows[0].cells[c].text = h
#         stages = [
#             ("Before update", "before"),
#             ("After update", "after_update"),
#             ("Before projection", "before_projection"),
#             ("After projection", "after_projection"),
#         ]
#         for label, suffix in stages:
#             cells = tbl.add_row().cells
#             cells[0].text = label
#             cells[1].text = _vec_to_str(row.get(f"P_bar_{suffix}", []))
#             cells[2].text = _vec_to_str(row.get(f"P_tilda_{suffix}", []))
#             cells[3].text = _vec_to_str(row.get(f"P_prime_{suffix}", []))

#     doc.save(out_path)
#     return out_path

def train_one_batch(
    mat_file,
    model,
    optimizer,
    setup_batch,
    Pmax_db,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    init_delta=0.50,
    init_lambda=0.50,
):
    model.train()

    batch_joint_tensors = []
    batch_layer_histories = []

    optimizer.zero_grad()

    for setup_idx in setup_batch:

        result = run_one_setup_power(
            mat_file=mat_file,
            Pmax_db=float(Pmax_db),
            model=model,
            train_unfolding=True,
            run_classical=False,
            run_deep_unfolding=True, 
            print_components=False,
            setup_idx=int(setup_idx),
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=sigma_w2,
            max_iters=1,
            epsilon=1e-4,
            verbose=False,
            check_sensing=False,
            print_sensing=False,
            init_delta=init_delta,
            init_lambda=init_lambda,
            print_power_tables=False,
            run_sensing_only=False,
        )

        batch_joint_tensors.append(result["deep_final_joint_wsr_tensor"])
        batch_layer_histories.append(result["deep_joint_history"])

    batch_mean_joint_tensor = torch.stack(batch_joint_tensors).mean()

    loss = -batch_mean_joint_tensor

    loss.backward()

    # with torch.no_grad():
    #     print(
    #         "[GRAD DEBUG] "
    #         f"Pbar grad norm = {model.step_Pbar.grad.norm().item():.3e} | "
    #         f"Ptilda grad norm = {model.step_Ptilda.grad.norm().item():.3e} | "
    #         f"Pprime grad norm = {model.step_Pprime.grad.norm().item():.3e}"
    #     )

    optimizer.step()

    # with torch.no_grad():
    #     model.step_Pbar.clamp_(min=1e-5, max=1.0)
    #     model.step_Ptilda.clamp_(min=1e-5, max=1.0)
    #     model.step_Pprime.clamp_(min=1e-5, max=1.0)

    batch_mean_joint = batch_mean_joint_tensor.detach().cpu().item()

    batch_layer_history_mean = np.mean(
        np.stack(batch_layer_histories, axis=0),
        axis=0,
    )

    return batch_mean_joint, batch_layer_history_mean


# def train_joint_unfolding_model(
#     mat_file,
#     model,
#     optimizer,
#     train_setups,
#     train_power_list,
#     K_use=8,
#     M_use=16,
#     N_use=16,
#     sigma_w2=1.0,
#     num_epochs=15,
#     batch_size=10,
#     init_delta=0.50,
#     init_lambda=0.50,
# ):
#     train_setups = np.asarray(train_setups, dtype=int)
#     train_power_list = np.asarray(train_power_list, dtype=float)

#     train_batches = [
#         train_setups[i:i + batch_size]
#         for i in range(0, len(train_setups), batch_size)
#     ]

#     train_joint_log = {}
#     train_layer_joint_log = {}

#     for Pdb in train_power_list:
#         train_joint_log[float(Pdb)] = []
#         train_layer_joint_log[float(Pdb)] = []

#     print("\n" + "=" * 90)
#     print(f"Training Joint Deep Unfolding for Pmax values = {train_power_list}")
#     print("=" * 90)

#     for epoch in range(num_epochs):

#         epoch_batch_final_joints = []
#         epoch_batch_histories = []

#         for Pdb in train_power_list:

#             Pdb = float(Pdb)

#             for batch_id, setup_batch in enumerate(train_batches):

#                 batch_mean_wsr, batch_layer_history_mean = train_one_batch(
#                     mat_file=mat_file,
#                     model=model,
#                     optimizer=optimizer,
#                     setup_batch=setup_batch,
#                     Pmax_db=Pdb,
#                     K_use=K_use,
#                     M_use=M_use,
#                     N_use=N_use,
#                     sigma_w2=sigma_w2,
#                     init_delta=init_delta,
#                     init_lambda=init_lambda,
#                 )

#                 epoch_batch_final_joints.append(batch_mean_wsr)
#                 epoch_batch_histories.append(batch_layer_history_mean)

#                 print(
#                     f"P={Pdb:>6.1f} dB | "
#                     f"Epoch {epoch + 1}/{num_epochs} | "
#                     f"Batch {batch_id + 1}/{len(train_batches)} | "
#                     f"Setups {setup_batch[0]}-{setup_batch[-1]} | "
#                     f"Batch mean final Joint DU = {batch_mean_wsr:.6f}"
#                 )

#         epoch_mean_final_joint = np.mean(epoch_batch_final_joints)

#         epoch_mean_layer_history = np.mean(
#             np.stack(epoch_batch_histories, axis=0),
#             axis=0,
#         )

#         for Pdb in train_power_list:
#             train_joint_log[float(Pdb)].append(epoch_mean_final_joint)
#             train_layer_joint_log[float(Pdb)].append(epoch_mean_layer_history)

#         print(
#             f"[Epoch Summary] "
#             f"Epoch {epoch + 1}/{num_epochs} | "
#             f"Mean final Joint DU over {len(train_batches)} batches = "
#             f"{epoch_mean_final_joint:.6f}"
#         )

#     return {
#         "train_joint_log": train_joint_log,
#         "train_layer_joint_log": train_layer_joint_log,
#     }

def run_joint_power_sweep(
    mat_file,
    power_list=np.arange(10, 20, 10),
    test_setups=np.arange(0, 1),
    model=None,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-4,
    save_plots=False,
    check_sensing=False,
    print_sensing=True,
    init_delta=0.50,
    init_lambda=0.50,
    print_components=True,
    print_power_tables=False,
    run_sensing_only=True,
    run_deep_unfolding=False,
    w_c_user=None,
    w_s_user=None,
):
    """Run joint communication-sensing optimization over powers and setups.

    Printing is intentionally clean:
      1) one BEFORE OPTIMIZATION objective table per power/setup,
      2) one AFTER OPTIMIZATION objective table per power/setup,
      3) communication component tables and sensing component tables.

    The old sweep summary, mean target-rate table, optimized allocation table,
    and before/after sensing-only tables are suppressed here.
    """

    initial_joint_by_power = []
    joint_obj_by_power = []
    comm_wsr_by_power = []
    sensing_wsr_by_power = []
    sensing_only_wsr_by_power = []
    joint_history_by_power = []
    sensing_sinr_by_power = []
    sensing_sinr_db_by_power = []
    du_joint_by_power = []
    du_comm_by_power = []
    du_sensing_by_power = []
    du_history_by_power = []
    all_results = {}

    for Pdb in power_list:
        initial_vals = []
        joint_vals = []
        comm_vals = []
        sensing_vals = []
        sensing_only_vals = []
        joint_histories = []
        convergence_iters = []
        elapsed_times = []
        converged_flags = []
        du_joint_vals = []
        du_comm_vals = []
        du_sensing_vals = []
        du_histories = []
        sensing_sinr_vals = []
        all_results[float(Pdb)] = []

        for setup_idx in test_setups:
            result = run_one_setup_power(
                mat_file=mat_file,
                Pmax_db=float(Pdb),
                model=model,
                train_unfolding=False,
                run_classical=True,
                print_components=False,
                setup_idx=int(setup_idx),
                K_use=K_use,
                M_use=M_use,
                N_use=N_use,
                sigma_w2=sigma_w2,
                max_iters=max_iters,
                epsilon=epsilon,
                verbose=False,
                check_sensing=check_sensing,
                print_sensing=False,
                init_delta=init_delta,
                init_lambda=init_lambda,
                print_power_tables=False,
                run_sensing_only=run_sensing_only,
                run_deep_unfolding=run_deep_unfolding,
                w_c_user=w_c_user,
                w_s_user=w_s_user,
            )

            print_full_power_report(result, print_components=print_components)

            print(
                f"[CONVERGENCE] Pmax={Pdb:>6.1f} dB | "
                f"setup={setup_idx:02d} | "
                f"converged={result['converged']} | "
                f"iters={result['convergence_iter']:03d}/{max_iters} | "
                f"time={result['elapsed_time_sec']:.3f} sec | "
                f"final_delta={result['final_delta']:.3e} | "
                f"final_joint={result['final_wsr']:.6f}"
            )

            if np.isfinite(result.get("deep_final_joint_wsr", np.nan)):
                print(
                    f"[DU TEST] Pmax={Pdb:>6.1f} dB | setup={setup_idx:02d} | "
                    f"DU joint={result['deep_final_joint_wsr']:.6f} | "
                    f"FP/QT joint={result['final_wsr']:.6f} | "
                    f"DU-FP/QT={result['deep_final_joint_wsr'] - result['final_wsr']:.6f}"
                )
            else:
                print(
                    f"[DU TEST] Pmax={Pdb:>6.1f} dB | setup={setup_idx:02d} | "
                    f"DU skipped | FP/QT joint={result['final_wsr']:.6f}"
                )

            sens_only_final = np.nan
            if result.get("sensing_only_result") is not None:
                sens_only_final = result["sensing_only_result"]["final_sensing_wsr"]

            sensing = result["sensing_projection"]
            if sensing is not None:
                sensing_sinr_vals.append(sensing["SINR"])

            initial_vals.append(result["initial_joint_wsr"])
            joint_vals.append(result["final_wsr"])
            comm_vals.append(result["final_comm_wsr"])
            sensing_vals.append(result["final_sensing_wsr"])
            sensing_only_vals.append(sens_only_final)
            joint_histories.append(result["full_joint_history"])
            convergence_iters.append(result["convergence_iter"])
            elapsed_times.append(result["elapsed_time_sec"])
            converged_flags.append(result["converged"])
            du_joint_vals.append(result["deep_final_joint_wsr"])
            du_comm_vals.append(result["deep_final_comm_wsr"])
            du_sensing_vals.append(result["deep_final_sensing_wsr"])
            du_histories.append(result["deep_joint_history"])
            all_results[float(Pdb)].append(result)

        initial_mean = float(np.mean(initial_vals))
        joint_mean = float(np.mean(joint_vals))
        comm_mean = float(np.mean(comm_vals))
        sensing_mean = float(np.mean(sensing_vals))
        sensing_only_mean = float(np.nanmean(sensing_only_vals)) if sensing_only_vals else np.nan
        joint_mean_history = pad_with_last_value(joint_histories).mean(axis=0)
        mean_conv_iter = float(np.mean(convergence_iters))
        mean_elapsed_time = float(np.mean(elapsed_times))
        num_converged = int(np.sum(converged_flags))
        finite_du = np.asarray(du_joint_vals, dtype=float)
        if np.any(np.isfinite(finite_du)):
            du_joint_mean = float(np.nanmean(du_joint_vals))
            du_comm_mean = float(np.nanmean(du_comm_vals))
            du_sensing_mean = float(np.nanmean(du_sensing_vals))
            du_mean_history = pad_with_last_value([h for h in du_histories if len(h) > 0]).mean(axis=0)
        else:
            du_joint_mean = np.nan
            du_comm_mean = np.nan
            du_sensing_mean = np.nan
            du_mean_history = np.array([])

        initial_joint_by_power.append(initial_mean)
        joint_obj_by_power.append(joint_mean)
        comm_wsr_by_power.append(comm_mean)
        sensing_wsr_by_power.append(sensing_mean)
        sensing_only_wsr_by_power.append(sensing_only_mean)
        joint_history_by_power.append(joint_mean_history)
        du_joint_by_power.append(du_joint_mean)
        du_comm_by_power.append(du_comm_mean)
        du_sensing_by_power.append(du_sensing_mean)
        du_history_by_power.append(du_mean_history)

        print(
            f"[POWER CONVERGENCE SUMMARY] P={Pdb:>6.1f} dB | "
            f"mean iters={mean_conv_iter:.2f}/{max_iters} | "
            f"mean time={mean_elapsed_time:.3f} sec | "
            f"converged setups={num_converged}/{len(test_setups)} | "
            f"mean final joint={joint_mean:.6f}"
        )
        if np.isfinite(du_joint_mean):
            print(
                f"[POWER SUMMARY] P={Pdb:>6.1f} dB | initial={initial_mean:.6f} | "
                f"FP/QT={joint_mean:.6f} | DU={du_joint_mean:.6f} | "
                f"DU-FP/QT={du_joint_mean - joint_mean:.6f}"
            )
        else:
            print(
                f"[POWER SUMMARY] P={Pdb:>6.1f} dB | initial={initial_mean:.6f} | "
                f"FP/QT={joint_mean:.6f} | DU=N/A"
            )

        if sensing_sinr_vals:
            sensing_sinr_mean = np.mean(np.stack(sensing_sinr_vals, axis=0), axis=0)
            sensing_sinr_by_power.append(sensing_sinr_mean)
            sensing_sinr_db_by_power.append(lin2db(sensing_sinr_mean))
        else:
            sensing_sinr_by_power.append(None)
            sensing_sinr_db_by_power.append(None)
        if save_plots:
            joint_iterations = np.arange(len(joint_mean_history))
            du_layers = np.arange(len(du_mean_history))

            plt.figure(figsize=(7, 5))

            plt.plot(
                joint_iterations,
                joint_mean_history,
                marker="o",
                linewidth=2,
                # label="Mean Joint Objective - Classical",
            )

            if len(du_mean_history) > 0 and np.any(np.isfinite(du_mean_history)):
                plt.plot(
                    du_layers,
                    du_mean_history,
                    marker="s",
                    linewidth=2,
                    # label="Mean Joint Objective - Deep Unfolding",
                )

            plt.xlabel("Iteration / Layer")
            plt.ylabel("Mean Joint Objective")
            plt.title(f"Joint Classical vs Deep Unfolding Convergence @ {Pdb} dB")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"joint_classical_vs_du_convergence_{Pdb}dB.png", dpi=300)
            plt.show()
        # if save_plots:
        #     joint_iterations = np.arange(len(joint_mean_history))
        #     plt.figure(figsize=(7, 5))
        #     plt.plot(
        #         joint_iterations,
        #         joint_mean_history,
        #         marker="o",
        #         linewidth=2,
        #         label="Mean Joint FP/QT",
        #     )
        #     plt.xlabel("Joint iteration")
        #     plt.ylabel("Mean joint objective")
        #     plt.title(f"Mean Joint FP/QT Convergence @ {Pdb} dB")
        #     plt.grid(True)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(f"mean_joint_convergence_{Pdb}dB.png", dpi=300)
        #     plt.show()

        #     du_layers = np.arange(len(du_mean_history))
        #     plt.figure(figsize=(7, 5))
        #     plt.plot(
        #         du_layers,
        #         du_mean_history,
        #         marker="o",
        #         linewidth=2,
        #         label="Mean Joint Deep Unfolding",
        #     )
        #     plt.xlabel("DU layer")
        #     plt.ylabel("Mean joint objective")
        #     plt.title(f"Mean Joint DU Convergence @ {Pdb} dB")
        #     plt.grid(True)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(f"mean_joint_du_convergence_{Pdb}dB.png", dpi=300)
        #     plt.show()

    initial_joint_by_power = np.asarray(initial_joint_by_power, dtype=float)
    joint_obj_by_power = np.asarray(joint_obj_by_power, dtype=float)
    comm_wsr_by_power = np.asarray(comm_wsr_by_power, dtype=float)
    sensing_wsr_by_power = np.asarray(sensing_wsr_by_power, dtype=float)
    sensing_only_wsr_by_power = np.asarray(sensing_only_wsr_by_power, dtype=float)
    du_joint_by_power = np.asarray(du_joint_by_power, dtype=float)
    du_comm_by_power = np.asarray(du_comm_by_power, dtype=float)
    du_sensing_by_power = np.asarray(du_sensing_by_power, dtype=float)

    if save_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(
            power_list,
            initial_joint_by_power,
            "--s",
            linewidth=2,
            label="EPA before optimization",
        )
        plt.plot(
            power_list,
            joint_obj_by_power,
            "-o",
            linewidth=2,
            label="Joint Classical Optimization",
        )
        if np.any(np.isfinite(du_joint_by_power)):
            plt.plot(
                power_list,
                du_joint_by_power,
                "-^",
                linewidth=2,
                label="Joint Deep Unfolding Optimization",
            )
        plt.xlabel("Transmit Power $P_{max}$ (dB)")
        plt.ylabel("Mean joint objective")
        plt.title("Power Sweep: EPA vs Classical vs Deep Unfolding")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("power_sweep_epa_classical_du.png", dpi=300)
        plt.show()

        if any(x is not None for x in sensing_sinr_db_by_power):
            sensing_db_mat = np.array([x for x in sensing_sinr_db_by_power if x is not None])
            power_valid = np.array([p for p, x in zip(power_list, sensing_sinr_db_by_power) if x is not None])
            Tg = sensing_db_mat.shape[1]
            plt.figure(figsize=(8, 6))
            for q in range(Tg):
                plt.plot(
                    power_valid,
                    sensing_db_mat[:, q],
                    marker="o",
                    linewidth=2,
                    label=f"Target {q}",
                )
            plt.xlabel("Transmit Power $P_{max}$ (dB)")
            plt.ylabel("Optimized sensing SINR (dB)")
            plt.title("Optimized Sensing SINR vs Power")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("optimized_sensing_sinr_vs_power.png", dpi=300)
            plt.show()

    return {
        "power_list": np.asarray(power_list, dtype=float),
        "initial_joint_by_power": initial_joint_by_power,
        "joint_obj_by_power": joint_obj_by_power,
        "comm_wsr_by_power": comm_wsr_by_power,
        "sensing_wsr_by_power": sensing_wsr_by_power,
        "sensing_only_wsr_by_power": sensing_only_wsr_by_power,
        "joint_history_by_power": joint_history_by_power,
        "sensing_sinr_by_power": sensing_sinr_by_power,
        "sensing_sinr_db_by_power": sensing_sinr_db_by_power,
        "du_joint_by_power": du_joint_by_power,
        "du_comm_by_power": du_comm_by_power,
        "du_sensing_by_power": du_sensing_by_power,
        "du_history_by_power": du_history_by_power,
        "all_results": all_results,

        # Backward-compatible aliases for older plotting scripts.
        "before_wsr_by_power": initial_joint_by_power,
        "classical_wsr_by_power": joint_obj_by_power,
        "classical_history_by_power": joint_history_by_power,
    }


# Backward-compatible alias. Prefer run_joint_power_sweep(...).
def run_classical_power_sweep(*args, **kwargs):
    return run_joint_power_sweep(*args, **kwargs)

if __name__ == "__main__":
    np.random.seed(42)

    # Change this to your actual local .mat dataset path if needed.a
    mat_file = "jointopt_unfolding_dataset_50_setups.mat"
    mat_file1 = "JointOpt_Dataset_with_V.mat"
    # Convenience fallback for this ChatGPT sandbox/session.
    sandbox_mat_file = "/mnt/data/1st_june_P10db_jopt_hwi5bit_itr_200_wcws_0.2_0.8_ch40_s1.mat"
    if not os.path.exists(mat_file) and os.path.exists(sandbox_mat_file):
        mat_file = sandbox_mat_file

    K_use = 8
    M_use = 16
    N_use = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = 30
    num_pgd_steps = 5

    model = JointCommSensingUnfoldingPGD(
        K=K_use,
        num_layers=num_layers,
        num_pgd_steps=num_pgd_steps,
        init_step_bar=0.05,
        init_step_tilda=0.05,
        init_step_prime=0.05,
        enforce_full_power=False,
    ).to(device).double()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Debug setting. For final experiments, use train_setups=np.arange(0, 30)
    # and test_setups=np.arange(30, 40) when your dataset has 40 setups.
    train_setups = np.arange(0, 40)
    # test_setups = np.arange(40, 50)

    # train_setups = np.arange(0, 1)
    test_setups = np.arange(0, 1)

    train_power_list = np.array([10.0])
    # For all-power training, use: train_power_list = np.arange(-20, 31, 5)

    train_logs = train_joint_unfolding_model(
        mat_file=mat_file,
        model=model,
        optimizer=optimizer,
        train_setups=train_setups,
        train_power_list=train_power_list,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        sigma_w2=1.0,
        num_epochs=2,
        batch_size=10,
        init_delta=0.50,
        init_lambda=0.50,
        shuffle_batches=True,
        seed=42,
    )
    learned_steps = print_learned_step_sizes(model)
    # power_list = np.arange(-20, 31, 5)
    power_list=np.arange(-20, 31, 5)
    results = run_joint_power_sweep(
        mat_file=mat_file1,
        power_list=power_list,
        test_setups=test_setups,
        model=model,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        sigma_w2=1.0,
        max_iters=50,
        epsilon=1e-4,
        check_sensing=False,
        print_sensing=False,
        init_delta=0.50,
        init_lambda=0.50,
        print_components=False,
        print_power_tables=False,
        run_sensing_only=True,
        run_deep_unfolding=True,
        save_plots=True,
        w_c_user=0.2,
        w_s_user=0.8,

    )

    Pdb_tradeoff = 10.0

    # Communication-sensing weight pairs
    # Here w_c + w_s = 1
    tradeoff_weights = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.1, 0.9),
        (0.0, 1.0),
    ]

    tradeoff_rows = []

    # ============================================================
    # Run classical joint optimization for each weight pair
    # ============================================================
    for wc, ws in tradeoff_weights:

        # print("\n" + "=" * 100)
        # print(f"TRADEOFF POINT: w_c = {wc:.2f}, w_s = {ws:.2f}")
        # print("=" * 100)

        setup_comm_raw = []
        setup_sensing_raw = []
        setup_joint = []
        setup_sum_Pbar = []
        setup_sum_Ptilda = []
        setup_sum_Pprime = []

        for setup_idx in test_setups:

            result = run_one_setup_power(
                mat_file=mat_file1,
                Pmax_db=Pdb_tradeoff,
                model=None,
                train_unfolding=False,
                run_classical=True,
                run_deep_unfolding=False,
                setup_idx=int(setup_idx),
                K_use=K_use,
                M_use=M_use,
                N_use=N_use,
                sigma_w2=1.0,
                max_iters=50,
                epsilon=1e-4,
                verbose=False,
                check_sensing=False,
                print_sensing=False,
                print_components=False,
                print_power_tables=False,
                run_sensing_only=False,
                init_delta=0.50,
                init_lambda=0.50,
                w_c_user=wc,
                w_s_user=ws,
        
            )

            # Raw communication WSR
            comm_raw = float(result["final_comm_wsr"])

            # Raw sensing WSR without multiplying by w_s
            sensing_raw = float(
                result["sensing_projection"]["weighted_wsr_no_ws"]
            )

            setup_comm_raw.append(comm_raw)
            setup_sensing_raw.append(sensing_raw)
            setup_joint.append(float(result["final_wsr"]))

            setup_sum_Pbar.append(float(np.sum(result["P_bar_opt"])))
            setup_sum_Ptilda.append(float(np.sum(result["P_tilda_opt"])))
            setup_sum_Pprime.append(float(np.sum(result["P_prime_opt"])))

        tradeoff_rows.append({
            "w_c": wc,
            "w_s": ws,
            "comm_raw_mean": float(np.mean(setup_comm_raw)),
            "sensing_raw_mean": float(np.mean(setup_sensing_raw)),
            "joint_mean": float(np.mean(setup_joint)),
            "sum_Pbar_mean": float(np.mean(setup_sum_Pbar)),
            "sum_Ptilda_mean": float(np.mean(setup_sum_Ptilda)),
            "sum_Pprime_mean": float(np.mean(setup_sum_Pprime)),
        })

    # ============================================================
    # Convert to arrays
    # ============================================================
    wc_arr = np.array([row["w_c"] for row in tradeoff_rows])
    ws_arr = np.array([row["w_s"] for row in tradeoff_rows])

    comm_raw_arr = np.array([row["comm_raw_mean"] for row in tradeoff_rows])
    sensing_raw_arr = np.array([row["sensing_raw_mean"] for row in tradeoff_rows])

    # ============================================================
    # Tradeoff plot: raw communication WSR vs raw sensing WSR
    # ============================================================
    plt.figure(figsize=(8, 6))

    plt.plot(
        comm_raw_arr,
        sensing_raw_arr,
        marker="o",
        linewidth=2,
        label="Classical joint optimization"
    )

    for i in range(len(tradeoff_rows)):
        plt.annotate(
            f"({wc_arr[i]:.1f},{ws_arr[i]:.1f})",
            (comm_raw_arr[i], sensing_raw_arr[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9
        )

    plt.xlabel("Raw communication WSR")
    plt.ylabel("Raw sensing WSR")
    plt.title(f"Communication-Sensing Tradeoff @ {Pdb_tradeoff:g} dB")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_fig = f"comm_sensing_tradeoff_{Pdb_tradeoff:g}dB.png"
    plt.savefig(out_fig, dpi=300)
    plt.show()

    print(f"\nSaved tradeoff plot: {out_fig}")

    # ============================================================
    # Print summary table
    # ============================================================
    print("\n" + "=" * 120)
    print("COMMUNICATION-SENSING TRADEOFF SUMMARY")
    print("=" * 120)
    print(
        f"{'w_c':>6} | {'w_s':>6} | {'Comm raw':>12} | {'Sensing raw':>12} | "
        f"{'Joint obj':>12} | {'sum Pbar':>12} | {'sum Ptilda':>12} | {'sum Pprime':>12}"
    )
    print("-" * 120)

    for row in tradeoff_rows:
        print(
            f"{row['w_c']:6.2f} | "
            f"{row['w_s']:6.2f} | "
            f"{row['comm_raw_mean']:12.6f} | "
            f"{row['sensing_raw_mean']:12.6f} | "
            f"{row['joint_mean']:12.6f} | "
            f"{row['sum_Pbar_mean']:12.6f} | "
            f"{row['sum_Ptilda_mean']:12.6f} | "
            f"{row['sum_Pprime_mean']:12.6f}"
        )

    # ============================================================
    # Save tradeoff values to CSV
    # ============================================================
    out_csv = f"comm_sensing_tradeoff_{Pdb_tradeoff:g}dB.csv"

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "w_c,w_s,comm_raw_mean,sensing_raw_mean,joint_mean,"
            "sum_Pbar_mean,sum_Ptilda_mean,sum_Pprime_mean\n"
        )

        for row in tradeoff_rows:
            f.write(
                f"{row['w_c']},{row['w_s']},"
                f"{row['comm_raw_mean']},{row['sensing_raw_mean']},"
                f"{row['joint_mean']},"
                f"{row['sum_Pbar_mean']},{row['sum_Ptilda_mean']},"
                f"{row['sum_Pprime_mean']}\n"
            )

    print(f"Saved tradeoff CSV: {out_csv}")

    # ============================================================
    # Plot communication, sensing, and joint objective histories
    # for one selected power and one selected setup.
    # ============================================================

    Pdb_to_plot = 10.0
    setup_pos = 0   # because test_setups = np.arange(0, 1)

    r = results["all_results"][float(Pdb_to_plot)][setup_pos]

    w_c = float(r["w_c"])

    # Classical histories stored inside one setup result
    comm_full = np.concatenate([
        [r["initial_comm_wsr"]],
        np.asarray(r["comm_history"], dtype=float)
    ])

    sensing_full = np.concatenate([
        [r["initial_sensing_wsr"]],
        np.asarray(r["sensing_history"], dtype=float)
    ])

    joint_full = np.asarray(r["full_joint_history"], dtype=float)

    # Communication history is raw WSR, so multiply by w_c
    comm_contribution_full = w_c * comm_full

    # Sensing history is already weighted by w_s in your code
    sensing_contribution_full = sensing_full

    plt.figure(figsize=(8, 5))
    plt.plot(
        comm_contribution_full,
        marker="o",
        linewidth=2,
        label="Weighted communication part"
    )
    plt.plot(
        sensing_contribution_full,
        marker="s",
        linewidth=2,
        label="Weighted sensing part"
    )
    plt.plot(
        joint_full,
        marker="^",
        linewidth=2,
        label="Joint objective"
    )

    plt.xlabel("Classical iteration")
    plt.ylabel("Objective contribution")
    plt.title(f"Classical Objective Decomposition @ {Pdb_to_plot:g} dB")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"classical_comm_sensing_joint_decomposition_{Pdb_to_plot:g}dB.png", dpi=300)
    plt.show()


    
# if __name__ == "__main__":
#     np.random.seed(42)

#     mat_file1 = "JointOpt_Dataset_with_V.mat"

#     K_use = 8
#     M_use = 16
#     N_use = 16

#     setup_idx = 0

#     # ============================================================
#     # Power list
#     # ============================================================
#     power_list = np.arange(-20, 31, 5)
#     # This gives:
#     # [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

#     all_results = {
#         "joint": {},
#         "comm_only": {},
#         "sens_only": {},
#     }

#     summary_rows = []

#     for Pdb in power_list:
#         Pdb = float(Pdb)

#         print(f"RUNNING ALL CLASSICAL MODES FOR Pmax = {Pdb:.1f} dB")


#         common_kwargs = dict(
#             mat_file=mat_file1,
#             Pmax_db=Pdb,
#             model=None,
#             train_unfolding=False,
#             run_classical=True,
#             run_deep_unfolding=False,
#             setup_idx=setup_idx,
#             K_use=K_use,
#             M_use=M_use,
#             N_use=N_use,
#             sigma_w2=1.0,
#             max_iters=1000,
#             epsilon=1e-4,
#             verbose=True,
#             check_sensing=False,
#             init_delta=0.50,
#             init_lambda=0.50,
#             print_components=False,
#             print_power_tables=True,
#         )

#         print("\n\n JOINT ONLY CLASSICAL ")
#         joint_classical = run_one_setup_power(
#             **common_kwargs,
#             classical_mode="joint",
#             run_sensing_only=False,
#             print_sensing=True,
#         )

#         print("\n\n COMM ONLY CLASSICAL ")
#         comm_classical = run_one_setup_power(
#             **common_kwargs,
#             classical_mode="comm_only",
#             run_sensing_only=False,
#             print_sensing=False,
#         )

#         print("\n\n SENS ONLY CLASSICAL ")
#         sens_classical = run_one_setup_power(
#             **common_kwargs,
#             classical_mode="sens_only",
#             run_sensing_only=True,
#             print_sensing=True,
#         )

#         all_results["joint"][Pdb] = joint_classical
#         all_results["comm_only"][Pdb] = comm_classical
#         all_results["sens_only"][Pdb] = sens_classical

#         # ========================================================
#         # Compact summary values
#         # ========================================================
#         joint_initial = joint_classical.get("initial_joint_wsr", np.nan)
#         joint_final = joint_classical.get("final_wsr", np.nan)
#         joint_comm_final = joint_classical.get("final_comm_wsr", np.nan)
#         joint_sens_final = joint_classical.get("final_sensing_wsr", np.nan)

#         comm_initial = comm_classical.get("initial_comm_wsr", np.nan)
#         comm_final = comm_classical.get("final_comm_wsr", np.nan)

#         sens_initial = sens_classical.get("initial_sensing_wsr", np.nan)
#         sens_final = sens_classical.get("final_sensing_wsr", np.nan)

#         summary_rows.append(
#             {
#                 "Pmax_db": Pdb,
#                 "joint_initial": joint_initial,
#                 "joint_final": joint_final,
#                 "joint_comm_final": joint_comm_final,
#                 "joint_sens_final": joint_sens_final,
#                 "comm_initial": comm_initial,
#                 "comm_final": comm_final,
#                 "sens_initial": sens_initial,
#                 "sens_final": sens_final,
#             }
#         )

#         print("\n" + "=" * 120)
#         print(f"SUMMARY FOR Pmax = {Pdb:.1f} dB")
#         print("=" * 120)
#         print(f"Joint initial      = {joint_initial:.6f}")
#         print(f"Joint final        = {joint_final:.6f}")
#         print(f"Joint comm final   = {joint_comm_final:.6f}")
#         print(f"Joint sens final   = {joint_sens_final:.6f}")
#         print(f"Comm-only initial  = {comm_initial:.6f}")
#         print(f"Comm-only final    = {comm_final:.6f}")
#         print(f"Sens-only initial  = {sens_initial:.6f}")
#         print(f"Sens-only final    = {sens_final:.6f}")
#         print("=" * 120)

#     # ============================================================
#     # Save sweep summary CSV
#     # ============================================================
#     out_csv = "classical_three_modes_power_sweep_summary.csv"

#     with open(out_csv, "w", encoding="utf-8") as f:
#         f.write(
#             "Pmax_db,"
#             "joint_initial,joint_final,joint_comm_final,joint_sens_final,"
#             "comm_initial,comm_final,"
#             "sens_initial,sens_final\n"
#         )

#         for row in summary_rows:
#             f.write(
#                 f"{row['Pmax_db']},"
#                 f"{row['joint_initial']},"
#                 f"{row['joint_final']},"
#                 f"{row['joint_comm_final']},"
#                 f"{row['joint_sens_final']},"
#                 f"{row['comm_initial']},"
#                 f"{row['comm_final']},"
#                 f"{row['sens_initial']},"
#                 f"{row['sens_final']}\n"
#             )

#     print("\n" + "=" * 120)
#     print("POWER SWEEP FINISHED")
#     print("=" * 120)
#     print("Saved CSV:", out_csv)

#     print("\nCompact final table")
#     print("-" * 120)
#     print(
#         f"{'Pmax':>8} | {'Joint init':>12} | {'Joint final':>12} | "
#         f"{'Comm-only':>12} | {'Sens-only':>12}"
#     )
#     print("-" * 120)

#     for row in summary_rows:
#         print(
#             f"{row['Pmax_db']:8.1f} | "
#             f"{row['joint_initial']:12.6f} | "
#             f"{row['joint_final']:12.6f} | "
#             f"{row['comm_final']:12.6f} | "
#             f"{row['sens_final']:12.6f}"
#         )

#     print("-" * 120)
