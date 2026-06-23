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
# =============================================================================
# Imports and global configuration
# =============================================================================
# Core scientific libraries, project-specific kappa builders, matrix builders,
# and PyTorch are imported here. No optimization is executed at import time.

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

# -----------------------------------------------------------------------------
# Lazy unfolding imports
# -----------------------------------------------------------------------------
# Do NOT import an unfolding implementation globally.  The communication-only
# path and the joint communication-sensing path use different unfolding files.
# Loading is done only inside the mode that actually needs it.

JointCommSensingUnfoldingPGD = None
to_torch_1d = None
to_torch_2d = None
scalar_to_torch = None

COMMUNICATION_RUN_MODES = {
    "comm",
    "comm_lambda",
    "comm_only",
    "comm_only_compare",
    "communication_lambda",
    "only_comm",
    "comm_fpqt",
}


# =============================================================================
# Mode selection and lazy unfolding API loading
# =============================================================================
# The script supports both communication-only and joint communication-sensing
# modes. The correct unfolding implementation is imported only when needed.
# Return True when the selected run mode should ignore all sensing variables.
def is_communication_mode(mode):
    """True for pure communication modes. These must not use sensing tensors."""
    return str(mode).lower().strip() in COMMUNICATION_RUN_MODES



# Load the communication-only or joint unfolding class only for the requested mode.
def load_unfolding_api(mode):
    """
    Load the correct unfolding implementation for the requested mode.

    Communication-only/lambda mode:
        unfolding_for_communication.py
        This is now pure communication-only. It exposes
        JointCommSensingUnfoldingPGD as a backward-compatible alias.

    Joint communication-sensing mode:
        unfolding_for_jointopt_latest.py
    """
    global JointCommSensingUnfoldingPGD, to_torch_1d, to_torch_2d, scalar_to_torch

    if mode in ["comm_lambda", "comm_only_compare", "communication_lambda", "comm", "only_comm", "comm_only", "comm_fpqt"]:
        from unfolding_for_communication import (
            JointCommSensingUnfoldingPGD as _JointCommSensingUnfoldingPGD,
            to_torch_1d as _to_torch_1d,
            to_torch_2d as _to_torch_2d,
            scalar_to_torch as _scalar_to_torch,
        )
        loaded_from = "unfolding_for_communication"

    elif mode in ["joint", "joint_comm_sensing", "joint_lambda", "joint_comm_sensing_lambda"]:
        from unfolding_for_jointopt_latest import (
            JointCommSensingUnfoldingPGD as _JointCommSensingUnfoldingPGD,
            to_torch_1d as _to_torch_1d,
            to_torch_2d as _to_torch_2d,
            scalar_to_torch as _scalar_to_torch,
        )
        loaded_from = "unfolding_for_jointopt_latest"

    else:
        raise ValueError(f"Unknown unfolding mode: {mode}")

    JointCommSensingUnfoldingPGD = _JointCommSensingUnfoldingPGD
    to_torch_1d = _to_torch_1d
    to_torch_2d = _to_torch_2d
    scalar_to_torch = _scalar_to_torch

    return {
        "JointCommSensingUnfoldingPGD": JointCommSensingUnfoldingPGD,
        "to_torch_1d": to_torch_1d,
        "to_torch_2d": to_torch_2d,
        "scalar_to_torch": scalar_to_torch,
        "loaded_from": loaded_from,
    }

# from comm_lambda_qt_unfolding_alpha import (
#     run_communication_only_lambda_qt,
#     run_classical_lambda_qt_comparison,
#     plot_comm_only_method_comparison,
# )
from Communication_Accelerated_QT import (
# =============================================================================
# Accelerated QT / lambda-QT optimizer imports
# =============================================================================
# Communication-only lambda-QT and joint lambda-QT routines are imported here
# and selected later depending on RUN_MODE.
    run_communication_only_lambda_qt,
    run_classical_lambda_qt_comparison,
    plot_comm_only_method_comparison,
    print_lambda_step_size_summary,
    save_lambda_step_size_summary,
)
from Joint_Accelerated_QT import (
    run_joint_lambda_qt,
    plot_joint_method_comparison,
    print_joint_lambda_step_size_summary,
    save_joint_lambda_step_size_summary,
)
# Utility
# =============================================================================
# Communication kappa coefficient construction
# =============================================================================
# These helpers build the scalar coefficients used in the communication SINR
# expression. They are shared by classical FP/QT, lambda-QT, and DU modes.
# Extract all AP/panel entries for a fixed user index k from an L x K nested list.
def _slice_k(lst_2d, k):
    """1-D slice over panels for fixed user k."""
    return [lst_2d[l][k] for l in range(len(lst_2d))]


# Build every scalar kappa — called ONCE before the iteration loop
# Build all communication SINR kappa coefficients once for the current setup.
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

# Compute the pilot-power-independent thermal and ADC distortion terms.
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

# Build ADC quantization covariance matrices using a fixed reference power split.
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
# =============================================================================
# Communication SINR, WSR, and FP/QT auxiliary-variable updates
# =============================================================================
# This block evaluates the communication objective and computes closed-form
# updates for pilot power P_bar and data power P_tilda.
# Compute the desired-signal numerator coefficient A_k for user k.
def compute_Ak(k, kappa_S, P_tilda):
    return P_tilda[k] * kappa_S[k]


# Compute the pilot-power-dependent denominator coefficient B_k for user k.
def compute_Bk(k, K, kappa_V1, kappa_K1, kappa_DAC1, kappa_M1, P_tilda):
    self_term = P_tilda[k] * (kappa_V1[k] + kappa_K1[k])
    mui_term  = sum(P_tilda[i] * kappa_M1[k, i]
                    for i in range(K) if i != k)
    return self_term + mui_term + kappa_DAC1[k]


# Compute the pilot-power-independent denominator coefficient C_k for user k.
def compute_Ck(k, K, kappa_V0, kappa_K0, kappa_DAC0, kappa_M0, P_tilda):
    self_term = P_tilda[k] * (kappa_V0[k] + kappa_K0[k])
    mui_term  = sum(P_tilda[i] * kappa_M0[k, i]
                    for i in range(K) if i != k)
    return self_term + mui_term + kappa_DAC0[k]


# SINR and WSR
# Evaluate the exact communication SINR for one user from the current powers.
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

# OPTIONAL DIAGNOSTIC: print the communication denominator/numerator components user by user.
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

# OPTIONAL LEGACY HELPER: clip P_bar for a two-block communication-only constraint.
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


# OPTIONAL LEGACY HELPER: clip P_tilda for a two-block communication-only constraint.
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

# Compute weighted sum-rate from the per-user communication SINR values.
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
# Update the FP/QT auxiliary variable mu for the communication objective.
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
  
# Closed-form FP/QT update for pilot/channel-estimation power P_bar.
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
    
# Closed-form FP/QT update for communication data power P_tilda.
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

# =============================================================================
# Power feasibility, projection, and clipping utilities
# =============================================================================
# These utilities enforce per-user power budgets for two-block and three-block
# allocations: P_bar, P_tilda, and P_prime.
# Convert scalar or vector Pmax input into a K-length power-budget vector.
def _as_power_vector(P_total_max, K):
    if P_total_max is None:
        return None
    if np.isscalar(P_total_max):
        return float(P_total_max) * np.ones(K, dtype=float)
    return np.asarray(P_total_max, dtype=float).reshape(K)

# Project a vector onto the nonnegative simplex with total-power budget.
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


# Jointly project P_bar, P_tilda, and P_prime to avoid sequential clipping bias.
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
# OPTIONAL SAFETY HELPER: reset users to EPA if a candidate update violates power constraints.
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

# Clip P_bar while keeping P_tilda and P_prime fixed under the joint budget.
def clip_P_bar_joint(P_bar_new, P_tilda, P_prime, P_total_max, eps=1e-12):
    P_bar_new = np.asarray(P_bar_new, dtype=float)
    if P_total_max is None:
        return np.maximum(P_bar_new, eps)
    Pmax = _as_power_vector(P_total_max, len(P_bar_new))
    upper = np.maximum(Pmax - np.asarray(P_tilda) - np.asarray(P_prime) - eps, eps)
    return np.clip(P_bar_new, eps, upper)


# Clip P_tilda while keeping P_bar and P_prime fixed under the joint budget.
def clip_P_tilda_joint(P_tilda_new, P_bar, P_prime, P_total_max, eps=1e-12):
    P_tilda_new = np.asarray(P_tilda_new, dtype=float)
    if P_total_max is None:
        return np.maximum(P_tilda_new, eps)
    Pmax = _as_power_vector(P_total_max, len(P_tilda_new))
    upper = np.maximum(Pmax - np.asarray(P_bar) - np.asarray(P_prime) - eps, eps)
    return np.clip(P_tilda_new, eps, upper)


# Clip P_prime while keeping P_bar and P_tilda fixed under the joint budget.
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

# =============================================================================
# Sensing coefficient loading and MATLAB-matching coefficient construction
# =============================================================================
# This block either loads sensing coefficients from .mat files or reconstructs
# them from geometry, target steering vectors, and Monte-Carlo samples.
# Normalize MATLAB/Python sensing coefficient orientation to target x user format.
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


# OPTIONAL FALLBACK: load precomputed sensing coefficients if they are present in the dataset.
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


# Convert a vector-like object into a complex column vector.
def _col(x):
    return np.asarray(x, dtype=complex).reshape(-1, 1)


# Fetch the steering vector b_{l,q} for AP/panel l and target q.
def _get_b_lq(b_target_all, l, q):
    return _col(b_target_all[:, l, q])


# OPTIONAL HELPER: build sensing combiners for one target from steering vectors or V_combiner.
def _build_v_list_for_q(b_target_all, q, V_combiner_all=None, v_iter=None):
    M = b_target_all.shape[1]
    if V_combiner_all is None:
        return [_get_b_lq(b_target_all, l, q) for l in range(M)]
    return [_col(V_combiner_all[:, l, q, v_iter]) for l in range(M)]

# Compute desired sensing coefficient a_{q,k} without multiplying by P_prime[k].
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


# Compute inter-target sensing denominator coefficient for b_{q,k}.
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


# Compute target-wise clutter coefficient from direct-channel Monte-Carlo samples.
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


# Compute the fixed DAC distortion floor for one target and one MC iteration.
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


# Compute the thermal-noise sensing floor for one target.
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


# Compute the ADC-quantization sensing floor for one target.
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

# Build a_sens, b_sens, and n_sens using MATLAB-matching MC formulas.
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

# =============================================================================
# Sensing SINR, sensing WSR, and sensing FP/QT auxiliary updates
# =============================================================================
# These functions evaluate sensing performance and update the LDT/QT auxiliary
# variables used to optimize sensing power P_prime.
# Compute desired sensing signal S_q and denominator D_q for all targets.
def compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens):
    P_prime = np.asarray(P_prime, dtype=float).reshape(-1)
    S = np.maximum(np.asarray(a_sens, dtype=float) @ P_prime, 0.0)
    D = np.maximum(np.asarray(b_sens, dtype=float) @ P_prime + np.asarray(n_sens, dtype=float), 1e-30)
    return S, D


# Evaluate sensing SINR for all targets from P_prime.
def compute_sensing_sinr(P_prime, a_sens, b_sens, n_sens):
    S, D = compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens)
    return S / (D + 1e-30)


# Compute weighted sensing sum-rate contribution.
def compute_sensing_wsr(P_prime, a_sens, b_sens, n_sens, target_weights=None, w_s=1.0):
    sinr = compute_sensing_sinr(P_prime, a_sens, b_sens, n_sens)
    Tg = len(sinr)
    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)
    rho = float(w_s) * np.asarray(target_weights, dtype=float).reshape(Tg)
    return float(np.sum(rho * np.log2(1.0 + np.maximum(sinr, 0.0))))


# LDT update for sensing auxiliary mu_s.
def update_sensing_mu(P_prime, a_sens, b_sens, n_sens):
    """LDT update: mu_q = SINR_sens,q."""
    return compute_sensing_sinr(P_prime, a_sens, b_sens, n_sens)


# QT update for sensing auxiliary y_s.
def update_sensing_y(P_prime, a_sens, b_sens, n_sens, mu_s, target_weights=None, w_s=1.0):
    """QT update: y_q = sqrt(rho_q(1+mu_q)S_q)/(S_q+D_q)."""
    S, D = compute_sensing_S_D(P_prime, a_sens, b_sens, n_sens)
    Tg = len(S)
    if target_weights is None:
        target_weights = np.ones(Tg, dtype=float)
    rho = float(w_s) * np.asarray(target_weights, dtype=float).reshape(Tg)
    numerator = np.sqrt(np.maximum(rho * (1.0 + np.asarray(mu_s)) * S, 0.0))
    return numerator / (S + D + 1e-30)


# Derivative of the sensing transformed objective with respect to one P_prime coordinate.
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


# Coordinate-wise bisection update for sensing power P_prime.
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

# Repeat coordinate updates until the P_prime block converges.
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

# Combine communication WSR and sensing WSR into the joint objective.
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

# Evaluate all objective components and hardware terms for one feasible power state.
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
# Accept a three-block update only when it does not decrease the joint objective.
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

# Return detailed sensing components, SINR, rates, and weighted contributions.
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
        "raw_wsr": float(np.sum(weighted_no_ws)),
        "wsr": float(np.sum(weighted_with_ws)),
        "weighted_wsr": float(np.sum(weighted_with_ws)),
    }

# ============
# Replacement optimizer: joint BCO loop
# Requires your existing communication functions:
#   build_all_kappas, build_kappa_Th0_ADC0, compute_SINR_exact,
#   update_auxiliary, compute_P_bar, compute_P_tilda, compute_WSR_exact
# ============

# =============================================================================
# Classical joint communication-sensing optimization loop
# =============================================================================
# This is the baseline block-coordinate optimizer over P_bar, P_tilda, and
# P_prime, with objective-safety backtracking after each full update.
# Classical joint FP/QT baseline over P_bar, P_tilda, and P_prime.
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

# =============================================================================
# MATLAB dataset loading and array conversion helpers
# =============================================================================
# These helpers convert MATLAB arrays into Python lists indexed by AP/user and
# load the channel, sensing, and hardware variables needed by the optimizers.
# Fetch a variable from a .mat file using multiple possible names.
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


# Select one setup from a MATLAB array if a setup dimension exists.
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
# Convert dB power to linear scale.
def db2lin(db):
    return 10.0 ** (db / 10.0)


# Convert linear power/SINR values to dB with numerical protection.
def lin2db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


# Force a square matrix to be Hermitian by averaging with its conjugate transpose.
def hermitianize(X):
    return 0.5 * (X + X.conj().T)


# OPTIONAL DIAGNOSTIC: compute relative Frobenius-norm error between matrices.
def relerr(A, B):
    return np.linalg.norm(A - B, "fro") / (np.linalg.norm(A, "fro") + 1e-30)


# Extract all AP/panel entries for a fixed user index k from an L x K nested list.
def _slice_k(lst_2d, k):
    return [lst_2d[l][k] for l in range(len(lst_2d))]


# Convert N x N x K x M MATLAB arrays into AP/user nested lists.
def mat_4d_to_lk_list(X, K, M, hermitian=True):
    out = [[] for _ in range(M)]

    for l in range(M):
        for k in range(K):
            X_lk = np.asarray(X[:, :, k, l], dtype=complex)
            if hermitian:
                X_lk = hermitianize(X_lk)
            out[l].append(X_lk)

    return out


# Convert N x N x M MATLAB arrays into AP-indexed lists.
def mat_3d_ap_to_list(X, M, hermitian=True):
    out = []

    for l in range(M):
        X_l = np.asarray(X[:, :, l], dtype=complex)
        if hermitian:
            X_l = hermitianize(X_l)
        out.append(X_l)

    return out


# Convert h_bar MATLAB arrays into AP/user nested column-vector lists.
def hbar_to_lk_list(Hbar, K, M):
    out = [[] for _ in range(M)]

    for l in range(M):
        for k in range(K):
            h_lk = np.asarray(Hbar[:, k, l], dtype=complex).reshape(-1, 1)
            out[l].append(h_lk)

    return out


# Convert per-user pilot/data matrices into a Python list.
def Z_to_list(Z_matrices, K):

    return [
        np.asarray(Z_matrices[:, :, k], dtype=complex)
        for k in range(K)
    ]


# Build identity MR combiners for all AP/user pairs.
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

# Convert a vector-like object into a complex column vector.
def _col(x):
    return np.asarray(x, dtype=complex).reshape(-1, 1)


# Fetch target steering vector b_{l,q} as a column vector.
def get_b_lq(b_target_all, l, q):
    """
    b_target_all has shape N x M x Tg.
    Returns b_{l,q} as N x 1.
    """
    return _col(b_target_all[:, l, q])


# OPTIONAL DIAGNOSTIC: build MR sensing combiners v_{l,q}=b_{l,q}.
def build_sensing_v_list_MR(b_target_all, q):
    """
    MR sensing combiner:
        v_{l,q} = b_{l,q}
    """
    M = b_target_all.shape[1]
    return [get_b_lq(b_target_all, l, q) for l in range(M)]

# Build clutter-aware sensing combiners from V_combiner_all for one MC iteration.
def build_sensing_v_list_from_V(V_combiner_all, q, iter_idx):
    """
    V_combiner_all shape: N x M x Tg x num_iterations

    Returns:
        v_list[l] = v_{l,q} for MC iteration iter_idx
    """
    M = V_combiner_all.shape[1]
    return [_col(V_combiner_all[:, l, q, iter_idx]) for l in range(M)]

# OPTIONAL FALLBACK: estimate target variances from reference covariance matrices.
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
# OPTIONAL DIAGNOSTIC: compute scalar sensing desired power for MR/V projection checks.
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


# OPTIONAL DIAGNOSTIC: compute scalar sensing inter-target power for projection checks.
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


# OPTIONAL DIAGNOSTIC: compute scalar clutter power for sensing projection checks.
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


# OPTIONAL DIAGNOSTIC: compute scalar DAC power for sensing projection checks.
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


# OPTIONAL DIAGNOSTIC: compute scalar thermal power for sensing projection checks.
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


# OPTIONAL DIAGNOSTIC: compute scalar ADC power for sensing projection checks.
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


# OPTIONAL DIAGNOSTIC: compute target-wise sensing projection terms using MR combiners.
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

# OPTIONAL DIAGNOSTIC: compute sensing projection terms using stored V combiners.
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

# OPTIONAL DIAGNOSTIC: print sensing projection components target by target.
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

# Load and convert all MATLAB variables needed by the older one-setup runner.
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

# =============================================================================
# Pretty-printing and diagnostic report helpers
# =============================================================================
# These functions only format results for terminal reports; they do not change
# optimization states.
# Format floating-point values consistently in printed tables.
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


# Print a horizontal table rule.
def print_rule(width=118, char="-"):
    print(char * width)


# Print a centered section heading for terminal reports.
def print_heading(title, width=118):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


# Print key-value diagnostics in a compact table.
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


# Print the initial per-user power split and slack.
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


# Print per-user P_bar, P_tilda, P_prime, total power, and slack.
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


# OPTIONAL DIAGNOSTIC: print sensing coefficients, SINR, rates, and weights.
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



# Print target-wise sensing performance before and after optimization.
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


# Print the communication/sensing objective weights used by the run.
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

# Sum a power vector after converting it to a numeric array.
def _sum_power(x):
    return float(np.sum(np.asarray(x, dtype=float)))


# Prepare rows for objective contribution tables.
def _objective_row_values(comm_wsr, sensing_raw_no_ws, sensing_contribution, joint_obj, w_c, w_s):
    return [
        ("Communication", float(comm_wsr), float(w_c), float(w_c) * float(comm_wsr)),
        ("Sensing", float(sensing_raw_no_ws), float(w_s), float(sensing_contribution)),
        ("Joint", np.nan, np.nan, float(joint_obj)),
    ]


# Print raw communication/sensing values and weighted objective contributions.
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


# Print a compact before/after power summary for a result dictionary.
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


# Print communication components using stored result powers and kappas.
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


# Print sensing desired/interference/clutter/noise components.
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
# Print the full per-power report containing objective and component tables.
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

# =============================================================================
# Classical communication-only and sensing-only reference optimizers
# =============================================================================
# These reference modes optimize one subsystem while holding the other power
# blocks fixed. They are useful for comparisons and ablation studies.
# Classical communication-only baseline with P_prime fixed.
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
    epsilon=1e-4,
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

# Classical sensing-only baseline with P_bar and P_tilda fixed.
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
    epsilon=1e-4,
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


# Print a compact summary of one joint optimization result.
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




# Pad convergence histories to equal length for averaging/plotting.
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

# Load target weights omega_q from the .mat file, or default to ones.
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


# =============================================================================
# Older one-setup runner used by legacy sweep/training paths
# =============================================================================
# This runner supports classical joint, comm-only, sensing-only, and DU paths.
# The newer clean pipeline below is used by the current __main__ block.
# Legacy one-setup experiment runner supporting classical, DU, and ablation modes.
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
    epsilon=1e-4,
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

    if classical_mode not in ["joint", "comm_only", "sens_only", "comm_only_compare"]:
        raise ValueError(
            "classical_mode must be 'joint', 'comm_only', 'sens_only', "
            f"or 'comm_only_compare', got {classical_mode}"
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
    if classical_mode == "comm_only_compare":

        # ============================================================
        # Communication-only setup:
        #   P_prime = 0
        #   optimize only P_bar and P_tilda
        # ============================================================
        P_prime_zero = np.zeros(K, dtype=float)

        P_bar_comm_init = init_lambda * Pmax_lin * np.ones(K, dtype=float)
        P_tilda_comm_init = (1.0 - init_lambda) * Pmax_lin * np.ones(K, dtype=float)

        P_bar_comm_init = clip_P_bar_joint(
            P_bar_comm_init,
            P_tilda=P_tilda_comm_init,
            P_prime=P_prime_zero,
            P_total_max=P_total_max,
            eps=1e-12,
        )

        P_tilda_comm_init = clip_P_tilda_joint(
            P_tilda_comm_init,
            P_bar=P_bar_comm_init,
            P_prime=P_prime_zero,
            P_total_max=P_total_max,
            eps=1e-12,
        )

        # ============================================================
        # Communication-only unfolding history
        # ============================================================
        unfolding_comm_history = None

        if model is not None:
            # Communication-only DU uses unfolding_for_communication.py
            load_unfolding_api("comm_lambda")
            device = next(model.parameters()).device
            model = model.to(device).double()
            model.eval()

            with torch.no_grad():
                du_out = model(
                    P_bar_init=to_torch_1d(P_bar_comm_init, device),
                    P_tilda_init=to_torch_1d(P_tilda_comm_init, device),

                    w=to_torch_1d(w, device),
                    P_total_max=scalar_to_torch(P_total_max, device),
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

                    # Pure communication DU: sensing is disabled and P_prime remains exactly zero.

                    return_history=True,
                )

            unfolding_comm_history = (
                du_out["comm_history"]
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
            )

        # ============================================================
        # Classical vs lambda-QT vs unfolding comparison
        # ============================================================
        comparison = run_classical_lambda_qt_comparison(
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
            P_bar_init=P_bar_comm_init,
            P_tilda_init=P_tilda_comm_init,
            P_prime_fixed=P_prime_zero,
            P_total_max=P_total_max,

            max_iters=max_iters,
            epsilon=epsilon,
            eps_power=1e-12,
            eps_lambda=1e-30,
            lambda_mode_bar="actual",
            use_backtracking=True,

            print_step_sizes=True,
            step_print_every=50,
            step_print_last_n=10,
            step_size_save_path=f"lambda_step_sizes_Pmax_{Pmax_db:g}dB_setup_{setup_idx}.txt",
            
            du_history=unfolding_comm_history,
            save_plot_path=f"comm_only_compare_Pmax_{Pmax_db}_setup_{setup_idx}.png",
            verbose=verbose,

            # Required callbacks from this main file
            compute_sinr_exact_fn=compute_SINR_exact,
            compute_wsr_exact_fn=compute_WSR_exact,
            update_auxiliary_fn=update_auxiliary,
            classical_runner_fn=run_communication_only_optimisation,
        )

        classical_result = comparison["classical"]
        lambda_result = comparison["lambda_qt"]

        return {
            "mode": "comm_only_compare",
            "Pmax_db": Pmax_db,
            "setup_idx": setup_idx,

            "comparison": comparison,
            "classical_result": classical_result,
            "lambda_qt_result": lambda_result,
            "unfolding_comm_history": unfolding_comm_history,

            "P_bar_initial": P_bar_comm_init,
            "P_tilda_initial": P_tilda_comm_init,
            "P_prime_initial": P_prime_zero,

            "P_bar_classical_opt": classical_result["P_bar_opt"],
            "P_tilda_classical_opt": classical_result["P_tilda_opt"],

            "P_bar_lambda_opt": lambda_result["P_bar_opt"],
            "P_tilda_lambda_opt": lambda_result["P_tilda_opt"],

            "classical_full_comm_history": classical_result["full_comm_history"],
            "lambda_full_comm_history": lambda_result["full_comm_history"],

            "lambda_bar_history": lambda_result["lambda_bar_history"],
            "lambda_tilda_history": lambda_result["lambda_tilda_history"],
            "D_bar_aug_history": lambda_result["D_bar_aug_history"],
            "D_tilda_matrix_history": lambda_result["D_tilda_matrix_history"],

            "P_total_max": P_total_max,
            "Pmax_lin": Pmax_lin,
        }

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
        # Joint DU uses unfolding_for_jointopt_latest.py
        load_unfolding_api("joint")
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

# =============================================================================
# Legacy joint-DU training utilities
# =============================================================================
# These functions train/evaluate the older joint DU path. The clean-mode
# training functions later in the file are used by the current main block.
# Create shuffled mini-batches of setup indices for training.
def make_shuffled_batches(train_setups, batch_size, shuffle_batches=True, rng=None):
    """Create epoch-wise batches. If shuffle_batches=True, batches change every epoch."""
    setups = np.asarray(train_setups, dtype=int).copy()
    if shuffle_batches:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(setups)
    return [setups[i:i + batch_size] for i in range(0, len(setups), batch_size)]


# Legacy joint DU training loop over setup batches and power values.
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
    init_delta=0.0,
    init_lambda=0.50,
    w_c_user=1.0,
    w_s_user=0.0,
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
                    w_c_user=w_c_user,
                    w_s_user=w_s_user,
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


# Extract learned DU step sizes from a trained model.
def get_learned_step_sizes(model):
    """Return learned PGD step-size arrays for Pbar, Ptilda, and Pprime."""
    with torch.no_grad():
        return {
            "Pbar": model.step_Pbar.detach().cpu().numpy().reshape(-1),
            "Ptilda": model.step_Ptilda.detach().cpu().numpy().reshape(-1),
            "Pprime": model.step_Pprime.detach().cpu().numpy().reshape(-1),
        }


# Print learned step sizes and summary statistics.
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


# Plot classical and DU convergence for one power value.
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


# Plot EPA, classical, and DU objectives over the power sweep.
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


# Format a vector compactly for debug/report output.
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

# Legacy one-batch DU training step using the older one-setup runner.
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
    init_delta=0.0,
    init_lambda=0.5,
    w_c_user=1.0,
    w_s_user=0.0,
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
            w_c_user=w_c_user,
            w_s_user=w_s_user,
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

# =============================================================================
# Legacy power-sweep routines
# =============================================================================
# These sweep functions support older experiment flows. They are retained for
# reproducibility and comparison, but the current main block uses clean sweep.
# Legacy power sweep supporting older joint/comm/sensing experiment paths.
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
    init_delta=0.0,
    init_lambda=0.50,
    print_components=True,
    print_power_tables=False,
    run_sensing_only=False,
    run_deep_unfolding=False,
    w_c_user=1.0,
    w_s_user=0.0,
    classical_mode="comm_only_compare",
):
    """
    Sweep over Pmax and test setups.

    For classical_mode="comm_only_compare", this compares:
        1) Classical communication-only FP/QT
        2) D/lambda communication-only QT
        3) Communication-only deep unfolding, if model is provided

    Communication-only setting:
        P_prime = 0
        w_c = 1
        w_s = 0

    For other classical_mode values, this falls back to the older joint-style
    result handling.
    """

    initial_joint_by_power = []
    joint_obj_by_power = []
    comm_wsr_by_power = []
    sensing_wsr_by_power = []
    sensing_only_wsr_by_power = []

    joint_history_by_power = []

    # New lambda-QT sweep storage
    lambda_qt_by_power = []
    lambda_history_by_power = []
    du_by_power = []
    du_history_by_power = []

    sensing_sinr_by_power = []
    sensing_sinr_db_by_power = []

    du_joint_by_power = []
    du_comm_by_power = []
    du_sensing_by_power = []
    du_history_by_power = []

    all_results = {}

    for Pdb in power_list:
        Pdb = float(Pdb)

        initial_vals = []
        joint_vals = []
        comm_vals = []
        sensing_vals = []
        sensing_only_vals = []

        joint_histories = []

        # New lambda-QT setup-level storage
        lambda_vals = []
        lambda_histories = []
        du_vals = []
        du_histories = []

        convergence_iters = []
        elapsed_times = []
        converged_flags = []

        du_joint_vals = []
        du_comm_vals = []
        du_sensing_vals = []
        du_histories = []

        sensing_sinr_vals = []

        all_results[Pdb] = []

        for setup_idx in test_setups:
            setup_idx = int(setup_idx)

            # ------------------------------------------------------------
            # Run one setup
            # ------------------------------------------------------------
            result = run_one_setup_power(
                mat_file=mat_file,
                Pmax_db=Pdb,
                model=model,
                train_unfolding=False,
                run_classical=True,
                run_deep_unfolding=run_deep_unfolding,

                setup_idx=setup_idx,
                K_use=K_use,
                M_use=M_use,
                N_use=N_use,
                sigma_w2=sigma_w2,

                max_iters=max_iters,
                epsilon=epsilon,
                verbose=False,
                check_sensing=check_sensing,
                print_sensing=print_sensing,
                print_components=print_components,
                print_power_tables=print_power_tables,

                # For communication-only comparison, sensing is disabled.
                # The comm_only_compare block inside run_one_setup_power
                # uses P_prime_zero anyway, but keep this consistent.
                init_delta=0.0 if classical_mode == "comm_only_compare" else init_delta,
                init_lambda=init_lambda,

                classical_mode=classical_mode,
                run_sensing_only=False if classical_mode == "comm_only_compare" else run_sensing_only,

                w_c_user=1.0 if classical_mode == "comm_only_compare" else w_c_user,
                w_s_user=0.0 if classical_mode == "comm_only_compare" else w_s_user,
            )

            # ============================================================
            # Case 1: communication-only comparison
            # ============================================================
            if result.get("mode") == "comm_only_compare":
                classical_result = result["classical_result"]
                lambda_result = result["lambda_qt_result"]
                du_hist = result["unfolding_comm_history"]

                if du_hist is not None:
                    print(
                        f"[COMM-ONLY COMPARE] Pmax={Pdb:>6.1f} dB | "
                        f"setup={setup_idx:02d} | "
                        f"classical={classical_result['final_comm_wsr']:.6f} | "
                        f"lambda-QT={lambda_result['final_comm_wsr']:.6f} | "
                        f"DU={du_hist[-1]:.6f}"
                    )
                else:
                    print(
                        f"[COMM-ONLY COMPARE] Pmax={Pdb:>6.1f} dB | "
                        f"setup={setup_idx:02d} | "
                        f"classical={classical_result['final_comm_wsr']:.6f} | "
                        f"lambda-QT={lambda_result['final_comm_wsr']:.6f} | "
                        f"DU=N/A"
                    )

                # Existing list names are reused:
                # joint_vals = classical communication-only final WSR
                initial_vals.append(classical_result["initial_comm_wsr"])
                joint_vals.append(classical_result["final_comm_wsr"])
                comm_vals.append(classical_result["final_comm_wsr"])
                sensing_vals.append(0.0)
                sensing_only_vals.append(np.nan)

                joint_histories.append(classical_result["full_comm_history"])

                # New lambda-QT storage
                lambda_vals.append(lambda_result["final_comm_wsr"])
                lambda_histories.append(lambda_result["full_comm_history"])

                convergence_iters.append(classical_result["iterations"])
                elapsed_times.append(0.0)
                converged_flags.append(classical_result["monotonic"])

                if du_hist is not None:
                    du_joint_vals.append(du_hist[-1])
                    du_comm_vals.append(du_hist[-1])
                    du_sensing_vals.append(0.0)
                    du_histories.append(np.asarray(du_hist, dtype=float))
                else:
                    du_joint_vals.append(np.nan)
                    du_comm_vals.append(np.nan)
                    du_sensing_vals.append(np.nan)
                    du_histories.append(np.array([]))

                all_results[Pdb].append(result)
                continue

            # ============================================================
            # Case 2: normal joint/sensing/other classical modes
            # ============================================================
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

            sensing = result.get("sensing_projection", None)
            if sensing is not None:
                sensing_sinr_vals.append(sensing["SINR"])

            initial_vals.append(result["initial_joint_wsr"])
            joint_vals.append(result["final_wsr"])
            comm_vals.append(result["final_comm_wsr"])
            sensing_vals.append(result["final_sensing_wsr"])
            sensing_only_vals.append(sens_only_final)

            joint_histories.append(result["full_joint_history"])

            # No lambda-QT result in normal joint mode.
            lambda_vals.append(np.nan)
            lambda_histories.append(np.array([]))

            convergence_iters.append(result["convergence_iter"])
            elapsed_times.append(result["elapsed_time_sec"])
            converged_flags.append(result["converged"])

            du_joint_vals.append(result["deep_final_joint_wsr"])
            du_comm_vals.append(result["deep_final_comm_wsr"])
            du_sensing_vals.append(result["deep_final_sensing_wsr"])
            du_histories.append(result["deep_joint_history"])

            all_results[Pdb].append(result)

        # ================================================================
        # Power-level averaging
        # ================================================================
        initial_mean = float(np.mean(initial_vals)) if initial_vals else np.nan
        joint_mean = float(np.mean(joint_vals)) if joint_vals else np.nan
        comm_mean = float(np.mean(comm_vals)) if comm_vals else np.nan
        sensing_mean = float(np.mean(sensing_vals)) if sensing_vals else np.nan

        sensing_only_arr = np.asarray(sensing_only_vals, dtype=float)
        if sensing_only_arr.size > 0 and np.any(np.isfinite(sensing_only_arr)):
            sensing_only_mean = float(np.nanmean(sensing_only_arr))
        else:
            sensing_only_mean = np.nan

        joint_mean_history = (
            pad_with_last_value(joint_histories).mean(axis=0)
            if joint_histories
            else np.array([])
        )

        # New lambda-QT power-level average
        lambda_arr = np.asarray(lambda_vals, dtype=float)
        if lambda_arr.size > 0 and np.any(np.isfinite(lambda_arr)):
            lambda_mean = float(np.nanmean(lambda_arr))
        else:
            lambda_mean = np.nan

        valid_lambda_histories = [
            np.asarray(h, dtype=float).reshape(-1)
            for h in lambda_histories
            if len(np.asarray(h).reshape(-1)) > 0
        ]

        lambda_mean_history = (
            pad_with_last_value(valid_lambda_histories).mean(axis=0)
            if valid_lambda_histories
            else np.array([])
        )

        mean_conv_iter = float(np.mean(convergence_iters)) if convergence_iters else np.nan
        mean_elapsed_time = float(np.mean(elapsed_times)) if elapsed_times else np.nan
        num_converged = int(np.sum(converged_flags)) if converged_flags else 0

        finite_du = np.asarray(du_joint_vals, dtype=float)
        if finite_du.size > 0 and np.any(np.isfinite(finite_du)):
            du_joint_mean = float(np.nanmean(du_joint_vals))
            du_comm_mean = float(np.nanmean(du_comm_vals))
            du_sensing_mean = float(np.nanmean(du_sensing_vals))

            valid_du_histories = [
                np.asarray(h, dtype=float).reshape(-1)
                for h in du_histories
                if len(np.asarray(h).reshape(-1)) > 0
            ]

            du_mean_history = (
                pad_with_last_value(valid_du_histories).mean(axis=0)
                if valid_du_histories
                else np.array([])
            )
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

        lambda_qt_by_power.append(lambda_mean)
        lambda_history_by_power.append(lambda_mean_history)

        du_joint_by_power.append(du_joint_mean)
        du_comm_by_power.append(du_comm_mean)
        du_sensing_by_power.append(du_sensing_mean)
        du_history_by_power.append(du_mean_history)

        print(
            f"[POWER CONVERGENCE SUMMARY] P={Pdb:>6.1f} dB | "
            f"mean iters={mean_conv_iter:.2f}/{max_iters} | "
            f"mean time={mean_elapsed_time:.3f} sec | "
            f"converged setups={num_converged}/{len(test_setups)} | "
            f"mean classical={joint_mean:.6f} | "
            f"mean lambda-QT={lambda_mean:.6f}"
        )

        if np.isfinite(du_joint_mean):
            print(
                f"[POWER SUMMARY] P={Pdb:>6.1f} dB | "
                f"initial={initial_mean:.6f} | "
                f"classical={joint_mean:.6f} | "
                f"lambda-QT={lambda_mean:.6f} | "
                f"DU={du_joint_mean:.6f} | "
                f"DU-classical={du_joint_mean - joint_mean:.6f}"
            )
        else:
            print(
                f"[POWER SUMMARY] P={Pdb:>6.1f} dB | "
                f"initial={initial_mean:.6f} | "
                f"classical={joint_mean:.6f} | "
                f"lambda-QT={lambda_mean:.6f} | "
                f"DU=N/A"
            )

        if sensing_sinr_vals:
            sensing_sinr_mean = np.mean(np.stack(sensing_sinr_vals, axis=0), axis=0)
            sensing_sinr_by_power.append(sensing_sinr_mean)
            sensing_sinr_db_by_power.append(lin2db(sensing_sinr_mean))
        else:
            sensing_sinr_by_power.append(None)
            sensing_sinr_db_by_power.append(None)

        # ================================================================
        # Per-power convergence plot
        # ================================================================
        if save_plots:
            plt.figure(figsize=(7.5, 5.2))

            if len(joint_mean_history) > 0:
                plt.plot(
                    np.arange(len(joint_mean_history)),
                    joint_mean_history,
                    marker="o",
                    linewidth=2,
                    label="Classical FP/QT",
                )

            if len(lambda_mean_history) > 0:
                plt.plot(
                    np.arange(len(lambda_mean_history)),
                    lambda_mean_history,
                    marker="s",
                    linewidth=2,
                    label="D/lambda QT",
                )

            if len(du_mean_history) > 0 and np.any(np.isfinite(du_mean_history)):
                plt.plot(
                    np.arange(len(du_mean_history)),
                    du_mean_history,
                    marker="^",
                    linewidth=2,
                    label="Deep unfolding",
                )

            plt.xlabel("Iteration / Layer")
            if classical_mode == "comm_only_compare":
                plt.ylabel("Mean communication WSR")
                plt.title(f"Communication-only convergence @ {Pdb:g} dB")
            else:
                plt.ylabel("Mean joint objective")
                plt.title(f"Joint convergence @ {Pdb:g} dB")

            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"convergence_classical_lambda_du_{Pdb:g}dB.png", dpi=300)
            plt.show()

    # ====================================================================
    # Convert power-level lists to arrays
    # ====================================================================
    initial_joint_by_power = np.asarray(initial_joint_by_power, dtype=float)
    joint_obj_by_power = np.asarray(joint_obj_by_power, dtype=float)
    comm_wsr_by_power = np.asarray(comm_wsr_by_power, dtype=float)
    sensing_wsr_by_power = np.asarray(sensing_wsr_by_power, dtype=float)
    sensing_only_wsr_by_power = np.asarray(sensing_only_wsr_by_power, dtype=float)

    lambda_qt_by_power = np.asarray(lambda_qt_by_power, dtype=float)

    du_joint_by_power = np.asarray(du_joint_by_power, dtype=float)
    du_comm_by_power = np.asarray(du_comm_by_power, dtype=float)
    du_sensing_by_power = np.asarray(du_sensing_by_power, dtype=float)

    power_list_np = np.asarray(power_list, dtype=float)

    # ====================================================================
    # Power sweep plot
    # ====================================================================
    if save_plots:
        plt.figure(figsize=(8, 6))

        plt.plot(
            power_list_np,
            initial_joint_by_power,
            "--s",
            linewidth=2,
            label="EPA before optimization",
        )

        plt.plot(
            power_list_np,
            joint_obj_by_power,
            "-o",
            linewidth=2,
            label="Classical FP/QT",
        )

        if np.any(np.isfinite(lambda_qt_by_power)):
            plt.plot(
                power_list_np,
                lambda_qt_by_power,
                "-d",
                linewidth=2,
                label="D/lambda QT",
            )

        if np.any(np.isfinite(du_joint_by_power)):
            plt.plot(
                power_list_np,
                du_joint_by_power,
                "-^",
                linewidth=2,
                label="Deep unfolding",
            )

        plt.xlabel("Transmit Power $P_{max}$ (dB)")

        if classical_mode == "comm_only_compare":
            plt.ylabel("Mean communication WSR")
            plt.title("Power Sweep: EPA vs Classical vs D/lambda QT vs DU")
            plt.savefig("power_sweep_comm_only_classical_lambda_du.png", dpi=300)
        else:
            plt.ylabel("Mean joint objective")
            plt.title("Power Sweep: EPA vs Classical vs D/lambda QT vs DU")
            plt.savefig("power_sweep_joint_classical_lambda_du.png", dpi=300)

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Sensing plot only makes sense for joint/sensing modes.
        if classical_mode != "comm_only_compare":
            if any(x is not None for x in sensing_sinr_db_by_power):
                sensing_db_mat = np.array(
                    [x for x in sensing_sinr_db_by_power if x is not None]
                )
                power_valid = np.array(
                    [p for p, x in zip(power_list_np, sensing_sinr_db_by_power) if x is not None]
                )

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
        "power_list": power_list_np,

        # Existing sweep outputs
        "initial_joint_by_power": initial_joint_by_power,
        "joint_obj_by_power": joint_obj_by_power,
        "comm_wsr_by_power": comm_wsr_by_power,
        "sensing_wsr_by_power": sensing_wsr_by_power,
        "sensing_only_wsr_by_power": sensing_only_wsr_by_power,
        "joint_history_by_power": joint_history_by_power,

        # New lambda-QT outputs
        "lambda_qt_by_power": lambda_qt_by_power,
        "lambda_history_by_power": lambda_history_by_power,

        # Sensing outputs
        "sensing_sinr_by_power": sensing_sinr_by_power,
        "sensing_sinr_db_by_power": sensing_sinr_db_by_power,

        # DU outputs
        "du_joint_by_power": du_joint_by_power,
        "du_comm_by_power": du_comm_by_power,
        "du_sensing_by_power": du_sensing_by_power,
        "du_history_by_power": du_history_by_power,

        "all_results": all_results,

        # Backward-compatible aliases for older plotting scripts.
        # For comm_only_compare, these are communication WSR values.
        "before_wsr_by_power": initial_joint_by_power,
        "classical_wsr_by_power": joint_obj_by_power,
        "classical_history_by_power": joint_history_by_power,

        # New aliases for clean comparison.
        "lambda_wsr_by_power": lambda_qt_by_power,
        "lambda_qt_history_by_power": lambda_history_by_power,
        "du_wsr_by_power": du_joint_by_power,
    }

# Backward-compatible alias. Prefer run_joint_power_sweep(...).
# OPTIONAL LEGACY WRAPPER: run the old classical power sweep path.
def run_classical_power_sweep(*args, **kwargs):
    return run_joint_power_sweep(*args, **kwargs)


# ============================================================================
# CLEAN MODE API ADDED FOR SEPARATING COMM-ONLY/LAMBDA AND JOINT COMM-SENSING
# ============================================================================
# Main rule:
#   - comm/lambda mode uses the communication-only dataset and Code-B matching
#     preprocessing/reference powers.
#   - joint mode uses the joint communication-sensing dataset and builds sensing
#     coefficients only there.
#   - communication kappas are built inside each top-level clean runner.


# =============================================================================
# Clean-mode dataset loading and communication context construction
# =============================================================================
# The clean pipeline builds a reusable context containing matrices, kappas,
# sensing coefficients, and initial powers for one setup and power value.
# Clean-mode loader for selecting setup-specific MATLAB variables.
def load_matlab_direct_values_clean(
    mat_file,
    setup_idx=0,
    K_use=8,
    M_use=16,
    N_use=16,
    hermitian_comm=True,
    load_sensing=False,
):
    """
    Clean loader.

    For communication-only matching with Code B, call with:
        hermitian_comm=True, load_sensing=False

    For joint communication-sensing, call with:
        hermitian_comm=True, load_sensing=True

    Sensing variables are optional in the loader and are not required for
    communication-only runs.
    """
    mat = loadmat(mat_file)

    A_l_all, A_name = get_mat_var(mat, ["A_l_allsetups", "A_l_all"])
    C_h_all_CF, C_h_all_name = get_mat_var(mat, ["C_h_all_CF_allsetups", "C_h_all_CF"])
    C_h_dir_all_CF, C_h_dir_name = get_mat_var(mat, ["C_h_dir_all_CF_allsetups", "C_h_dir_all_CF"])
    C_h_ref_all_CF, C_h_ref_name = get_mat_var(mat, ["C_h_ref_all_CF_allsetups", "C_h_ref_all_CF"])
    Z_matrices, Z_name = get_mat_var(mat, ["Z_matrices_allsetups", "Z_matrices"])
    alpha_dac_all, _ = get_mat_var(mat, ["alpha_dac_allsetups", "alpha_dac_all"])

    R_mat, R_name = get_mat_var(mat, ["R_allsetups", "R"], required=False)
    h_bar_mat, hbar_name = get_mat_var(mat, ["h_bar_allsetups", "h_bar", "h_bar_max"], required=False)

    if R_mat is None or h_bar_mat is None:
        raise KeyError(
            "Dataset is missing R and/or h_bar. Communication kappas require both."
        )

    # Select setup if a setup dimension exists.
    A_l_all = take_setup_last_dim(A_l_all, setup_idx, 3, A_name)
    C_h_all_CF = take_setup_last_dim(C_h_all_CF, setup_idx, 4, C_h_all_name)
    C_h_dir_all_CF = take_setup_last_dim(C_h_dir_all_CF, setup_idx, 4, C_h_dir_name)
    C_h_ref_all_CF = take_setup_last_dim(C_h_ref_all_CF, setup_idx, 4, C_h_ref_name)
    Z_matrices = take_setup_last_dim(Z_matrices, setup_idx, 3, Z_name)
    R_mat = take_setup_last_dim(R_mat, setup_idx, 4, R_name)
    h_bar_mat = take_setup_last_dim(h_bar_mat, setup_idx, 3, hbar_name)

    N = N_use
    M = M_use
    K = K_use

    tau = Z_matrices.shape[0]
    d = Z_matrices.shape[1]

    A_l_all = A_l_all[:N, :N, :M]
    C_h_all_CF = C_h_all_CF[:N, :N, :K, :M]
    C_h_dir_all_CF = C_h_dir_all_CF[:N, :N, :K, :M]
    C_h_ref_all_CF = C_h_ref_all_CF[:N, :N, :K, :M]
    Z_matrices = Z_matrices[:, :, :K]
    R_mat = R_mat[:N, :N, :K, :M]
    h_bar_mat = h_bar_mat[:N, :K, :M]

    alpha_dac_all = np.asarray(alpha_dac_all)
    if alpha_dac_all.ndim == 2 and alpha_dac_all.shape[1] > 1:
        alpha_list = np.real(alpha_dac_all[:K, setup_idx]).reshape(-1)
    else:
        alpha_list = np.real(alpha_dac_all).reshape(-1)[:K]

    A_list = mat_3d_ap_to_list(A_l_all, M, hermitian=hermitian_comm)
    h_bar_list = hbar_to_lk_list(h_bar_mat, K, M)
    R_list = mat_4d_to_lk_list(R_mat, K, M, hermitian=hermitian_comm)
    Sigma_list = mat_4d_to_lk_list(C_h_dir_all_CF, K, M, hermitian=hermitian_comm)
    C_h_ref_all = mat_4d_to_lk_list(C_h_ref_all_CF, K, M, hermitian=hermitian_comm)
    C_h_total_all = mat_4d_to_lk_list(C_h_all_CF, K, M, hermitian=hermitian_comm)
    Z_list = Z_to_list(Z_matrices, K)

    out = {
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
        "Z_list": Z_list,
        "A_l_all_raw": A_l_all,
        "C_h_all_CF_raw": C_h_all_CF,
        "C_h_ref_all_CF_raw": C_h_ref_all_CF,
    }

    if load_sensing:
        alpha_var_matrix_all, alpha_var_name = get_mat_var(
            mat,
            ["alpha_var_matrix_allsetups", "alpha_var_matrix_all"],
            required=False,
        )
        b_target_all, b_target_name = get_mat_var(
            mat,
            ["b_target_allsetups", "b_target_all"],
            required=False,
        )
        H_direct_iter, H_direct_name = get_mat_var(
            mat,
            ["H_direct_iter_allsetups", "H_direct_iter"],
            required=False,
        )
        H_true_iter, H_true_name = get_mat_var(
            mat,
            ["H_true_iter_allsetups", "H_true_iter"],
            required=False,
        )
        V_combiner_all, V_combiner_name = get_mat_var(
            mat,
            ["V_combiner_all", "V_combiner_allsetups"],
            required=False,
        )

        if b_target_all is not None:
            b_target_all = take_setup_last_dim(b_target_all, setup_idx, 3, b_target_name)
            b_target_all = np.asarray(b_target_all[:N, :M, :], dtype=complex)

        if H_direct_iter is not None:
            H_direct_iter = take_setup_last_dim(H_direct_iter, setup_idx, 4, H_direct_name)
            H_direct_iter = np.asarray(H_direct_iter[:N, :K, :M, :], dtype=complex)

        if H_true_iter is not None:
            H_true_iter = take_setup_last_dim(H_true_iter, setup_idx, 4, H_true_name)
            H_true_iter = np.asarray(H_true_iter[:N, :K, :M, :], dtype=complex)

        if V_combiner_all is not None:
            V_combiner_all = take_setup_last_dim(V_combiner_all, setup_idx, 4, V_combiner_name)
            V_combiner_all = np.asarray(V_combiner_all[:N, :M, :, :], dtype=complex)

        if alpha_var_matrix_all is not None:
            alpha_var_matrix_all = take_setup_last_dim(alpha_var_matrix_all, setup_idx, 3, alpha_var_name)
            alpha_var_matrix_all = np.asarray(alpha_var_matrix_all[:K, :M, :], dtype=float)

        out.update({
            "alpha_var_matrix_all": alpha_var_matrix_all,
            "b_target_all": b_target_all,
            "H_direct_iter": H_direct_iter,
            "H_true_iter": H_true_iter,
            "V_combiner_all": V_combiner_all,
        })

    return out


# Build the complete reusable clean-mode problem context for one setup and power.
def build_communication_problem_context(
    mat_file,
    Pmax_db,
    setup_idx=0,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    reference_mode="comm_only",
    hermitian_comm=True,
    load_sensing=False,
):
    """
    Build all communication matrices and kappas inside the top-level runner.

    reference_mode='comm_only' matches Code B:
        P_bar_ref = 0.5 * Pmax
        P_tot_list = Pmax
        hermitian_comm=True

    reference_mode='joint' keeps the joint hardware/reference split:
        P_bar_ref = 0.25 * Pmax
        P_prime_ref = 0.5 * Pmax
        P_tot_list = Pmax
    """
    data = load_matlab_direct_values_clean(
        mat_file=mat_file,
        setup_idx=setup_idx,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        hermitian_comm=hermitian_comm,
        load_sensing=load_sensing,
    )

    K = data["K"]
    M = data["M"]
    N = data["N"]
    tau = data["tau"]
    d = data["d"]
    alpha_list = data["alpha_list"]
    A_list = data["A_list"]
    R_list = data["R_list"]
    h_bar_list = data["h_bar_list"]
    Sigma_list = data["Sigma_list"]
    C_h_ref_all = data["C_h_ref_all"]
    C_h_total_all = data["C_h_total_all"]
    Z_list = data["Z_list"]

    Pmax_lin = db2lin(Pmax_db)
    P_total_max = Pmax_lin

    if reference_mode == "comm_only":
        P_bar_ref = 0.5 * Pmax_lin * np.ones(K)
        P_tilda_ref = 0.5 * Pmax_lin * np.ones(K)
        P_prime_ref = np.zeros(K)
        P_tot_list = Pmax_lin * np.ones(K)
    elif reference_mode == "joint":
        P_comm_ref_hw = 0.5 * Pmax_lin
        P_sense_ref_hw = 0.5 * Pmax_lin
        P_bar_ref = 0.5 * P_comm_ref_hw * np.ones(K)
        P_tilda_ref = 0.5 * P_comm_ref_hw * np.ones(K)
        P_prime_ref = P_sense_ref_hw * np.ones(K)
        P_tot_list = P_bar_ref + P_tilda_ref + P_prime_ref
    else:
        raise ValueError("reference_mode must be 'comm_only' or 'joint'.")

    sigma_d2 = alpha_list * (1.0 - alpha_list) * P_tot_list
    W_list = build_W_list_MR(M, K, N)

    Theta_list = build_theta_list_for_power(
        A_list=A_list,
        C_h_total_all=C_h_total_all,
        alpha_list=alpha_list,
        P_tot_list=P_tot_list,
        sigma_w2=sigma_w2,
    )

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
        R_list,
        h_bar_list,
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

    kappas = {
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
    }

    data.update({
        "Pmax_db": Pmax_db,
        "Pmax_lin": Pmax_lin,
        "P_total_max": P_total_max,
        "reference_mode": reference_mode,
        "P_bar_ref": P_bar_ref,
        "P_tilda_ref": P_tilda_ref,
        "P_prime_ref": P_prime_ref,
        "P_tot_list": P_tot_list,
        "sigma_d2": sigma_d2,
        "sigma_w2": sigma_w2,
        "W_list": W_list,
        "Theta_list": Theta_list,
        "R_tilda_all": R_tilda_all,
        "C_yy_all": C_yy_all,
        "G_list": G_list,
        "U_list": U_list,
        "V_list": V_list,
        "U_q_list": U_q_list,
        "V_q_list": V_q_list,
        "D_Th_list": D_Th_list,
        "D_ADC_list": D_ADC_list,
        "g_lik": g_lik,
        "kappas": kappas,
    })

    return data


# =============================================================================
# Clean-mode classical/lambda single-setup runners
# =============================================================================
# These functions run communication-only FP/QT, communication lambda-QT,
# joint classical, and joint lambda-QT from a common prepared context.
# Run communication-only FP/QT using already-built kappas from the context.
def run_communication_only_given_kappas(
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
    P_prime_fixed=None,
    P_total_max=None,
    max_iters=200,
    epsilon=1e-4,
    eps_power=1e-12,
    verbose=False,
):
    """
    Classical communication-only optimizer with the same update order as Code B.

    This is intentionally kappa-based because the lambda comparison wrapper needs
    this callback format. The public clean runner below builds the kappas before
    calling this callback.
    """
    P_bar = np.asarray(P_bar_init, dtype=float).reshape(K).copy()
    P_tilda = np.asarray(P_tilda_init, dtype=float).reshape(K).copy()
    w = np.asarray(w, dtype=float).reshape(K)

    if P_prime_fixed is None:
        P_prime_fixed = np.zeros(K, dtype=float)
    else:
        P_prime_fixed = np.asarray(P_prime_fixed, dtype=float).reshape(K)

    P_th_initial = P_bar * kappa_Th1 + kappa_Th0
    P_adc_initial = P_bar * kappa_ADC1 + kappa_ADC0
    initial_comm_wsr = compute_WSR_exact(
        K, w,
        kappa_S,
        kappa_V1, kappa_K1,
        kappa_V0, kappa_K0,
        kappa_DAC1, kappa_DAC0,
        kappa_M1, kappa_M0,
        kappa_Th1, kappa_ADC1,
        P_bar, P_tilda,
        P_th_initial, P_adc_initial,
        d, tau,
    )

    comm_history = []
    wsr_prev = -np.inf
    convergence_iter = max_iters

    for t in range(max_iters):
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

        mu = update_auxiliary(
            K, w, gamma,
            kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1, kappa_ADC1,
            P_bar, P_tilda,
            P_th, P_adc,
        )

        P_bar = compute_P_bar(
            K, P_tilda, w,
            kappa_S, kappa_V1, kappa_K1,
            kappa_M1, kappa_Th1,
            kappa_DAC1, kappa_ADC1,
            Z_list,
            gamma, mu,
        )

        if np.allclose(P_prime_fixed, 0.0):
            P_bar = clip_P_bar_given_P_tilda(P_bar, P_tilda, P_total_max, eps_power)
        else:
            P_bar = clip_P_bar_joint(P_bar, P_tilda, P_prime_fixed, P_total_max, eps_power)

        P_tilda = compute_P_tilda(
            K, P_bar, w,
            kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_M1, kappa_M0,
            gamma, mu,
        )

        if np.allclose(P_prime_fixed, 0.0):
            P_tilda = clip_P_tilda_given_P_bar(P_tilda, P_bar, P_total_max, eps_power)
        else:
            P_tilda = clip_P_tilda_joint(P_tilda, P_bar, P_prime_fixed, P_total_max, eps_power)

        P_th = P_bar * kappa_Th1 + kappa_Th0
        P_adc = P_bar * kappa_ADC1 + kappa_ADC0

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

        comm_history.append(float(comm_wsr))
        delta = float(comm_wsr - wsr_prev)

        if t > 0 and delta < -1e-10:
            print(f"Warning: communication-only WSR decreased: {delta:.3e}")

        if verbose:
            print(f"comm-only iter={t+1:03d} | comm={comm_wsr:.6f} | delta={delta:.3e}")

        if t > 0 and abs(delta) < epsilon:
            convergence_iter = t + 1
            break

        wsr_prev = float(comm_wsr)

    if not comm_history:
        comm_history = [float(initial_comm_wsr)]

    P_th_opt = P_bar * kappa_Th1 + kappa_Th0
    P_adc_opt = P_bar * kappa_ADC1 + kappa_ADC0
    gamma_opt = np.array([
        compute_SINR_exact(
            k, K,
            kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1, kappa_ADC1,
            P_bar, P_tilda,
            P_th_opt, P_adc_opt,
        )
        for k in range(K)
    ])

    full_comm_history = np.concatenate([
        [float(initial_comm_wsr)],
        np.asarray(comm_history, dtype=float),
    ])

    return {
        "initial_comm_wsr": float(initial_comm_wsr),
        "final_comm_wsr": float(comm_history[-1]),
        "comm_history": np.asarray(comm_history, dtype=float),
        "full_comm_history": full_comm_history,
        "P_bar_initial": np.asarray(P_bar_init, dtype=float).reshape(K),
        "P_tilda_initial": np.asarray(P_tilda_init, dtype=float).reshape(K),
        "P_prime_fixed": P_prime_fixed,
        "P_bar_opt": P_bar,
        "P_tilda_opt": P_tilda,
        "P_prime_opt": P_prime_fixed,
        "P_th_opt": P_th_opt,
        "P_adc_opt": P_adc_opt,
        "gamma_opt": gamma_opt,
        "iterations": len(comm_history),
        "convergence_iter": convergence_iter,
        "monotonic": bool(np.all(np.diff(full_comm_history) >= -1e-10)),
    }


# Run communication lambda-QT for one clean-mode setup.
def run_communication_lambda_single_setup(
    comm_mat_file,
    Pmax_db,
    setup_idx=0,
    model=None,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-4,
    init_lambda=0.5,
    verbose=True,
    save_plot_path=None,
):
    """
    Clean communication-only + D/lambda-QT run.

    This uses the communication-only dataset and matches Code B settings:
        - hermitian_comm=True
        - P_bar_ref = 0.5 Pmax
        - P_tot_list = Pmax
        - P_prime = 0
        - P_bar_init = init_lambda Pmax
        - P_tilda_init = (1-init_lambda) Pmax
    """
    ctx = build_communication_problem_context(
        mat_file=comm_mat_file,
        Pmax_db=Pmax_db,
        setup_idx=setup_idx,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        sigma_w2=sigma_w2,
        reference_mode="comm_only",
        hermitian_comm=True,
        load_sensing=False,
    )

    K = ctx["K"]
    w = np.ones(K)
    Pmax_lin = ctx["Pmax_lin"]
    P_total_max = ctx["P_total_max"]
    P_prime_zero = np.zeros(K, dtype=float)
    P_bar_init = float(init_lambda) * Pmax_lin * np.ones(K, dtype=float)
    P_tilda_init = (1.0 - float(init_lambda)) * Pmax_lin * np.ones(K, dtype=float)

    k = ctx["kappas"]

    # Optional communication-only DU history. The pure comm model uses no sensing tensors.
    unfolding_comm_history = None
    if model is not None:
        # Communication-only DU uses unfolding_for_communication.py
        load_unfolding_api("comm_lambda")
        device = next(model.parameters()).device
        model = model.to(device).double()
        model.eval()
        with torch.no_grad():
            du_out = model(
                P_bar_init=to_torch_1d(P_bar_init, device),
                P_tilda_init=to_torch_1d(P_tilda_init, device),
                w=to_torch_1d(w, device),
                P_total_max=scalar_to_torch(P_total_max, device),
                d=ctx["d"],
                tau=ctx["tau"],
                kappa_S=to_torch_1d(k["kappa_S"], device),
                kappa_V1=to_torch_1d(k["kappa_V1"], device),
                kappa_K1=to_torch_1d(k["kappa_K1"], device),
                kappa_V0=to_torch_1d(k["kappa_V0"], device),
                kappa_K0=to_torch_1d(k["kappa_K0"], device),
                kappa_DAC1=to_torch_1d(k["kappa_DAC1"], device),
                kappa_DAC0=to_torch_1d(k["kappa_DAC0"], device),
                kappa_Th1=to_torch_1d(k["kappa_Th1"], device),
                kappa_ADC1=to_torch_1d(k["kappa_ADC1"], device),
                kappa_Th0=to_torch_1d(k["kappa_Th0"], device),
                kappa_ADC0=to_torch_1d(k["kappa_ADC0"], device),
                kappa_M1=to_torch_2d(k["kappa_M1"], device),
                kappa_M0=to_torch_2d(k["kappa_M0"], device),
                return_history=True,
            )

        unfolding_comm_history = du_out["comm_history"].mean(dim=1).detach().cpu().numpy()

    comparison = run_classical_lambda_qt_comparison(
        K=K,
        w=w,
        d=ctx["d"],
        tau=ctx["tau"],
        kappa_S=k["kappa_S"],
        kappa_V1=k["kappa_V1"],
        kappa_K1=k["kappa_K1"],
        kappa_V0=k["kappa_V0"],
        kappa_K0=k["kappa_K0"],
        kappa_DAC1=k["kappa_DAC1"],
        kappa_DAC0=k["kappa_DAC0"],
        kappa_Th1=k["kappa_Th1"],
        kappa_ADC1=k["kappa_ADC1"],
        kappa_M1=k["kappa_M1"],
        kappa_M0=k["kappa_M0"],
        kappa_Th0=k["kappa_Th0"],
        kappa_ADC0=k["kappa_ADC0"],
        Z_list=ctx["Z_list"],
        P_bar_init=P_bar_init,
        P_tilda_init=P_tilda_init,
        P_prime_fixed=P_prime_zero,
        P_total_max=P_total_max,
        max_iters=max_iters,
        epsilon=epsilon,
        eps_power=1e-12,
        eps_lambda=1e-30,
        lambda_mode_bar="actual",
        use_backtracking=True,
        print_step_sizes=True,
        step_print_every=50,
        step_print_last_n=10,
        step_size_save_path=f"lambda_step_sizes_Pmax_{Pmax_db:g}dB_setup_{setup_idx}.txt",
        du_history=unfolding_comm_history,
        save_plot_path=save_plot_path,
        verbose=verbose,
        compute_sinr_exact_fn=compute_SINR_exact,
        compute_wsr_exact_fn=compute_WSR_exact,
        update_auxiliary_fn=update_auxiliary,
        classical_runner_fn=run_communication_only_given_kappas,
    )

    classical_result = comparison["classical"]
    lambda_result = comparison["lambda_qt"]

    return {
        "mode": "comm_lambda",
        "Pmax_db": Pmax_db,
        "setup_idx": setup_idx,
        "comm_mat_file": comm_mat_file,
        "context": ctx,
        "comparison": comparison,
        "classical_result": classical_result,
        "lambda_qt_result": lambda_result,
        "unfolding_comm_history": unfolding_comm_history,
        "du_final_wsr": (float(np.asarray(unfolding_comm_history).reshape(-1)[-1]) if unfolding_comm_history is not None and len(np.asarray(unfolding_comm_history).reshape(-1)) > 0 else np.nan),
        "P_bar_initial": P_bar_init,
        "P_tilda_initial": P_tilda_init,
        "P_prime_initial": P_prime_zero,
        "P_bar_classical_opt": classical_result["P_bar_opt"],
        "P_tilda_classical_opt": classical_result["P_tilda_opt"],
        "P_bar_lambda_opt": lambda_result["P_bar_opt"],
        "P_tilda_lambda_opt": lambda_result["P_tilda_opt"],
        "classical_full_comm_history": classical_result["full_comm_history"],
        "lambda_full_comm_history": lambda_result["full_comm_history"],
        "P_total_max": P_total_max,
        "Pmax_lin": Pmax_lin,
    }


# Run communication-only classical and optional DU for one clean-mode setup.
def run_communication_only_single_setup(
    comm_mat_file,
    Pmax_db,
    setup_idx=0,
    model=None,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-4,
    init_lambda=0.5,
    verbose=True,
):
    """
    Clean communication-only FP/QT run without D/lambda-QT.

    This mode is for RUN_MODE="only_comm". It uses the same communication-only
    reference construction as Code-B matching, fixes P_prime=0, and does not call
    the lambda-QT comparison routine.
    """
    ctx = build_communication_problem_context(
        mat_file=comm_mat_file,
        Pmax_db=Pmax_db,
        setup_idx=setup_idx,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        sigma_w2=sigma_w2,
        reference_mode="comm_only",
        hermitian_comm=True,
        load_sensing=False,
    )

    K = ctx["K"]
    w = np.ones(K)
    Pmax_lin = ctx["Pmax_lin"]
    P_total_max = ctx["P_total_max"]
    P_prime_zero = np.zeros(K, dtype=float)
    P_bar_init = float(init_lambda) * Pmax_lin * np.ones(K, dtype=float)
    P_tilda_init = (1.0 - float(init_lambda)) * Pmax_lin * np.ones(K, dtype=float)

    k = ctx["kappas"]

    classical_result = run_communication_only_given_kappas(
        K=K,
        w=w,
        d=ctx["d"],
        tau=ctx["tau"],
        kappa_S=k["kappa_S"],
        kappa_V1=k["kappa_V1"],
        kappa_K1=k["kappa_K1"],
        kappa_V0=k["kappa_V0"],
        kappa_K0=k["kappa_K0"],
        kappa_DAC1=k["kappa_DAC1"],
        kappa_DAC0=k["kappa_DAC0"],
        kappa_Th1=k["kappa_Th1"],
        kappa_ADC1=k["kappa_ADC1"],
        kappa_M1=k["kappa_M1"],
        kappa_M0=k["kappa_M0"],
        kappa_Th0=k["kappa_Th0"],
        kappa_ADC0=k["kappa_ADC0"],
        Z_list=ctx["Z_list"],
        P_bar_init=P_bar_init,
        P_tilda_init=P_tilda_init,
        P_prime_fixed=P_prime_zero,
        P_total_max=P_total_max,
        max_iters=max_iters,
        epsilon=epsilon,
        eps_power=1e-12,
        verbose=verbose,
    )

    unfolding_comm_history = None
    if model is not None:
        load_unfolding_api("only_comm")
        device = next(model.parameters()).device
        model = model.to(device).double()
        model.eval()
        with torch.no_grad():
            du_out = model(
                P_bar_init=to_torch_1d(P_bar_init, device),
                P_tilda_init=to_torch_1d(P_tilda_init, device),
                w=to_torch_1d(w, device),
                P_total_max=scalar_to_torch(P_total_max, device),
                d=ctx["d"],
                tau=ctx["tau"],
                kappa_S=to_torch_1d(k["kappa_S"], device),
                kappa_V1=to_torch_1d(k["kappa_V1"], device),
                kappa_K1=to_torch_1d(k["kappa_K1"], device),
                kappa_V0=to_torch_1d(k["kappa_V0"], device),
                kappa_K0=to_torch_1d(k["kappa_K0"], device),
                kappa_DAC1=to_torch_1d(k["kappa_DAC1"], device),
                kappa_DAC0=to_torch_1d(k["kappa_DAC0"], device),
                kappa_Th1=to_torch_1d(k["kappa_Th1"], device),
                kappa_ADC1=to_torch_1d(k["kappa_ADC1"], device),
                kappa_Th0=to_torch_1d(k["kappa_Th0"], device),
                kappa_ADC0=to_torch_1d(k["kappa_ADC0"], device),
                kappa_M1=to_torch_2d(k["kappa_M1"], device),
                kappa_M0=to_torch_2d(k["kappa_M0"], device),
                return_history=True,
            )

        unfolding_comm_history = du_out["comm_history"].mean(dim=1).detach().cpu().numpy()

    return {
        "mode": "only_comm",
        "Pmax_db": Pmax_db,
        "setup_idx": setup_idx,
        "comm_mat_file": comm_mat_file,
        "context": ctx,
        "classical_result": classical_result,
        "unfolding_comm_history": unfolding_comm_history,
        "du_final_wsr": (float(np.asarray(unfolding_comm_history).reshape(-1)[-1]) if unfolding_comm_history is not None and len(np.asarray(unfolding_comm_history).reshape(-1)) > 0 else np.nan),
        "P_bar_initial": P_bar_init,
        "P_tilda_initial": P_tilda_init,
        "P_prime_initial": P_prime_zero,
        "P_bar_classical_opt": classical_result["P_bar_opt"],
        "P_tilda_classical_opt": classical_result["P_tilda_opt"],
        "classical_full_comm_history": classical_result["full_comm_history"],
        "P_total_max": P_total_max,
        "Pmax_lin": Pmax_lin,
    }


# Build joint-mode sensing coefficients using context variables.
def build_joint_sensing_coefficients_for_context(ctx, verbose=False):
    """Build sensing coefficients only for joint communication-sensing mode."""
    b_target_all = ctx.get("b_target_all")
    alpha_var_matrix_all = ctx.get("alpha_var_matrix_all")
    H_direct_iter = ctx.get("H_direct_iter")
    H_true_iter = ctx.get("H_true_iter")
    V_combiner_all = ctx.get("V_combiner_all")

    if b_target_all is None:
        raise KeyError("Joint dataset is missing b_target_all; sensing coefficients cannot be built.")

    if alpha_var_matrix_all is None:
        alpha_var_matrix_all = estimate_sigma2_ilt_from_Cref(
            C_h_ref_all=ctx["C_h_ref_all"],
            b_target_all=b_target_all,
        )
        if verbose:
            print("[SENSING] alpha_var_matrix_all was missing; estimated it from C_h_ref_all and b_target_all.")

    if V_combiner_all is None:
        raise KeyError(
            "Joint dataset is missing V_combiner_all. For MATLAB-matching sensing, add V_combiner_all "
            "or change build_sensing_coefficients_matlab_matching to use MR combiners."
        )

    a_sens, b_sens, n_sens, sensing_coeff_parts = build_sensing_coefficients_matlab_matching(
        T=ctx["tau"],
        alpha_list=ctx["alpha_list"],
        sigma_d2=ctx["sigma_d2"],
        A_list=ctx["A_list"],
        Theta_list=ctx["Theta_list"],
        b_target_all=b_target_all,
        alpha_var_matrix_all=alpha_var_matrix_all,
        V_combiner_all=V_combiner_all,
        H_direct_iter=H_direct_iter,
        H_true_iter=H_true_iter,
        sigma_w2=ctx["sigma_w2"],
        return_parts=True,
    )

    target_weights = get_target_weights_from_mat(ctx["mat"], a_sens.shape[0])

    return a_sens, b_sens, n_sens, sensing_coeff_parts, target_weights


# Run joint classical and optional DU for one clean-mode setup.
def run_joint_comm_sensing_single_setup(
    joint_mat_file,
    Pmax_db,
    setup_idx=0,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-4,
    init_delta=0.5,
    init_lambda=0.5,
    w_c_user=None,
    w_s_user=None,
    verbose=True,
    model=None,
):
    """
    Clean joint communication-sensing classical FP/QT run.

    This uses the joint dataset and builds sensing coefficients only here.
    Communication kappas are built inside this function via the communication
    context and also inside run_joint_wsr_optimisation for the joint loop.
    """
    ctx = build_communication_problem_context(
        mat_file=joint_mat_file,
        Pmax_db=Pmax_db,
        setup_idx=setup_idx,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        sigma_w2=sigma_w2,
        reference_mode="joint",
        hermitian_comm=True,
        load_sensing=True,
    )

    K = ctx["K"]
    Pmax_lin = ctx["Pmax_lin"]
    P_total_max = ctx["P_total_max"]
    w = np.ones(K)

    mat = ctx["mat"]
    w_c = float(np.squeeze(mat["w_c"])) if (w_c_user is None and "w_c" in mat) else (1.0 if w_c_user is None else float(w_c_user))
    w_s = float(np.squeeze(mat["w_s"])) if (w_s_user is None and "w_s" in mat) else (1.0 if w_s_user is None else float(w_s_user))

    delta0 = float(init_delta)
    lambda0 = float(init_lambda)
    if not (0.0 <= delta0 <= 1.0):
        raise ValueError(f"init_delta must be in [0,1], got {delta0}")
    if not (0.0 <= lambda0 <= 1.0):
        raise ValueError(f"init_lambda must be in [0,1], got {lambda0}")

    P_prime_init = delta0 * Pmax_lin * np.ones(K)
    P_comm_init = (1.0 - delta0) * Pmax_lin
    P_bar_init = lambda0 * P_comm_init * np.ones(K)
    P_tilda_init = (1.0 - lambda0) * P_comm_init * np.ones(K)
    P_bar_init, P_tilda_init, P_prime_init = project_joint_three_power_blocks(
        P_bar_raw=P_bar_init,
        P_tilda_raw=P_tilda_init,
        P_prime_raw=P_prime_init,
        P_total_max=P_total_max,
        eps=1e-12,
    )

    a_sens, b_sens, n_sens, sensing_coeff_parts, target_weights = build_joint_sensing_coefficients_for_context(
        ctx,
        verbose=verbose,
    )

    k = ctx["kappas"]
    P_th_initial = P_bar_init * k["kappa_Th1"] + k["kappa_Th0"]
    P_adc_initial = P_bar_init * k["kappa_ADC1"] + k["kappa_ADC0"]

    initial_comm_wsr = compute_WSR_exact(
        K, w,
        k["kappa_S"],
        k["kappa_V1"], k["kappa_K1"],
        k["kappa_V0"], k["kappa_K0"],
        k["kappa_DAC1"], k["kappa_DAC0"],
        k["kappa_M1"], k["kappa_M0"],
        k["kappa_Th1"], k["kappa_ADC1"],
        P_bar_init, P_tilda_init,
        P_th_initial, P_adc_initial,
        ctx["d"], ctx["tau"],
    )

    initial_sensing_components = compute_sensing_components_from_coeffs(
        P_prime=P_prime_init,
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
        sensing_coeff_parts=sensing_coeff_parts,
    )
    # Keep sensing raw rate separate from its weighted objective contribution.
    # "weighted_wsr_no_ws" = sum_q omega_q log2(1+SINR_q), without w_s.
    # "wsr"                = w_s * sum_q omega_q log2(1+SINR_q), contribution to joint objective.
    initial_sensing_raw_wsr = float(initial_sensing_components["weighted_wsr_no_ws"])
    initial_sensing_contribution = float(initial_sensing_components["wsr"])
    initial_joint_wsr = float(w_c * initial_comm_wsr + initial_sensing_contribution)

    joint_result = run_joint_wsr_optimisation(
        K=K,
        w=w,
        alpha_list=ctx["alpha_list"],
        tau=ctx["tau"],
        d=ctx["d"],
        sigma_w2=sigma_w2,
        sigma_d2=ctx["sigma_d2"],
        U_list=ctx["U_list"],
        V_list=ctx["V_list"],
        Sigma_list=ctx["Sigma_list"],
        R_list=ctx["R_list"],
        h_bar_list=ctx["h_bar_list"],
        C_h_ref_all=ctx["C_h_ref_all"],
        C_h_total_all=ctx["C_h_total_all"],
        D_Th_list=ctx["D_Th_list"],
        D_ADC_list=ctx["D_ADC_list"],
        A_list=ctx["A_list"],
        Theta_list=ctx["Theta_list"],
        Z_list=ctx["Z_list"],
        G_list=ctx["G_list"],
        W_list=ctx["W_list"],
        T=ctx["tau"],
        U_q_list=ctx["U_q_list"],
        V_q_list=ctx["V_q_list"],
        g_lik=ctx["g_lik"],
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_c=w_c,
        w_s=w_s,
        P_bar_init=P_bar_init,
        P_tilda_init=P_tilda_init,
        P_prime_init=P_prime_init,
        max_iters=max_iters,
        epsilon=epsilon,
        P_total_max=P_total_max,
        eps_power=1e-12,
        verbose=verbose,
    )

    final_sensing_components = compute_sensing_components_from_coeffs(
        P_prime=joint_result["P_prime_opt"],
        a_sens=a_sens,
        b_sens=b_sens,
        n_sens=n_sens,
        target_weights=target_weights,
        w_s=w_s,
        sensing_coeff_parts=sensing_coeff_parts,
    )

    final_sensing_raw_wsr = float(final_sensing_components["weighted_wsr_no_ws"])
    final_sensing_contribution = float(final_sensing_components["wsr"])

    # Internally run_joint_wsr_optimisation stores weighted sensing contribution.
    sensing_history_contribution = np.asarray(joint_result["sensing_history"], dtype=float)
    if abs(float(w_s)) > 1e-30:
        sensing_history_raw = sensing_history_contribution / float(w_s)
    else:
        sensing_history_raw = np.zeros_like(sensing_history_contribution)

    full_joint_history = np.concatenate([
        [initial_joint_wsr],
        np.asarray(joint_result["joint_history"], dtype=float),
    ])


    # Optional joint DU evaluation using the trained/saved step sizes.
    if model is not None:
        du_out = _du_forward_from_clean_context(
            ctx=ctx,
            model=model,
            run_mode="joint",
            P_bar_init=P_bar_init,
            P_tilda_init=P_tilda_init,
            P_prime_init=P_prime_init,
            w=w,
            a_sens=a_sens,
            b_sens=b_sens,
            n_sens=n_sens,
            target_weights=target_weights,
            w_c=w_c,
            w_s=w_s,
            train_unfolding=False,
        )
        deep_final_joint_wsr = du_out["joint_obj"].mean().detach().cpu().item()
        deep_final_comm_wsr = du_out["comm_wsr"].mean().detach().cpu().item()
        deep_final_sensing_raw_wsr = du_out["sensing_wsr"].mean().detach().cpu().item()
        deep_final_sensing_contribution = float(w_s) * deep_final_sensing_raw_wsr
        deep_joint_history = du_out["joint_history"].mean(dim=1).detach().cpu().numpy()
        deep_comm_history = du_out["comm_history"].mean(dim=1).detach().cpu().numpy()
        deep_sensing_history_raw = du_out["sensing_history"].mean(dim=1).detach().cpu().numpy()
        deep_sensing_history_contribution = float(w_s) * deep_sensing_history_raw
        deep_P_bar = du_out["P_bar"].detach().cpu().numpy().reshape(-1)
        deep_P_tilda = du_out["P_tilda"].detach().cpu().numpy().reshape(-1)
        deep_P_prime = du_out["P_prime"].detach().cpu().numpy().reshape(-1)
        if verbose:
            print(
                f"[JOINT-DU] setup={setup_idx:02d} | "
                f"DU_joint={deep_final_joint_wsr:.6f} | "
                f"DU_comm_raw={deep_final_comm_wsr:.6f} | "
                f"DU_sens_raw={deep_final_sensing_raw_wsr:.6f} | "
                f"DU_sens_contrib={deep_final_sensing_contribution:.6f}"
            )
    else:
        deep_final_joint_wsr = np.nan
        deep_final_comm_wsr = np.nan
        deep_final_sensing_raw_wsr = np.nan
        deep_final_sensing_contribution = np.nan
        deep_joint_history = np.array([])
        deep_comm_history = np.array([])
        deep_sensing_history_raw = np.array([])
        deep_sensing_history_contribution = np.array([])
        deep_P_bar = np.full(K, np.nan)
        deep_P_tilda = np.full(K, np.nan)
        deep_P_prime = np.full(K, np.nan)

    result = {
        "mode": "joint",
        "Pmax_db": Pmax_db,
        "setup_idx": setup_idx,
        "joint_mat_file": joint_mat_file,
        "initial_wsr": initial_joint_wsr,
        "initial_joint_wsr": initial_joint_wsr,
        "initial_comm_wsr": float(initial_comm_wsr),
        "initial_comm_contribution": float(w_c * initial_comm_wsr),
        "initial_sensing_wsr": initial_sensing_raw_wsr,          # raw sensing rate, no w_s
        "initial_sensing_raw_wsr": initial_sensing_raw_wsr,
        "initial_sensing_contribution": initial_sensing_contribution,
        "initial_sensing_weighted_wsr": initial_sensing_contribution,
        "final_wsr": float(joint_result["joint_history"][-1]),
        "final_comm_wsr": float(joint_result["comm_history"][-1]),
        "final_comm_contribution": float(w_c * joint_result["comm_history"][-1]),
        "final_sensing_wsr": final_sensing_raw_wsr,              # raw sensing rate, no w_s
        "final_sensing_raw_wsr": final_sensing_raw_wsr,
        "final_sensing_contribution": final_sensing_contribution,
        "final_sensing_weighted_wsr": final_sensing_contribution,
        "deep_final_joint_wsr": deep_final_joint_wsr,
        "deep_final_comm_wsr": deep_final_comm_wsr,
        "deep_final_sensing_raw_wsr": deep_final_sensing_raw_wsr,
        "deep_final_sensing_contribution": deep_final_sensing_contribution,
        "deep_joint_history": deep_joint_history,
        "deep_comm_history": deep_comm_history,
        "deep_sensing_history_raw": deep_sensing_history_raw,
        "deep_sensing_history_contribution": deep_sensing_history_contribution,
        "deep_P_bar": deep_P_bar,
        "deep_P_tilda": deep_P_tilda,
        "deep_P_prime": deep_P_prime,
        "joint_history": joint_result["joint_history"],
        "comm_history": joint_result["comm_history"],
        "sensing_history": sensing_history_raw,
        "sensing_history_raw": sensing_history_raw,
        "sensing_history_contribution": sensing_history_contribution,
        "full_joint_history": full_joint_history,
        "P_bar_initial": P_bar_init,
        "P_tilda_initial": P_tilda_init,
        "P_prime_initial": P_prime_init,
        "P_bar_opt": joint_result["P_bar_opt"],
        "P_tilda_opt": joint_result["P_tilda_opt"],
        "P_prime_opt": joint_result["P_prime_opt"],
        "gamma_opt": joint_result["gamma_opt"],
        "sensing_projection": final_sensing_components,
        "initial_sensing_components": initial_sensing_components,
        "target_weights": target_weights,
        "a_sens": a_sens,
        "b_sens": b_sens,
        "n_sens": n_sens,
        "sensing_coeff_parts": sensing_coeff_parts,
        "w_c": w_c,
        "w_s": w_s,
        "Pmax_lin": Pmax_lin,
        "P_total_max": P_total_max,
        "K": K,
        "M": ctx["M"],
        "N": ctx["N"],
        "tau": ctx["tau"],
        "d": ctx["d"],
        "converged": joint_result["converged"],
        "convergence_iter": joint_result["convergence_iter"],
        "elapsed_time_sec": joint_result["elapsed_time_sec"],
        "final_delta": joint_result["final_delta"],
        "comm_kappas": joint_result["kappas"],
        "context": ctx,
    }

    if verbose:
        print_joint_summary_table("Clean Joint Communication-Sensing Summary", result)

    return result



# =============================================================================
# Clean deep-unfolding training / saving / testing helpers
# =============================================================================

# =============================================================================
# Clean-mode DU model creation, saving, loading, and forward evaluation
# =============================================================================
# This block creates the correct DU model for each mode and handles learned
# step-size persistence for reproducible testing.
# Create the correct DU model for communication-only or joint mode.
def create_unfolding_model_for_mode(
    run_mode,
    K_use=8,
    num_layers=30,
    num_pgd_steps=5,
    init_step_bar=0.05,
    init_step_tilda=0.05,
    init_step_prime=0.05,
    enforce_full_power=False,
    device=None,
):
    """Create the correct unfolding model for comm-only or joint mode."""
    api = load_unfolding_api(run_mode)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointCommSensingUnfoldingPGD(
        K=K_use,
        num_layers=num_layers,
        num_pgd_steps=num_pgd_steps,
        init_step_bar=init_step_bar,
        init_step_tilda=init_step_tilda,
        init_step_prime=init_step_prime,
        enforce_full_power=enforce_full_power,
    ).to(device).double()
    print(f"[UNFOLDING IMPORT] RUN_MODE={run_mode} loaded {api['loaded_from']}")
    return model


# Save learned DU step sizes as TXT and NPZ together with configuration metadata.
def save_learned_step_sizes(model, out_dir, prefix="learned_step_sizes", config=None):
    """Save learned step sizes as both .txt and .npz."""
    os.makedirs(out_dir, exist_ok=True)
    steps = get_learned_step_sizes(model)
    txt_path = os.path.join(out_dir, f"{prefix}.txt")
    npz_path = os.path.join(out_dir, f"{prefix}.npz")

    with open(txt_path, "w") as f:
        f.write("Learned Deep-Unfolding Step Sizes\n")
        f.write("================================\n")
        if config is not None:
            f.write("\nConfiguration\n")
            f.write("-------------\n")
            for key, val in config.items():
                f.write(f"{key}: {val}\n")
        f.write("\nLayer-wise values\n")
        f.write("-----------------\n")
        f.write(f"{'layer':>5} {'step_Pbar':>18} {'step_Ptilda':>18} {'step_Pprime':>18}\n")
        L = len(steps["Pbar"])
        for i in range(L):
            f.write(
                f"{i:5d} "
                f"{steps['Pbar'][i]:18.10e} "
                f"{steps['Ptilda'][i]:18.10e} "
                f"{steps['Pprime'][i]:18.10e}\n"
            )
        f.write("\nSummary\n")
        f.write("-------\n")
        for name, arr in steps.items():
            f.write(
                f"{name}: mean={np.mean(arr):.10e}, "
                f"min={np.min(arr):.10e}, max={np.max(arr):.10e}\n"
            )

    np.savez(
        npz_path,
        step_Pbar=steps["Pbar"],
        step_Ptilda=steps["Ptilda"],
        step_Pprime=steps["Pprime"],
    )
    print(f"[SAVE] learned step sizes text: {txt_path}")
    print(f"[SAVE] learned step sizes npz : {npz_path}")
    return {"txt": txt_path, "npz": npz_path, "steps": steps}


# Save the full DU model checkpoint with configuration metadata.
def save_unfolding_model_checkpoint(model, out_dir, prefix="learned_unfolding_model", config=None):
    """Save the trained model state_dict. This is optional but useful for reruns."""
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{prefix}.pt")
    payload = {
        "state_dict": model.state_dict(),
        "step_sizes": get_learned_step_sizes(model),
        "config": {} if config is None else dict(config),
    }
    torch.save(payload, ckpt_path)
    print(f"[SAVE] model checkpoint: {ckpt_path}")
    return ckpt_path


# Load saved learned step sizes into a freshly created DU model.
def load_step_sizes_into_model(model, npz_path, device=None):
    """Load only saved step sizes into a fresh model and return the model."""
    if device is None:
        device = next(model.parameters()).device
    data = np.load(npz_path)
    with torch.no_grad():
        model.step_Pbar.copy_(torch.as_tensor(data["step_Pbar"], dtype=torch.double, device=device))
        model.step_Ptilda.copy_(torch.as_tensor(data["step_Ptilda"], dtype=torch.double, device=device))
        model.step_Pprime.copy_(torch.as_tensor(data["step_Pprime"], dtype=torch.double, device=device))
    print(f"[LOAD] loaded learned step sizes from {npz_path}")
    return model


# Run a DU forward pass using tensors built from a clean-mode context.
def _du_forward_from_clean_context(
    ctx,
    model,
    run_mode,
    P_bar_init,
    P_tilda_init,
    P_prime_init,
    w,
    a_sens,
    b_sens,
    n_sens,
    target_weights,
    w_c,
    w_s,
    train_unfolding=False,
):
    """Run one differentiable/eval DU forward pass from a prebuilt clean context.

    Communication modes call the pure two-block communication unfolding and do
    not pass sensing tensors at all. Joint mode keeps the original three-block
    communication+sensing call.
    """
    load_unfolding_api(run_mode)
    device = next(model.parameters()).device
    model = model.to(device).double()
    model.train(mode=bool(train_unfolding))
    k = ctx["kappas"]
    mode = str(run_mode).lower().strip()

    ctx_mgr = torch.enable_grad() if train_unfolding else torch.no_grad()
    with ctx_mgr:
        if is_communication_mode(mode):
            du_out = model(
                P_bar_init=to_torch_1d(P_bar_init, device),
                P_tilda_init=to_torch_1d(P_tilda_init, device),
                w=to_torch_1d(w, device),
                P_total_max=scalar_to_torch(ctx["P_total_max"], device),
                d=ctx["d"],
                tau=ctx["tau"],
                kappa_S=to_torch_1d(k["kappa_S"], device),
                kappa_V1=to_torch_1d(k["kappa_V1"], device),
                kappa_K1=to_torch_1d(k["kappa_K1"], device),
                kappa_V0=to_torch_1d(k["kappa_V0"], device),
                kappa_K0=to_torch_1d(k["kappa_K0"], device),
                kappa_DAC1=to_torch_1d(k["kappa_DAC1"], device),
                kappa_DAC0=to_torch_1d(k["kappa_DAC0"], device),
                kappa_Th1=to_torch_1d(k["kappa_Th1"], device),
                kappa_ADC1=to_torch_1d(k["kappa_ADC1"], device),
                kappa_Th0=to_torch_1d(k["kappa_Th0"], device),
                kappa_ADC0=to_torch_1d(k["kappa_ADC0"], device),
                kappa_M1=to_torch_2d(k["kappa_M1"], device),
                kappa_M0=to_torch_2d(k["kappa_M0"], device),
                return_history=True,
            )
        else:
            du_out = model(
                P_bar_init=to_torch_1d(P_bar_init, device),
                P_tilda_init=to_torch_1d(P_tilda_init, device),
                P_prime_init=to_torch_1d(P_prime_init, device),
                w=to_torch_1d(w, device),
                P_total_max=scalar_to_torch(ctx["P_total_max"], device),
                d=ctx["d"],
                tau=ctx["tau"],
                kappa_S=to_torch_1d(k["kappa_S"], device),
                kappa_V1=to_torch_1d(k["kappa_V1"], device),
                kappa_K1=to_torch_1d(k["kappa_K1"], device),
                kappa_V0=to_torch_1d(k["kappa_V0"], device),
                kappa_K0=to_torch_1d(k["kappa_K0"], device),
                kappa_DAC1=to_torch_1d(k["kappa_DAC1"], device),
                kappa_DAC0=to_torch_1d(k["kappa_DAC0"], device),
                kappa_Th1=to_torch_1d(k["kappa_Th1"], device),
                kappa_ADC1=to_torch_1d(k["kappa_ADC1"], device),
                kappa_Th0=to_torch_1d(k["kappa_Th0"], device),
                kappa_ADC0=to_torch_1d(k["kappa_ADC0"], device),
                kappa_M1=to_torch_2d(k["kappa_M1"], device),
                kappa_M0=to_torch_2d(k["kappa_M0"], device),
                a_sens=to_torch_2d(a_sens, device),
                b_sens=to_torch_2d(b_sens, device),
                n_sens=to_torch_1d(n_sens, device),
                target_weights=to_torch_1d(target_weights, device),
                w_c=w_c,
                w_s=w_s,
                return_history=True,
            )
    return du_out


# Prepare clean-mode DU inputs for one setup and Pmax value.
def _build_clean_du_inputs_for_setup(
    run_mode,
    mat_file,
    Pmax_db,
    setup_idx,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    init_delta=0.5,
    init_lambda=0.5,
    w_c_user=0.2,
    w_s_user=0.8,
):
    """Build clean-mode tensors/arrays for one setup, used by both training and testing."""
    mode = str(run_mode).lower().strip()

    if mode in ["comm", "comm_lambda", "comm_only", "comm_only_compare", "communication_lambda", "only_comm", "comm_fpqt"]:
        ctx = build_communication_problem_context(
            mat_file=mat_file,
            Pmax_db=Pmax_db,
            setup_idx=setup_idx,
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=sigma_w2,
            reference_mode="comm_only",
            hermitian_comm=True,
            load_sensing=False,
        )
        K = ctx["K"]
        Pmax_lin = ctx["Pmax_lin"]
        P_bar_init = float(init_lambda) * Pmax_lin * np.ones(K)
        P_tilda_init = (1.0 - float(init_lambda)) * Pmax_lin * np.ones(K)
        P_prime_init = np.zeros(K)
        w = np.ones(K)
        a_sens = np.zeros((1, K), dtype=float)
        b_sens = np.zeros((1, K), dtype=float)
        n_sens = np.ones(1, dtype=float)
        target_weights = np.ones(1, dtype=float)
        return {
            "mode": mode,
            "ctx": ctx,
            "P_bar_init": P_bar_init,
            "P_tilda_init": P_tilda_init,
            "P_prime_init": P_prime_init,
            "w": w,
            "a_sens": a_sens,
            "b_sens": b_sens,
            "n_sens": n_sens,
            "target_weights": target_weights,
            "w_c": 1.0,
            "w_s": 0.0,
            "sensing_coeff_parts": None,
        }

    if mode in ["joint", "joint_comm_sensing", "joint_lambda", "joint_comm_sensing_lambda"]:
        ctx = build_communication_problem_context(
            mat_file=mat_file,
            Pmax_db=Pmax_db,
            setup_idx=setup_idx,
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=sigma_w2,
            reference_mode="joint",
            hermitian_comm=True,
            load_sensing=True,
        )
        K = ctx["K"]
        Pmax_lin = ctx["Pmax_lin"]
        P_total_max = ctx["P_total_max"]
        delta0 = float(init_delta)
        lambda0 = float(init_lambda)
        P_prime_init = delta0 * Pmax_lin * np.ones(K)
        P_comm_init = (1.0 - delta0) * Pmax_lin
        P_bar_init = lambda0 * P_comm_init * np.ones(K)
        P_tilda_init = (1.0 - lambda0) * P_comm_init * np.ones(K)
        P_bar_init, P_tilda_init, P_prime_init = project_joint_three_power_blocks(
            P_bar_init, P_tilda_init, P_prime_init, P_total_max, eps=1e-12
        )
        a_sens, b_sens, n_sens, sensing_coeff_parts, target_weights = build_joint_sensing_coefficients_for_context(
            ctx, verbose=False
        )
        return {
            "mode": "joint" if mode in ["joint", "joint_comm_sensing"] else "joint_lambda",
            "ctx": ctx,
            "P_bar_init": P_bar_init,
            "P_tilda_init": P_tilda_init,
            "P_prime_init": P_prime_init,
            "w": np.ones(K),
            "a_sens": a_sens,
            "b_sens": b_sens,
            "n_sens": n_sens,
            "target_weights": target_weights,
            "w_c": float(w_c_user),
            "w_s": float(w_s_user),
            "sensing_coeff_parts": sensing_coeff_parts,
        }

    raise ValueError("run_mode must be 'only_comm', 'comm_lambda', or 'joint'.")


# =============================================================================
# Clean-mode DU training and evaluation loops
# =============================================================================
# These functions are used by the current __main__ block for training over
# batches and testing the saved learned step sizes.
# Train one DU batch using the clean-mode context pipeline.
def train_one_batch_clean_mode(
    run_mode,
    mat_file,
    model,
    optimizer,
    setup_batch,
    Pmax_db,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    init_delta=0.5,
    init_lambda=0.5,
    w_c_user=0.2,
    w_s_user=0.8,
):
    """One optimizer step for a clean-mode batch."""
    model.train()
    optimizer.zero_grad()
    batch_joint_tensors = []
    batch_layer_histories = []

    for setup_idx in setup_batch:
        inputs = _build_clean_du_inputs_for_setup(
            run_mode=run_mode,
            mat_file=mat_file,
            Pmax_db=float(Pmax_db),
            setup_idx=int(setup_idx),
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=sigma_w2,
            init_delta=init_delta,
            init_lambda=init_lambda,
            w_c_user=w_c_user,
            w_s_user=w_s_user,
        )
        du_out = _du_forward_from_clean_context(
            ctx=inputs["ctx"],
            model=model,
            run_mode=inputs["mode"],
            P_bar_init=inputs["P_bar_init"],
            P_tilda_init=inputs["P_tilda_init"],
            P_prime_init=inputs["P_prime_init"],
            w=inputs["w"],
            a_sens=inputs["a_sens"],
            b_sens=inputs["b_sens"],
            n_sens=inputs["n_sens"],
            target_weights=inputs["target_weights"],
            w_c=inputs["w_c"],
            w_s=inputs["w_s"],
            train_unfolding=True,
        )
        batch_joint_tensors.append(du_out["joint_obj"].mean())
        batch_layer_histories.append(du_out["joint_history"].mean(dim=1).detach().cpu().numpy())

    batch_mean_joint_tensor = torch.stack(batch_joint_tensors).mean()
    loss = -batch_mean_joint_tensor
    loss.backward()
    optimizer.step()

    batch_mean_joint = batch_mean_joint_tensor.detach().cpu().item()
    batch_layer_history_mean = np.mean(np.stack(batch_layer_histories, axis=0), axis=0)
    return batch_mean_joint, batch_layer_history_mean


# Active clean-mode DU training loop selected by RUN_MODE.
def train_unfolding_model_by_mode(
    run_mode,
    comm_mat_file=None,
    joint_mat_file=None,
    model=None,
    optimizer=None,
    train_setups=np.arange(0, 1),
    train_power_list=np.array([10.0]),
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    num_epochs=2,
    batch_size=10,
    init_delta=0.5,
    init_lambda=0.5,
    w_c_user=0.2,
    w_s_user=0.8,
    shuffle_batches=True,
    seed=42,
):
    """Train the correct unfolding model for comm-only or joint mode."""
    mode = str(run_mode).lower().strip()
    mat_file = comm_mat_file if mode in ["comm", "comm_lambda", "comm_only", "comm_only_compare", "communication_lambda", "only_comm", "comm_fpqt"] else joint_mat_file
    if mat_file is None:
        raise ValueError("Provide comm_mat_file for comm mode or joint_mat_file for joint mode.")
    if model is None:
        model = create_unfolding_model_for_mode(mode, K_use=K_use)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

    train_setups = np.asarray(train_setups, dtype=int)
    train_power_list = np.asarray(train_power_list, dtype=float)
    rng = np.random.default_rng(seed)
    train_joint_log = {float(Pdb): [] for Pdb in train_power_list}
    train_layer_joint_log = {float(Pdb): [] for Pdb in train_power_list}

    print("\n" + "=" * 90)
    print(f"Training clean DU | mode={mode} | Pmax values={train_power_list}")
    print(f"shuffle_batches={shuffle_batches}, batch_size={batch_size}, seed={seed}")
    print("=" * 90)

    for epoch in range(num_epochs):
        train_batches = make_shuffled_batches(train_setups, batch_size, shuffle_batches, rng)
        epoch_power_means = []
        for Pdb in train_power_list:
            Pdb = float(Pdb)
            power_batch_vals = []
            power_batch_histories = []
            for batch_id, setup_batch in enumerate(train_batches):
                batch_mean, batch_hist = train_one_batch_clean_mode(
                    run_mode=mode,
                    mat_file=mat_file,
                    model=model,
                    optimizer=optimizer,
                    setup_batch=setup_batch,
                    Pmax_db=Pdb,
                    K_use=K_use,
                    M_use=M_use,
                    N_use=N_use,
                    sigma_w2=sigma_w2,
                    init_delta=0.0 if mode in ["comm", "comm_lambda", "comm_only", "comm_only_compare", "communication_lambda", "only_comm", "comm_fpqt"] else init_delta,
                    init_lambda=init_lambda,
                    w_c_user=1.0 if mode in ["comm", "comm_lambda", "comm_only", "comm_only_compare", "communication_lambda", "only_comm", "comm_fpqt"] else w_c_user,
                    w_s_user=0.0 if mode in ["comm", "comm_lambda", "comm_only", "comm_only_compare", "communication_lambda", "only_comm", "comm_fpqt"] else w_s_user,
                )
                power_batch_vals.append(batch_mean)
                power_batch_histories.append(batch_hist)
                setup_list_str = ",".join(map(str, setup_batch.tolist()))
                print(
                    f"P={Pdb:>6.1f} dB | Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_id+1}/{len(train_batches)} | Setups [{setup_list_str}] | "
                    f"Batch mean final DU objective = {batch_mean:.6f}"
                )
            power_epoch_mean = float(np.mean(power_batch_vals))
            power_epoch_history = np.mean(np.stack(power_batch_histories, axis=0), axis=0)
            train_joint_log[Pdb].append(power_epoch_mean)
            train_layer_joint_log[Pdb].append(power_epoch_history)
            epoch_power_means.append(power_epoch_mean)
            print(
                f"[Epoch Power Summary] Epoch {epoch+1}/{num_epochs} | P={Pdb:>6.1f} dB | "
                f"Mean final DU objective = {power_epoch_mean:.6f}"
            )
        print(f"[Epoch Summary] Epoch {epoch+1}/{num_epochs} | Mean over powers = {float(np.mean(epoch_power_means)):.6f}")

    return {"model": model, "optimizer": optimizer, "train_joint_log": train_joint_log, "train_layer_joint_log": train_layer_joint_log}


# Evaluate a trained DU model for a clean-mode result/context pair.
def evaluate_du_for_clean_result(result, model, run_mode):
    """Attach DU testing outputs to one clean result dictionary."""
    if model is None:
        return result
    mode = str(run_mode).lower().strip()
    if result.get("mode") == "comm_lambda":
        hist = result.get("unfolding_comm_history")
        if hist is not None and len(hist) > 0:
            result["du_final_wsr"] = float(np.asarray(hist).reshape(-1)[-1])
        return result
    return result


# Run joint lambda-QT for one clean-mode setup and package its results.
def run_joint_lambda_single_setup(
    joint_mat_file,
    Pmax_db,
    setup_idx=0,
    model=None,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-4,
    init_delta=0.5,
    init_lambda=0.5,
    w_c_user=None,
    w_s_user=None,
    verbose=True,
    save_plot_path=None,
):
    """
    Clean joint comparison mode:
        EPA / initial,
        joint classical FP/QT,
        joint deep unfolding (if model is provided),
        joint D/lambda-QT including sensing power.
    """
    classical_result = run_joint_comm_sensing_single_setup(
        joint_mat_file=joint_mat_file,
        Pmax_db=Pmax_db,
        setup_idx=setup_idx,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        sigma_w2=sigma_w2,
        max_iters=max_iters,
        epsilon=epsilon,
        init_delta=init_delta,
        init_lambda=init_lambda,
        w_c_user=w_c_user,
        w_s_user=w_s_user,
        verbose=verbose,
        model=model,
    )

    k = classical_result["comm_kappas"]
    K = int(classical_result["K"])
    w = np.ones(K, dtype=float)

    lambda_result = run_joint_lambda_qt(
        K=K,
        w=w,
        d=classical_result["d"],
        tau=classical_result["tau"],
        w_c=classical_result["w_c"],
        w_s=classical_result["w_s"],
        kappa_S=k["kappa_S"],
        kappa_V1=k["kappa_V1"],
        kappa_K1=k["kappa_K1"],
        kappa_V0=k["kappa_V0"],
        kappa_K0=k["kappa_K0"],
        kappa_DAC1=k["kappa_DAC1"],
        kappa_DAC0=k["kappa_DAC0"],
        kappa_Th1=k["kappa_Th1"],
        kappa_ADC1=k["kappa_ADC1"],
        kappa_M1=k["kappa_M1"],
        kappa_M0=k["kappa_M0"],
        kappa_Th0=k["kappa_Th0"],
        kappa_ADC0=k["kappa_ADC0"],
        a_sens=classical_result["a_sens"],
        b_sens=classical_result["b_sens"],
        n_sens=classical_result["n_sens"],
        target_weights=classical_result["target_weights"],
        P_bar_init=classical_result["P_bar_initial"],
        P_tilda_init=classical_result["P_tilda_initial"],
        P_prime_init=classical_result["P_prime_initial"],
        P_total_max=classical_result["P_total_max"],
        max_iters=max_iters,
        epsilon=epsilon,
        eps_power=1e-12,
        eps_lambda=1e-30,
        lambda_mode_bar="actual",
        lambda_step_power=0.5,
        use_backtracking=True,
        verbose=verbose,
        print_step_sizes=True,
        step_print_every=50,
        step_print_last_n=10,
        compute_sinr_exact_fn=compute_SINR_exact,
        compute_wsr_exact_fn=compute_WSR_exact,
        update_auxiliary_fn=update_auxiliary,
    )

    save_joint_lambda_step_size_summary(
        lambda_result,
        f"joint_lambda_step_sizes_Pmax_{Pmax_db:g}dB_setup_{setup_idx}.txt"
    )

    if save_plot_path is not None:
        plot_joint_method_comparison(
            initial_value=classical_result["initial_joint_wsr"],
            classical_result=classical_result,
            lambda_qt_result=lambda_result,
            du_history=classical_result.get("deep_joint_history", np.array([])),
            save_path=save_plot_path,
        )

    if verbose:
        print("\nJOINT-LAMBDA COMPARISON")
        print("-" * 100)
        print(f"Initial EPA / joint objective : {classical_result['initial_joint_wsr']:.8f}")
        print(f"Joint classical FP/QT final   : {classical_result['final_wsr']:.8f}")
        print(f"Joint D/lambda-QT final       : {lambda_result['final_joint_wsr']:.8f}")
        du_final = classical_result.get("deep_final_joint_wsr", np.nan)
        if np.isfinite(du_final):
            print(f"Joint deep unfolding final    : {du_final:.8f}")

    return {
        "mode": "joint_lambda",
        "Pmax_db": Pmax_db,
        "setup_idx": setup_idx,
        "joint_mat_file": joint_mat_file,
        "initial_joint_wsr": classical_result["initial_joint_wsr"],
        "initial_wsr": classical_result["initial_joint_wsr"],
        "classical_result": classical_result,
        "lambda_qt_result": lambda_result,
        "comparison": {
            "classical": classical_result,
            "lambda_qt": lambda_result,
            "du_history": classical_result.get("deep_joint_history", np.array([])),
        },
        "final_wsr": classical_result["final_wsr"],
        "final_lambda_wsr": lambda_result["final_joint_wsr"],
        "final_comm_wsr": classical_result["final_comm_wsr"],
        "final_sensing_wsr": classical_result["final_sensing_wsr"],
        "final_comm_contribution": classical_result["final_comm_contribution"],
        "final_sensing_contribution": classical_result["final_sensing_contribution"],
        "deep_final_joint_wsr": classical_result.get("deep_final_joint_wsr", np.nan),
        "deep_final_comm_wsr": classical_result.get("deep_final_comm_wsr", np.nan),
        "deep_final_sensing_raw_wsr": classical_result.get("deep_final_sensing_raw_wsr", np.nan),
        "deep_final_sensing_contribution": classical_result.get("deep_final_sensing_contribution", np.nan),
        "deep_joint_history": classical_result.get("deep_joint_history", np.array([])),
        "deep_comm_history": classical_result.get("deep_comm_history", np.array([])),
        "deep_sensing_history_raw": classical_result.get("deep_sensing_history_raw", np.array([])),
        "deep_sensing_history_contribution": classical_result.get("deep_sensing_history_contribution", np.array([])),
        "classical_full_joint_history": classical_result["full_joint_history"],
        "lambda_full_joint_history": lambda_result["full_joint_history"],
        "P_bar_initial": classical_result["P_bar_initial"],
        "P_tilda_initial": classical_result["P_tilda_initial"],
        "P_prime_initial": classical_result["P_prime_initial"],
        "P_bar_classical_opt": classical_result["P_bar_opt"],
        "P_tilda_classical_opt": classical_result["P_tilda_opt"],
        "P_prime_classical_opt": classical_result["P_prime_opt"],
        "P_bar_lambda_opt": lambda_result["P_bar_opt"],
        "P_tilda_lambda_opt": lambda_result["P_tilda_opt"],
        "P_prime_lambda_opt": lambda_result["P_prime_opt"],
        "lambda_bar_history": lambda_result["lambda_bar_history"],
        "lambda_tilda_history": lambda_result["lambda_tilda_history"],
        "lambda_prime_history": lambda_result["lambda_prime_history"],
        "alpha_bar_history": lambda_result["alpha_bar_history"],
        "alpha_tilda_history": lambda_result["alpha_tilda_history"],
        "alpha_prime_history": lambda_result["alpha_prime_history"],
        "P_total_max": classical_result["P_total_max"],
        "Pmax_lin": classical_result["Pmax_lin"],
        "K": K,
        "M": classical_result["M"],
        "N": classical_result["N"],
        "tau": classical_result["tau"],
        "d": classical_result["d"],
        "w_c": classical_result["w_c"],
        "w_s": classical_result["w_s"],
        "target_weights": classical_result["target_weights"],
        "a_sens": classical_result["a_sens"],
        "b_sens": classical_result["b_sens"],
        "n_sens": classical_result["n_sens"],
        "context": classical_result.get("context", None),
    }


# =============================================================================
# Clean-mode one-setup runner and final power sweep
# =============================================================================
# This is the active experiment pipeline used by the __main__ configuration.
# It combines EPA, classical, lambda-QT, and DU outputs depending on mode.
# Active clean-mode runner for one setup and one power value.
def run_one_setup_power_clean(
    Pmax_db,
    comm_mat_file=None,
    joint_mat_file=None,
    mat_file=None,
    mode="comm_lambda",
    setup_idx=0,
    model=None,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-4,
    init_delta=0.5,
    init_lambda=0.5,
    w_c_user=None,
    w_s_user=None,
    verbose=True,
    save_plot_path=None,
):
    """
    Clean dispatcher.

    mode='comm_lambda' or 'comm_only_compare': uses comm_mat_file.
    mode='joint': uses joint_mat_file.
    """
    if mode in ["only_comm", "comm_only", "comm_fpqt"]:
        chosen = comm_mat_file or mat_file
        if chosen is None:
            raise ValueError("comm_mat_file or mat_file must be provided for only_comm mode.")
        return run_communication_only_single_setup(
            comm_mat_file=chosen,
            Pmax_db=Pmax_db,
            setup_idx=setup_idx,
            model=model,
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=sigma_w2,
            max_iters=max_iters,
            epsilon=epsilon,
            init_lambda=init_lambda,
            verbose=verbose,
        )

    if mode in ["comm_lambda", "comm_only_compare", "communication_lambda"]:
        chosen = comm_mat_file or mat_file
        if chosen is None:
            raise ValueError("comm_mat_file or mat_file must be provided for communication-lambda mode.")
        return run_communication_lambda_single_setup(
            comm_mat_file=chosen,
            Pmax_db=Pmax_db,
            setup_idx=setup_idx,
            model=model,
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=sigma_w2,
            max_iters=max_iters,
            epsilon=epsilon,
            init_lambda=init_lambda,
            verbose=verbose,
            save_plot_path=save_plot_path,
        )

    if mode in ["joint", "joint_comm_sensing"]:
        chosen = joint_mat_file or mat_file
        if chosen is None:
            raise ValueError("joint_mat_file or mat_file must be provided for joint mode.")

        return run_joint_comm_sensing_single_setup(
            joint_mat_file=chosen,
            Pmax_db=Pmax_db,
            setup_idx=setup_idx,
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=sigma_w2,
            max_iters=max_iters,
            epsilon=epsilon,
            init_delta=init_delta,
            init_lambda=init_lambda,
            w_c_user=w_c_user,
            w_s_user=w_s_user,
            verbose=verbose,
            model=model,
        )


    if mode in ["joint_lambda", "joint_comm_sensing_lambda"]:
        chosen = joint_mat_file or mat_file
        if chosen is None:
            raise ValueError("joint_mat_file or mat_file must be provided for joint-lambda mode.")

        return run_joint_lambda_single_setup(
            joint_mat_file=chosen,
            Pmax_db=Pmax_db,
            setup_idx=setup_idx,
            model=model,
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=sigma_w2,
            max_iters=max_iters,
            epsilon=epsilon,
            init_delta=init_delta,
            init_lambda=init_lambda,
            w_c_user=w_c_user,
            w_s_user=w_s_user,
            verbose=verbose,
            save_plot_path=save_plot_path,
        )

    raise ValueError("mode must be 'only_comm', 'comm_lambda'/'comm_only_compare', or 'joint'.")


# Active final power sweep used by the current __main__ block.
def run_clean_power_sweep(
    power_list,
    test_setups,
    comm_mat_file=None,
    joint_mat_file=None,
    mat_file=None,
    mode="comm_lambda",
    model=None,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-4,
    init_delta=0.5,
    init_lambda=0.5,
    w_c_user=None,
    w_s_user=None,
    save_plots=False,
):
    """Clean sweep for either communication-lambda or joint mode."""
    all_results = {}
    initial_by_power = []
    classical_by_power = []
    lambda_by_power = []
    comm_by_power = []
    sensing_by_power = []
    classical_history_by_power = []
    lambda_history_by_power = []
    du_by_power = []
    du_history_by_power = []

    for Pdb in np.asarray(power_list, dtype=float):
        setup_results = []
        initial_vals = []
        classical_vals = []
        lambda_vals = []
        comm_vals = []
        sensing_vals = []
        classical_histories = []
        lambda_histories = []
        du_vals = []
        du_histories = []

        print("\n" + "=" * 100)
        print(f"CLEAN RUN | mode={mode} | Pmax={Pdb:.1f} dB")
        print("=" * 100)

        for setup_idx in test_setups:
            setup_idx = int(setup_idx)
            result = run_one_setup_power_clean(
                Pmax_db=float(Pdb),
                comm_mat_file=comm_mat_file,
                joint_mat_file=joint_mat_file,
                mat_file=mat_file,
                mode=mode,
                setup_idx=setup_idx,
                model=model,
                K_use=K_use,
                M_use=M_use,
                N_use=N_use,
                sigma_w2=sigma_w2,
                max_iters=max_iters,
                epsilon=epsilon,
                init_delta=init_delta,
                init_lambda=init_lambda,
                w_c_user=w_c_user,
                w_s_user=w_s_user,
                verbose=False,
                save_plot_path=None,
            )

            setup_results.append(result)

            if result["mode"] == "only_comm":
                c = result["classical_result"]
                du_final = result.get("du_final_wsr", np.nan)
                if np.isfinite(du_final):
                    print(
                        f"[ONLY-COMM] setup={setup_idx:02d} | "
                        f"initial={c['initial_comm_wsr']:.6f} | "
                        f"classical={c['final_comm_wsr']:.6f} | "
                        f"DU={du_final:.6f}"
                    )
                else:
                    print(
                        f"[ONLY-COMM] setup={setup_idx:02d} | "
                        f"initial={c['initial_comm_wsr']:.6f} | "
                        f"classical={c['final_comm_wsr']:.6f} | "
                        f"DU=N/A"
                    )
                initial_vals.append(c["initial_comm_wsr"])
                classical_vals.append(c["final_comm_wsr"])
                lambda_vals.append(np.nan)
                comm_vals.append(c["final_comm_wsr"])
                sensing_vals.append(0.0)
                classical_histories.append(c["full_comm_history"])
                lambda_histories.append(np.array([]))
                du_vals.append(du_final)
                du_histories.append(result["unfolding_comm_history"] if result.get("unfolding_comm_history") is not None else np.array([]))

            elif result["mode"] == "comm_lambda":
                c = result["classical_result"]
                lq = result["lambda_qt_result"]
                du_final = result.get("du_final_wsr", np.nan)
                if np.isfinite(du_final):
                    print(
                        f"[COMM-LAMBDA] setup={setup_idx:02d} | "
                        f"initial={c['initial_comm_wsr']:.6f} | "
                        f"classical={c['final_comm_wsr']:.6f} | "
                        f"lambda={lq['final_comm_wsr']:.6f} | "
                        f"DU={du_final:.6f}"
                    )
                else:
                    print(
                        f"[COMM-LAMBDA] setup={setup_idx:02d} | "
                        f"initial={c['initial_comm_wsr']:.6f} | "
                        f"classical={c['final_comm_wsr']:.6f} | "
                        f"lambda={lq['final_comm_wsr']:.6f} | "
                        f"DU=N/A"
                    )
                initial_vals.append(c["initial_comm_wsr"])
                classical_vals.append(c["final_comm_wsr"])
                lambda_vals.append(lq["final_comm_wsr"])
                comm_vals.append(c["final_comm_wsr"])
                sensing_vals.append(0.0)
                classical_histories.append(c["full_comm_history"])
                lambda_histories.append(lq["full_comm_history"])
                du_vals.append(du_final)
                du_histories.append(result["unfolding_comm_history"] if result.get("unfolding_comm_history") is not None else np.array([]))

            elif result["mode"] == "joint_lambda":
                c = result["classical_result"]
                lq = result["lambda_qt_result"]
                du_final = result.get("deep_final_joint_wsr", np.nan)
                print(
                    f"[JOINT-LAMBDA] setup={setup_idx:02d} | "
                    f"initial={result['initial_joint_wsr']:.6f} | "
                    f"classical={c['final_wsr']:.6f} | "
                    f"lambda={lq['final_joint_wsr']:.6f} | "
                    f"DU={du_final:.6f} | "
                    f"comm_raw={c['final_comm_wsr']:.6f} | "
                    f"sens_raw={c['final_sensing_raw_wsr']:.6f}"
                )
                initial_vals.append(result["initial_joint_wsr"])
                classical_vals.append(c["final_wsr"])
                lambda_vals.append(lq["final_joint_wsr"])
                comm_vals.append(c["final_comm_wsr"])
                sensing_vals.append(c["final_sensing_wsr"])
                classical_histories.append(c["full_joint_history"])
                lambda_histories.append(lq["full_joint_history"])
                du_vals.append(du_final)
                du_histories.append(result.get("deep_joint_history", np.array([])))

            else:
                print(
                    f"[JOINT] setup={setup_idx:02d} | "
                    f"initial={result['initial_joint_wsr']:.6f} | "
                    f"final_joint={result['final_wsr']:.6f} | "
                    f"comm_raw={result['final_comm_wsr']:.6f} | "
                    f"comm_contrib={result['final_comm_contribution']:.6f} | "
                    f"sens_raw={result['final_sensing_raw_wsr']:.6f} | "
                    f"sens_contrib={result['final_sensing_contribution']:.6f}"
                )
                du_final = result.get("deep_final_joint_wsr", np.nan)
                if np.isfinite(du_final):
                    print(
                        f"[JOINT-DU] setup={setup_idx:02d} | "
                        f"DU_joint={du_final:.6f} | "
                        f"DU_comm_raw={result['deep_final_comm_wsr']:.6f} | "
                        f"DU_sens_raw={result['deep_final_sensing_raw_wsr']:.6f} | "
                        f"DU_sens_contrib={result['deep_final_sensing_contribution']:.6f} | "
                        f"DU-FP/QT={du_final - result['final_wsr']:.6f}"
                    )
                else:
                    print(f"[JOINT-DU] setup={setup_idx:02d} | DU=N/A")
                initial_vals.append(result["initial_joint_wsr"])
                classical_vals.append(result["final_wsr"])
                lambda_vals.append(np.nan)
                comm_vals.append(result["final_comm_wsr"])
                sensing_vals.append(result["final_sensing_wsr"])
                classical_histories.append(result["full_joint_history"])
                lambda_histories.append(np.array([]))
                du_vals.append(du_final)
                du_histories.append(result.get("deep_joint_history", np.array([])))

        all_results[float(Pdb)] = setup_results
        initial_by_power.append(float(np.mean(initial_vals)))
        classical_by_power.append(float(np.mean(classical_vals)))
        lambda_by_power.append(float(np.nanmean(lambda_vals)) if np.any(np.isfinite(lambda_vals)) else np.nan)
        comm_by_power.append(float(np.mean(comm_vals)))
        sensing_by_power.append(float(np.mean(sensing_vals)))
        classical_history_by_power.append(pad_with_last_value(classical_histories).mean(axis=0))

        valid_lambda_histories = [h for h in lambda_histories if len(np.asarray(h).reshape(-1)) > 0]
        if valid_lambda_histories:
            lambda_history_by_power.append(pad_with_last_value(valid_lambda_histories).mean(axis=0))
        else:
            lambda_history_by_power.append(np.array([]))

        finite_du_vals = [v for v in du_vals if np.isfinite(v)]
        du_by_power.append(float(np.mean(finite_du_vals)) if finite_du_vals else np.nan)
        valid_du_histories = [h for h in du_histories if len(np.asarray(h).reshape(-1)) > 0]
        if valid_du_histories:
            du_history_by_power.append(pad_with_last_value(valid_du_histories).mean(axis=0))
        else:
            du_history_by_power.append(np.array([]))

        if mode in ["only_comm", "comm_only", "comm_fpqt"]:
            print(
                f"[POWER MEAN] P={Pdb:.1f} dB | "
                f"initial={initial_by_power[-1]:.6f} | "
                f"classical={classical_by_power[-1]:.6f} | "
                f"DU={du_by_power[-1]:.6f} | "
                f"comm_raw={comm_by_power[-1]:.6f}"
            )
        else:
            print(
                f"[POWER MEAN] P={Pdb:.1f} dB | "
                f"initial={initial_by_power[-1]:.6f} | "
                f"classical={classical_by_power[-1]:.6f} | "
                f"lambda={lambda_by_power[-1]:.6f} | "
                f"DU={du_by_power[-1]:.6f} | "
                f"comm_raw={comm_by_power[-1]:.6f} | "
                f"sensing_raw={sensing_by_power[-1]:.6f}"
            )

    power_list_np = np.asarray(power_list, dtype=float)
    initial_by_power = np.asarray(initial_by_power, dtype=float)
    classical_by_power = np.asarray(classical_by_power, dtype=float)
    lambda_by_power = np.asarray(lambda_by_power, dtype=float)
    comm_by_power = np.asarray(comm_by_power, dtype=float)
    sensing_by_power = np.asarray(sensing_by_power, dtype=float)
    du_by_power = np.asarray(du_by_power, dtype=float)

    if save_plots:
        plt.figure(figsize=(8, 6))

        if mode in ["only_comm", "comm_only", "comm_fpqt"]:
            # Only communication FP/QT mode: no D/lambda-QT is run or plotted.
            # Keep the initial/EPA curve as requested.
            plt.plot(power_list_np, initial_by_power, "--s", linewidth=2, label="Initial EPA")
            plt.plot(power_list_np, classical_by_power, "-o", linewidth=2, label="Only communication FP/QT")
            if np.any(np.isfinite(du_by_power)):
                plt.plot(power_list_np, du_by_power, "-^", linewidth=2, label="Communication deep unfolding")
            plt.ylabel("Mean communication WSR")
            plt.title("Only Communication: EPA vs FP/QT vs DU")
            out_plot = "clean_only_comm_epa_fpqt_du.png"

        elif mode in ["comm_lambda", "comm_only_compare", "communication_lambda", "comm"]:
            # Communication-only comparison mode: FP/QT + D/lambda-QT + DU.
            # Keep the initial/EPA curve as requested.
            plt.plot(power_list_np, initial_by_power, "--s", linewidth=2, label="Initial EPA")
            plt.plot(power_list_np, classical_by_power, "-o", linewidth=2, label="Communication-only FP/QT")
            if np.any(np.isfinite(lambda_by_power)):
                plt.plot(power_list_np, lambda_by_power, "-d", linewidth=2, label="Communication D/lambda-QT")
            if np.any(np.isfinite(du_by_power)):
                plt.plot(power_list_np, du_by_power, "-^", linewidth=2, label="Communication deep unfolding")
            plt.ylabel("Mean communication WSR")
            plt.title("Communication-only: EPA vs FP/QT vs D/lambda-QT vs DU")
            out_plot = "clean_comm_epa_fpqt_lambda_du.png"

        elif mode in ["joint_lambda", "joint_comm_sensing_lambda"]:
            # Joint comparison mode: EPA + classical FP/QT + D/lambda-QT + DU.
            plt.plot(power_list_np, initial_by_power, "--s", linewidth=2, label="Initial EPA")
            plt.plot(power_list_np, classical_by_power, "-o", linewidth=2, label="Joint FP/QT")
            if np.any(np.isfinite(lambda_by_power)):
                plt.plot(power_list_np, lambda_by_power, "-d", linewidth=2, label="Joint D/lambda-QT")
            if np.any(np.isfinite(du_by_power)):
                plt.plot(power_list_np, du_by_power, "-^", linewidth=2, label="Joint deep unfolding")
            plt.ylabel("Mean joint objective")
            plt.title("Joint Communication-Sensing: EPA vs FP/QT vs D/lambda-QT vs DU")
            out_plot = "clean_joint_epa_fpqt_lambda_du.png"

            # ====================================================================
            # Per-power convergence plots
            # Saves one convergence plot for each Pmax in power_list
            # ====================================================================
            conv_dir = f"convergence_plots_{mode}"
            os.makedirs(conv_dir, exist_ok=True)

            for p_idx, Pdb in enumerate(power_list_np):
                plt.figure(figsize=(8, 5.5))

                # Classical FP/QT convergence
                classical_hist = np.asarray(classical_history_by_power[p_idx], dtype=float).reshape(-1)
                if classical_hist.size > 0:
                    plt.plot(
                        np.arange(classical_hist.size),
                        classical_hist,
                        "-o",
                        linewidth=2,
                        label="Classical FP/QT",
                    )

                # D/lambda-QT convergence
                lambda_hist = np.asarray(lambda_history_by_power[p_idx], dtype=float).reshape(-1)
                if lambda_hist.size > 0:
                    plt.plot(
                        np.arange(lambda_hist.size),
                        lambda_hist,
                        "-d",
                        linewidth=2,
                        label="D/lambda QT",
                    )

                # Deep unfolding layer-wise trajectory
                du_hist = np.asarray(du_history_by_power[p_idx], dtype=float).reshape(-1)
                if du_hist.size > 0:
                    plt.plot(
                        np.arange(du_hist.size),
                        du_hist,
                        "-^",
                        linewidth=2,
                        label="Deep unfolding",
                    )

                # Initial EPA reference
                plt.axhline(
                    initial_by_power[p_idx],
                    linestyle="--",
                    linewidth=1.8,
                    label="EPA initial",
                )

                plt.xlabel("Iteration / DU layer")
                if mode in ["only_comm", "comm_only", "comm_fpqt", "comm_lambda", "comm_only_compare"]:
                    plt.ylabel("Communication WSR")
                    plt.title(f"Communication convergence @ Pmax = {Pdb:g} dB")
                else:
                    plt.ylabel("Joint objective")
                    plt.title(f"Joint convergence @ Pmax = {Pdb:g} dB")

                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                out_path = os.path.join(
                    conv_dir,
                    f"convergence_{mode}_Pmax_{Pdb:g}dB.png"
                )
                plt.savefig(out_path, dpi=300)
                print(f"[CONVERGENCE PLOT SAVED] {out_path}")
                plt.close()
        else:
            # Joint plot keeps the initial/EPA curve because it is useful for joint comparison.
            plt.plot(power_list_np, initial_by_power, "--s", linewidth=2, label="Before optimization")
            plt.plot(power_list_np, classical_by_power, "-o", linewidth=2, label="Joint FP/QT")
            if np.any(np.isfinite(du_by_power)):
                plt.plot(power_list_np, du_by_power, "-^", linewidth=2, label="Joint deep unfolding")
            plt.ylabel("Mean joint objective")
            plt.title("Joint Communication-Sensing: EPA vs FP/QT vs DU")
            out_plot = "clean_joint_fpqt_du.png"

        plt.xlabel("Transmit Power $P_{max}$ (dB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_plot, dpi=300)
        print(f"[PLOT SAVED] {out_plot}")
        plt.show()

    return {
        "mode": mode,
        "power_list": power_list_np,
        "initial_by_power": initial_by_power,
        "classical_by_power": classical_by_power,
        "lambda_by_power": lambda_by_power,
        "comm_by_power": comm_by_power,
        "sensing_by_power": sensing_by_power,
        "classical_history_by_power": classical_history_by_power,
        "lambda_history_by_power": lambda_history_by_power,
        "du_by_power": du_by_power,
        "du_history_by_power": du_history_by_power,
        "all_results": all_results,
        # aliases compatible with your earlier plotting names
        "before_wsr_by_power": initial_by_power,
        "classical_wsr_by_power": classical_by_power,
        "lambda_wsr_by_power": lambda_by_power,
        "du_wsr_by_power": du_by_power,
    }




# =============================================================================
# Script entry point: configure training/testing and launch the clean power sweep
# =============================================================================
# Edit RUN_MODE, dataset paths, training hyperparameters, and power lists here.
# The code below is executed only when this file is run as a script.
if __name__ == "__main__":
    np.random.seed(42)

    # =====================================================================
    # CLEAN TRAIN + SAVE STEP SIZES + TEST SETTINGS
    # =====================================================================
    # Change only RUN_MODE for the experiment type:
    #   RUN_MODE = "only_comm"    -> communication-only FP/QT + optional DU, no D/lambda-QT
    #   RUN_MODE = "comm_lambda"  -> communication-only FP/QT + D/lambda-QT + optional DU
    #   RUN_MODE = "joint"        -> joint communication-sensing DU + classical FP/QT
    #   RUN_MODE = "joint_lambda" -> EPA + joint classical + corrected joint D/lambda-QT + joint DU
    RUN_MODE = "joint_lambda"

    # This does what your older code did: create a fresh model, train it,
    # save the learned step sizes, reload those step sizes into a fresh model,
    # and test using that model. No manual MODEL_PATH is required.
    TRAIN_MODEL = True
    TEST_WITH_SAVED_STEP_SIZES = True

    # Dataset paths. Only the one matching RUN_MODE is used for training/testing.
    COMM_MAT_FILE_TRAIN = "jointopt_unfolding_dataset_50_setups.mat"   
    COMM_MAT_FILE_TEST = "12th_may_ideal_dataset_with_alpha_var.mat"
    JOINT_MAT_FILE_TRAIN = "jointopt_unfolding_dataset_50_setups.mat"  
    JOINT_MAT_FILE_TEST = "JointOpt_Dataset_with_V.mat"

    K_use = 8
    M_use = 16
    N_use = 16

    # DU architecture/training hyperparameters.
    num_layers = 30
    num_pgd_steps = 5
    init_step_bar = 0.07
    init_step_tilda = 0.07
    init_step_prime = 0.07
    learning_rate = 1e-2
    num_epochs = 10
    batch_size = 10

    # Train/test split. Keep these disjoint.
    train_setups = np.arange(0, 40)
    test_setups = np.arange(41,50)
    train_power_list = np.array([10.0])
    power_list = np.arange(-20, 31, 5)

    # Power/objective split.
    # For RUN_MODE="only_comm" or "comm_lambda", the code automatically uses init_delta=0, w_c=1, w_s=0.
    init_delta = 0.5
    init_lambda = 0.5
    w_c_user = 0.2
    w_s_user = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("du_artifacts", f"{RUN_MODE}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    model = None
    train_logs = None
    step_paths = None

    if TRAIN_MODEL:
        model = create_unfolding_model_for_mode(
            RUN_MODE,
            K_use=K_use,
            num_layers=num_layers,
            num_pgd_steps=num_pgd_steps,
            init_step_bar=init_step_bar,
            init_step_tilda=init_step_tilda,
            init_step_prime=init_step_prime,
            enforce_full_power=False,
            device=device,
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_logs = train_unfolding_model_by_mode(
            run_mode=RUN_MODE,
            comm_mat_file=COMM_MAT_FILE_TRAIN,
            joint_mat_file=JOINT_MAT_FILE_TRAIN,
            model=model,
            optimizer=optimizer,
            train_setups=train_setups,
            train_power_list=train_power_list,
            K_use=K_use,
            M_use=M_use,
            N_use=N_use,
            sigma_w2=1.0,
            num_epochs=num_epochs,
            batch_size=batch_size,
            init_delta=init_delta,
            init_lambda=init_lambda,
            w_c_user=w_c_user,
            w_s_user=w_s_user,
            shuffle_batches=True,
            seed=42,
        )

        learned_steps = print_learned_step_sizes(model)

        config = {
            "RUN_MODE": RUN_MODE,
            "K_use": K_use,
            "M_use": M_use,
            "N_use": N_use,
            "num_layers": num_layers,
            "num_pgd_steps": num_pgd_steps,
            "train_setups": train_setups.tolist(),
            "test_setups": test_setups.tolist(),
            "train_power_list": train_power_list.tolist(),
            "test_power_list": power_list.tolist(),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "init_delta": init_delta,
            "init_lambda": init_lambda,
            "w_c_user": w_c_user,
            "w_s_user": w_s_user,
            "learning_rate": learning_rate,
        }
        step_paths = save_learned_step_sizes(
            model,
            out_dir=out_dir,
            prefix=f"learned_step_sizes_{RUN_MODE}",
            config=config,
        )
        checkpoint_path = save_unfolding_model_checkpoint(
            model,
            out_dir=out_dir,
            prefix=f"learned_model_{RUN_MODE}",
            config=config,
        )

        if TEST_WITH_SAVED_STEP_SIZES:
            # This proves that testing uses the saved learned step sizes, not an external path.
            test_model = create_unfolding_model_for_mode(
                RUN_MODE,
                K_use=K_use,
                num_layers=num_layers,
                num_pgd_steps=num_pgd_steps,
                init_step_bar=init_step_bar,
                init_step_tilda=init_step_tilda,
                init_step_prime=init_step_prime,
                enforce_full_power=False,
                device=device,
            )
            test_model = load_step_sizes_into_model(test_model, step_paths["npz"], device=device)
        else:
            test_model = model
    else:
        # Classical-only run. DU is skipped because model=None.
        test_model = None

    results = run_clean_power_sweep(
        power_list=power_list,
        test_setups=test_setups,
        comm_mat_file=COMM_MAT_FILE_TEST,
        joint_mat_file=JOINT_MAT_FILE_TEST,
        mode=RUN_MODE,
        model=test_model,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        sigma_w2=1.0,
        max_iters=500,
        epsilon=1e-4,
        init_delta=init_delta,
        init_lambda=init_lambda,
        w_c_user=w_c_user,
        w_s_user=w_s_user,
        save_plots=True,
    )

    if step_paths is not None:
        print("\nSaved artifacts:")
        print(f"  Step sizes TXT : {step_paths['txt']}")
        print(f"  Step sizes NPZ : {step_paths['npz']}")
        print(f"  Model checkpoint: {checkpoint_path}")
