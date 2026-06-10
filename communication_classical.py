"""Classical-only FP/QT WSR optimization.

Removed: all PyTorch/deep-unfolding model, training loop, DU testing, and DU plots.
"""

import numpy as np #type: ignore
import time
import matplotlib.pyplot as plt #type: ignore
from scipy.io import loadmat #type: ignore

from kappa import (compute_kappa_S, compute_kappa_V1, compute_kappa_V0,
                   compute_kappa_K1, compute_kappa_K0,
                   compute_kappa_DAC1, compute_kappa_DAC0,compute_kappa_Th1,compute_kappa_ADC1,
                   compute_g_lik)
from kappa_mui import build_kappa_MUI
from matrix_builders import (build_all_matrices, build_R_tilda_all, build_C_yy_all, build_G_list)

   
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

# MAIN optimisation loop
def run_wsr_optimisation(
    K, w, alpha_list,
    tau, d, sigma_w2, sigma_d2,
    U_list, V_list, Sigma_list,
    R_list, h_bar_list,
    C_h_ref_all, C_h_total_all,D_Th_list,D_ADC_list,
    A_list, Theta_list, Z_list,G_list,
    # For P_th and P_adc
    W_list, T,
    U_q_list, V_q_list,g_lik,
    P_bar_init=None, P_tilda_init=None,
    max_iters=200, epsilon=1e-6,
    P_total_max=None,
    eps_power=1e-12,
    verbose=False
):
    P_bar   = np.ones(K) if P_bar_init   is None else np.asarray(P_bar_init,   float)
    P_tilda = np.ones(K) if P_tilda_init is None else np.asarray(P_tilda_init, float) 
    if verbose:
        print("Building kappa coefficients …")
    (kappa_S,
     kappa_V1, kappa_V0,
     kappa_K1, kappa_K0,
     kappa_DAC1, kappa_DAC0,kappa_Th1,kappa_ADC1,
     kappa_M1,  kappa_M0) = build_all_kappas(
        K, alpha_list, tau, d,
        U_list, V_list,U_q_list,V_q_list, Sigma_list,
        R_list, h_bar_list,
        C_h_ref_all, C_h_total_all,
        sigma_d2, sigma_w2,D_Th_list, D_ADC_list,
        A_list, Theta_list, Z_list,
        g_lik)

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
        C_h_total_all=C_h_total_all
    )
    
    if verbose:
        print("Precomputing P_th and P_adc …")

    if verbose:
        print("Done.\n")
        print(f"{'Iter':>6}  {'WSR (bps/Hz)':>14}  {'|ΔWSR|':>12}")
        print("─" * 40)
    
    wsr_history = []
    wsr_prev    = -np.inf
    start_time = time.perf_counter()
    converged = False
    convergence_iter = max_iters
    for t in range(max_iters):
        P_th = P_bar * kappa_Th1 + kappa_Th0
        P_adc = P_bar * kappa_ADC1 + kappa_ADC0
        # Step 1: exact SINR
        gamma = np.array([
            compute_SINR_exact(
                k, K, kappa_S,
                kappa_V1, kappa_K1,
                kappa_V0, kappa_K0,
                kappa_DAC1, kappa_DAC0,
                kappa_M1, kappa_M0,
                kappa_Th1,kappa_ADC1,
                P_bar, P_tilda,P_th, P_adc
            )
            for k in range(K)
        ])

        # Step 2: auxiliary μ_k
        mu = update_auxiliary(
            K, w, gamma,
            kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1, kappa_ADC1,
            P_bar, P_tilda,P_th, P_adc
        )

        # Step 3: update P̄_k
        P_bar = compute_P_bar(
            K, P_tilda,w, kappa_S, kappa_V1, kappa_K1,
                    kappa_M1, kappa_Th1,
                    kappa_DAC1, kappa_ADC1,Z_list,gamma,mu)
        P_bar = clip_P_bar_given_P_tilda(
            P_bar, P_tilda, P_total_max, eps_power
        )

        P_tilda = compute_P_tilda(
                    K,P_bar, w,
                    kappa_S,
                    kappa_V1, kappa_K1,
                    kappa_V0, kappa_K0,
                    kappa_M1, kappa_M0,
                    gamma, mu)

        P_tilda = clip_P_tilda_given_P_bar(
            P_tilda, P_bar, P_total_max, eps_power
        )
     
        P_th = P_bar * kappa_Th1 + kappa_Th0
        P_adc = P_bar * kappa_ADC1 + kappa_ADC0
        # Step 5: WSR + convergence check
        wsr = compute_WSR_exact(
            K, w, kappa_S,
            kappa_V1, kappa_K1,
            kappa_V0, kappa_K0,
            kappa_DAC1, kappa_DAC0,
            kappa_M1, kappa_M0,
            kappa_Th1,kappa_ADC1,
            P_bar, P_tilda,P_th, P_adc,d,tau
        )

        gamma = np.array([
            compute_SINR_exact(
                k, K, kappa_S,
                kappa_V1, kappa_K1,
                kappa_V0, kappa_K0,
                kappa_DAC1, kappa_DAC0,
                kappa_M1, kappa_M0,
                kappa_Th1, kappa_ADC1,
                P_bar, P_tilda, P_th, P_adc
            )
            for k in range(K)
        ])

        wsr_history.append(wsr)
        delta_wsr = wsr - wsr_prev

        if t > 0 and delta_wsr < -1e-10:
            print(f"Warning: WSR decreased: {delta_wsr:.3e}")

        if t > 0 and abs(delta_wsr) < epsilon:

            converged = True
            convergence_iter = t + 1
            if verbose:
                print(f"\nConverged at iteration {t+1}  (|ΔWSR| = {delta_wsr:.2e})")
            break
     
        wsr_prev = wsr

    elapsed_time_sec = time.perf_counter() - start_time

    if not converged:
        convergence_iter = len(wsr_history)

    return wsr_history, P_bar, P_tilda, gamma, convergence_iter

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

    # alpha can be K×setups or K×1 or K
    alpha_dac_all = np.asarray(alpha_dac_all)

    if alpha_dac_all.ndim == 2 and alpha_dac_all.shape[1] > 1:
        alpha_list = np.real(alpha_dac_all[:K, setup_idx]).reshape(-1)
    else:
        alpha_list = np.real(alpha_dac_all).reshape(-1)[:K]

    A_list = mat_3d_ap_to_list(
        A_l_all,
        M,
        hermitian=True,
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
        hermitian=True,
    )

    Sigma_list = mat_4d_to_lk_list(
        C_h_dir_all_CF,
        K,
        M,
        hermitian=True,
    )

    C_h_ref_all = mat_4d_to_lk_list(
        C_h_ref_all_CF,
        K,
        M,
        hermitian=True,
    )

    C_h_total_all = mat_4d_to_lk_list(
        C_h_all_CF,
        K,
        M,
        hermitian=True,
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
      #  "Theta_list": Theta_list,
        "Z_list": Z_list,
    }


def run_one_setup_power(
    mat_file,
    Pmax_db,
    print_components=False,
    setup_idx=0,
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-6,
    verbose=True,
):
    """Run only the classical FP/QT WSR optimization for one setup and one power."""

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
 #   Theta_list = data["Theta_list"]
    Z_list = data["Z_list"]

    Pmax_lin = db2lin(Pmax_db)

    P_bar_init = 0.5 * Pmax_lin * np.ones(K)
    P_tilda_init = 0.5 * Pmax_lin * np.ones(K)
    P_tot_list = Pmax_lin * np.ones(K)

    sigma_d2 = alpha_list * (1.0 - alpha_list) * P_tot_list
    w = np.ones(K)
    W_list = build_W_list_MR(M, K, N)

    # Fixed channel-estimation reference power, same as your original code.
    P_bar_ref = 0.5 * Pmax_lin * np.ones(K)
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

    # Build g_lik
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

    # Kappas needed only to compute initial WSR and optional component print.
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

    P_bar_before = P_bar_init.copy()
    P_tilda_before = P_tilda_init.copy()

    P_th_before = P_bar_before * kappa_Th1 + kappa_Th0
    P_adc_before = P_bar_before * kappa_ADC1 + kappa_ADC0

    initial_wsr = compute_WSR_exact(
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

    # Classical optimization only.
    wsr_history, P_bar_opt, P_tilda_opt, gamma_opt, convergence_iter = run_wsr_optimisation(
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
        P_bar_init=P_bar_init,
        P_tilda_init=P_tilda_init,
        P_total_max=Pmax_lin,
        eps_power=1e-12,
        max_iters=max_iters,
        epsilon=epsilon,
        verbose=False,
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

    wsr_history = np.asarray(wsr_history, dtype=float)
    final_wsr = float(wsr_history[-1]) if len(wsr_history) else np.nan
    monotonic = bool(np.all(np.diff(wsr_history) >= -1e-10)) if len(wsr_history) else False

    if verbose:
        print(
            f"\n[Classical Baseline] "
            f"Pmax = {Pmax_db:>6.1f} dB | "
            f"Setup = {setup_idx:>3d} | "
            f"Initial WSR = {initial_wsr:.6f} | "
            f"Final WSR = {final_wsr:.6f} | "
            f"Iterations to convergence = {convergence_iter} | "
            f"Monotonic = {monotonic}"
        )

    return {
        "Pmax_db": Pmax_db,
        "setup_idx": setup_idx,
        "initial_wsr": float(initial_wsr),
        "final_wsr": final_wsr,
        "wsr_history": wsr_history,
        "P_bar_opt": P_bar_opt,
        "P_tilda_opt": P_tilda_opt,
        "gamma_opt": gamma_opt,
        "P_bar_initial": P_bar_before,
        "P_tilda_initial": P_tilda_before,
        "monotonic": monotonic,
    }


def pad_with_last_value(histories):
    histories = [np.asarray(h, dtype=float).reshape(-1) for h in histories]
    max_len = max(len(h) for h in histories)
    padded = np.zeros((len(histories), max_len), dtype=float)

    for i, h in enumerate(histories):
        padded[i, :len(h)] = h
        padded[i, len(h):] = h[-1]

    return padded


def run_classical_power_sweep(
    mat_file,
    power_list=np.arange(-20, 31, 5),
    test_setups=np.arange(0, 1),
    K_use=8,
    M_use=16,
    N_use=16,
    sigma_w2=1.0,
    max_iters=200,
    epsilon=1e-6,
    save_plots=True,
):
    """Run classical baseline over powers and setups. No unfolding/training."""

    before_wsr_by_power = []
    classical_wsr_by_power = []
    classical_history_by_power = []
    all_results = {}

    for Pdb in power_list:
        before_vals = []
        classical_vals = []
        classical_histories = []
        all_results[float(Pdb)] = []

        print("\n" + "=" * 90)
        print(f"Testing classical FP/QT for Pmax = {Pdb:.1f} dB")
        print("=" * 90)

        for setup_idx in test_setups:
            result = run_one_setup_power(
                mat_file=mat_file,
                Pmax_db=float(Pdb),
                print_components=False,
                setup_idx=int(setup_idx),
                K_use=K_use,
                M_use=M_use,
                N_use=N_use,
                sigma_w2=sigma_w2,
                max_iters=max_iters,
                epsilon=epsilon,
                verbose=False,
            )

            print(
                f"[setup {setup_idx:02d}] "
                f"Pmax = {Pdb:>6.1f} dB | "
                f"Before WSR = {result['initial_wsr']:.6f} | "
                f"Classical WSR = {result['final_wsr']:.6f} | "
                f"Gain = {result['final_wsr'] - result['initial_wsr']:.6f} | "
                f"Iterations = {len(result['wsr_history'])}"
            )

            before_vals.append(result["initial_wsr"])
            classical_vals.append(result["final_wsr"])
            classical_histories.append(
                np.concatenate([[result["initial_wsr"]], result["wsr_history"]])
            )
            all_results[float(Pdb)].append(result)

        before_mean = float(np.mean(before_vals))
        classical_mean = float(np.mean(classical_vals))
        classical_mean_history = pad_with_last_value(classical_histories).mean(axis=0)

        before_wsr_by_power.append(before_mean)
        classical_wsr_by_power.append(classical_mean)
        classical_history_by_power.append(classical_mean_history)

        print(
            f"[TEST] P={Pdb:>6.1f} dB | "
            f"Before mean = {before_mean:.6f} | "
            f"Classical mean = {classical_mean:.6f} | "
            f"Gain = {classical_mean - before_mean:.6f}"
        )

        if save_plots:
            classical_iterations = np.arange(len(classical_mean_history))
            plt.figure(figsize=(7, 5))
            plt.plot(
                classical_iterations,
                classical_mean_history,
                marker="o",
                linewidth=2,
                label="Mean Classical FP/QT",
            )
            plt.xlabel("Classical iteration")
            plt.ylabel("Mean WSR / Sum SE (bit/s/Hz)")
            plt.title(f"Mean Classical FP/QT Convergence @ {Pdb} dB")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"mean_classical_convergence_{Pdb}dB.png", dpi=300)
            plt.show()

    before_wsr_by_power = np.asarray(before_wsr_by_power, dtype=float)
    classical_wsr_by_power = np.asarray(classical_wsr_by_power, dtype=float)

    print("\nMean WSR vs Power over Test Setups")
    print(
        f"{'Power(dB)':>10} | "
        f"{'Before':>12} | "
        f"{'Classical':>12} | "
        f"{'Gain':>12}"
    )
    print("-" * 56)

    for i, Pdb in enumerate(power_list):
        gain = classical_wsr_by_power[i] - before_wsr_by_power[i]
        print(
            f"{Pdb:>10.1f} | "
            f"{before_wsr_by_power[i]:>12.6f} | "
            f"{classical_wsr_by_power[i]:>12.6f} | "
            f"{gain:>12.6f}"
        )

    if save_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(
            power_list,
            before_wsr_by_power,
            "--s",
            linewidth=2,
            label="Before Optimization",
        )
        plt.plot(
            power_list,
            classical_wsr_by_power,
            "-o",
            linewidth=2,
            label="Classical FP/QT",
        )
        plt.xlabel("Transmit Power $P_{max}$ (dB)")
        plt.ylabel("Mean WSR / Sum SE (bit/s/Hz)")
        plt.title("Mean Classical WSR vs Power over Test Setups")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("classical_mean_wsr_vs_power.png", dpi=300)
        plt.show()

    return {
        "power_list": np.asarray(power_list, dtype=float),
        "before_wsr_by_power": before_wsr_by_power,
        "classical_wsr_by_power": classical_wsr_by_power,
        "classical_history_by_power": classical_history_by_power,
        "all_results": all_results,
    }


if __name__ == "__main__":
    np.random.seed(42)

    # Change this to your actual .mat dataset path.
     ### mat_file = "12th_may_ideal_dataset.mat"
    mat_file = "jointopt_unfolding_dataset_50_setups.mat"
    # mat_file = "12th_may_ideal_dataset_with_alpha.mat"
    # mat_file = "one_setup_required_values_10dB.mat"
    K_use = 8
    M_use = 16
    N_use = 16

    # Current setting uses one setup, matching your uploaded code.
    # For 10 setups, use: test_setups = np.arange(0, 10)
    test_setups = np.arange(40,50)

    power_list = np.arange(0, 10, 10)

    results = run_classical_power_sweep(
        mat_file=mat_file,
        power_list=power_list,
        test_setups=test_setups,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
        sigma_w2=1.0,
        max_iters=1000,
        epsilon=1e-4,
        save_plots=False,
    )
