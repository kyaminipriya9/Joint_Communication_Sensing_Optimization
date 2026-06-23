import numpy as np


# HOW TO CALL THESE FUNCTIONS

# All inputs below are PER-USER-k slices, i.e. 1-D over panels [l].
# Extract them from the 2-D [l][k] outputs of matrix_utils like this:
#
#   k = 0   # user of interest
#   U_k      = [U_list[l][k]        for l in range(L)]
#   V_k      = [V_list[l][k]        for l in range(L)]
#   Sigma_k  = [Sigma_list[l][k]    for l in range(L)]
#   R_k      = [R_list[l][k]        for l in range(L)]
#   h_bar_k  = [h_bar_list[l][k]    for l in range(L)]
#   C_ref_k  = [C_h_ref_all[l][k]   for l in range(L)]
#
# For functions that also need cross-user data (kappa_V0, kappa_K0),
# pass the full 2-D structures:
#   h_bar_list   : [l][i]  for all users i
#   C_h_total    : [l][i]  for all users i



# 
# κ_S,k  (Eq. 1)
#
# κ_S,k = α_k^4 T^4 (Σ_l Tr(U_lk Σ_lk)) (Σ_l Tr(V_lk Σ_lk))
#
# Inputs (all 1-D over l, for fixed k):
#   U_list    : [U_lk]     length L
#   V_list    : [V_lk]     length L
#   Sigma_list: [Σ_lk^dir] length L
# 
def compute_kappa_S(alpha_k, tau, U_list, V_list, Sigma_list):

    trace_U_sum = 0.0
    trace_V_sum = 0.0

    for l in range(len(U_list)):
        trace_U_sum += np.trace(U_list[l] @ Sigma_list[l])
        trace_V_sum += np.trace(V_list[l] @ Sigma_list[l])

    alpha_k = float(alpha_k)
    tau = float(tau)
    kappa = (alpha_k ** 4) * (tau ** 4) * trace_U_sum * trace_V_sum

    return np.real(kappa)


# 
# κ_V,k^(1)  (Eq. 2)
#
# Inputs (all 1-D over l, for fixed k):
#   U_list    : [U_lk]        length L
#   V_list    : [V_lk]        length L
#   Sigma_list: [Σ_lk^dir]    length L
#   R_list    : [R_lk]        length L
#   h_bar_list: [h̄_lk]        length L,  each shape (N,1)
#   C_h_ref   : [C_h,lk^ref]  length L
# 
def compute_kappa_V1(alpha_k, tau,
                     U_list, V_list, Sigma_list,
                     R_list, h_bar_list, C_h_ref):

    L = len(U_list)
    sum_terms = 0.0
    diag_sum  = 0.0

    for l in range(L):
        U     = U_list[l]
        V     = V_list[l]
        Sigma = Sigma_list[l]
        R     = R_list[l]
        h     = h_bar_list[l]      # h̄_lk  shape (N,1)
        Cref  = C_h_ref[l]

        t1 = np.trace(Sigma @ U @ Sigma @ V)
        t2 = np.trace(R @ U) * np.trace(R @ V)
        t3 = np.trace(R @ U) * np.real((h.conj().T @ V @ h)[0, 0])
        t4 = np.real((h.conj().T @ U @ h)[0, 0] * np.trace(R @ V))
        t5 = np.trace(Cref @ U @ Sigma @ V)

        sum_terms += (t1 + t2 + t3 + t4 + t5)

        # l==m diagonal of the cross sum (subtracted per Eq. 2)
        diag_sum += np.trace(U @ Sigma) * np.trace(V @ Sigma)
    alpha_k = float(alpha_k)
    tau = float(tau)
    kappa = (alpha_k ** 4) * (tau ** 4) * (sum_terms - diag_sum)

    return np.real(kappa)


# 
# κ_V,k^(0)  (Eq. 3)
#
# Inputs:
#   k           : user index
#   U_list      : [U_lk]        1-D over l  (fixed k)
#   V_list      : [V_lk]        1-D over l  (fixed k)
#   Sigma_list  : [Σ_lk^dir]    1-D over l  (fixed k)
#   R_list      : [R_lk]        1-D over l  (fixed k)
#   h_bar_list  : FULL 2-D [l][i] for ALL users i  ← needed for cross terms
#   C_h_ref     : [C_h,lk^ref]  1-D over l  (fixed k)
#   C_h_total   : FULL 2-D [l][i] for ALL users i
#   sigma_d2    : list length K
#   sigma_w2    : scalar
# 
def compute_kappa_V0(alpha_k, tau, k,
                     U_list, V_list, Sigma_list,
                     R_list, h_bar_list,
                     C_h_ref, C_h_total,
                     sigma_d2, sigma_w2,
                     U_q_list, V_q_list, Theta_list):

    L = len(U_list)
    K = len(sigma_d2)

    kappa = 0.0

    # ── (A) Σ_l Σ_{i≠k} σ²_{d,i} Tr(C_{h,li} U_lk Σ_lk V_lk) ──────────────
    for l in range(L):
        U     = U_list[l]
        V     = V_list[l]
        Sigma = Sigma_list[l]
        for i in range(K):
            if i == k:
                continue
            kappa += (
                alpha_k**2 * sigma_d2[i] * tau *
                np.trace(C_h_total[l][i] @ U @ Sigma @ V)
            )

    # ── (B) Σ_l σ²_{d,k} (T1+T2+T3+T4+T5) ─────────────
    for l in range(L):
        U     = U_list[l]
        V     = V_list[l]
        Sigma = Sigma_list[l]
        R     = R_list[l]
        h     = h_bar_list[l][k]    # h̄_lk  — use 2-D h_bar_list
        Cref  = C_h_ref[l]

        t1 = np.trace(Sigma @ U @ Sigma @ V)
        t2 = np.trace(R @ U) * np.trace(R @ V)
        t3 = np.trace(R @ U) * np.real((h.conj().T @ V @ h)[0, 0])
        t4 = np.real((h.conj().T @ U @ h)[0, 0] * np.trace(R @ V))
        t5 = np.trace(Cref @ U @ Sigma @ V)
        alpha_k = float(alpha_k)
        tau = float(tau)
        kappa += alpha_k**2 * sigma_d2[k] * tau * (t1 + t2 + t3 + t4 + t5)

    # ── (C) Σ_{l≠m} cross terms ──────────
    for l in range(L):
        for m in range(L):
            if l == m:
                continue

            U_l = U_list[l]
            V_m = V_list[m]

            # (i≠k) part:  h̄_{li}^H U_lk h̄_{lk} · h̄_{mk}^H V_mk h̄_{mi}
            for i in range(K):
                if i == k:
                    continue
                h_li = h_bar_list[l][i]
                h_lk = h_bar_list[l][k]
                h_mk = h_bar_list[m][k]
                h_mi = h_bar_list[m][i]
                a = (h_li.conj().T @ U_l @ h_lk).item()
                b = (h_mk.conj().T @ V_m @ h_mi).item()
                kappa += (
                    alpha_k**2 * tau * sigma_d2[i] * a * b
                )

            # σ²_{d,k} Tr(U_lk Σ_lk) Tr(V_mk Σ_mk)
            kappa += (
                alpha_k**2 * tau * sigma_d2[k] *
                np.trace(U_l @ Sigma_list[l]) *
                np.trace(V_m @ Sigma_list[m])
            )

    # ── (D) Noise term 
    for l in range(L):
        kappa += (
            alpha_k**2 * sigma_w2 * tau *
            np.trace(U_list[l] @ Sigma_list[l] @ V_list[l])
        )

    # ── (E) Quantization term
    for l in range(L): 
        kappa += (
            alpha_k**2 * tau *
            np.trace(U_q_list[l] @ Sigma_list[l] @ V_q_list[l] @ Theta_list[l])
        )
    alpha_k = float(alpha_k)
    tau = float(tau)
    return np.real(kappa * (tau ** 2))

# 
# κ_K,k^(1)  (Eq. 4)
#
# Inputs (all 1-D over l, for fixed k):
#   U_list    : [U_lk]        length L
#   V_list    : [V_lk]        length L
#   Sigma_list: [Σ_lk^dir]    length L
#   C_h_ref   : [C_h,lk^ref]  length L
# 
def compute_kappa_K1(alpha_k, tau,
                     U_list, V_list, Sigma_list,
                     C_h_ref):

    L = len(U_list)
    sum_main  = 0.0
    cross_sum = 0.0

    for l in range(L):
        U    = U_list[l]
        V    = V_list[l]
        Cref = C_h_ref[l]

        t1 = np.trace(V @ Sigma_list[l] @ U @ Cref)
        t2 = np.trace(Cref @ U @ Cref @ V)
        t3 = np.trace(Cref @ U) * np.trace(Cref @ V)

        sum_main += (t1 + t2 + t3)

    for l in range(L):
        for m in range(L):
            if l == m:
                continue
            cross_sum += (
                np.trace(U_list[l] @ C_h_ref[l]) *
                np.trace(V_list[m] @ C_h_ref[m])
            )
    alpha_k = float(alpha_k)
    tau = float(tau)
    kappa = (alpha_k ** 4) * (tau ** 4) * (sum_main + cross_sum)

    return np.real(kappa)


# 
# κ_K,k^(0)  (Eq. 5)
#
# Inputs:
#   k           : user index
#   U_list      : [U_lk]        1-D over l  (fixed k)
#   V_list      : [V_lk]        1-D over l  (fixed k)
#   Sigma_list  : [Σ_lk^dir]    1-D over l  (fixed k)
#   C_h_ref     : [C_h,lk^ref]  1-D over l  (fixed k)
#   C_h_total   : FULL 2-D [l][i] for ALL users i
#   sigma_d2    : list length K
#   sigma_w2    : scalar
# 
def compute_kappa_K0(alpha_k, tau, k,
                     U_list, V_list, Sigma_list,
                     C_h_ref, C_h_total,
                     sigma_d2, sigma_w2,
                     U_q_list, V_q_list, Theta_list):

    L = len(U_list)
    K = len(sigma_d2)

    kappa = 0.0

    # ── (A) Σ_l Σ_{i≠k} σ²_{d,i} Tr(C_{h,li} U_lk C_h,lk^ref V_lk) ─────────
    for l in range(L):
        U    = U_list[l]
        V    = V_list[l]
        Cref = C_h_ref[l]
        for i in range(K):
            if i == k:
                continue
            kappa += (
                alpha_k**2 * sigma_d2[i] * tau *
                np.trace(C_h_total[l][i] @ U @ Cref @ V)
            )

    # ── (B) Σ_l σ²_{d,k} (T1+T2+T3) ────
    for l in range(L):
        U     = U_list[l]
        V     = V_list[l]
        Sigma = Sigma_list[l]
        Cref  = C_h_ref[l]

        t1 = np.trace(Cref @ V @ Sigma @ U)
        t2 = np.trace(Cref @ U @ Cref @ V)
        t3 = np.trace(Cref @ U) * np.trace(Cref @ V)

        kappa += alpha_k**2 * sigma_d2[k] * tau * (t1 + t2 + t3)

    # ── (C) Σ_{l≠m} σ²_{d,k} Tr(U_lk C_h,lk^ref) Tr(V_mk C_h,mk^ref) ───────
    for l in range(L):
        for m in range(L):
            if l == m:
                continue
            kappa += (
                alpha_k**2 * sigma_d2[k] * tau *
                np.trace(U_list[l] @ C_h_ref[l]) *
                np.trace(V_list[m] @ C_h_ref[m])
            )

    # ── (D) Noise term 
    for l in range(L):
        kappa += (
            alpha_k**2 * sigma_w2 * tau *
            np.trace(U_list[l] @ C_h_ref[l] @ V_list[l])
        )

    # ── (E) Quantization term
    for l in range(L):
        kappa += (
            alpha_k**2 * tau *
            np.trace(U_q_list[l] @ C_h_ref[l] @ V_q_list[l] @ Theta_list[l])
        )
    alpha_k = float(alpha_k)
    tau = float(tau)
    return np.real(kappa * (tau ** 2))

def compute_kappa_Th1(k, alpha_list, sigma_w2, T, d,
                      D_Th_list, C_h_total_all):
    """
    kappa^(1)_{Th,k} = alpha_k^2 * sigma_w^2 * T^3 * d
                       * sum_l tr( D^Th_{lk}  C_{h,lk} )
    """
    L = len(D_Th_list)

    inner = 0.0
    for l in range(L):
        D_Th_lk = D_Th_list[l][k]      # precomputed, no G_lk needed
        C_lk    = C_h_total_all[l][k]
        inner  += np.real(np.trace(D_Th_lk @ C_lk))

    T = float(T)
    return alpha_list[k]**2 * sigma_w2 * T**3 * d * inner

def compute_kappa_ADC1(k, alpha_list, T, d,
                       D_ADC_list,        # [l][k]  precomputed
                       C_h_total_all):    # [l][k]
    """
    kappa^(1)_{ADC,k} = alpha_k^2 * T^3 * d * sum_l tr( D^ADC_{lk}  C_{h,lk} )

    Inputs:
        k             : desired user index
        alpha_list    : [K]
        T             : scalar pilot length
        d             : scalar number of streams
        D_ADC_list    : [l][k]  precomputed D^ADC_{lk}
        C_h_total_all : [l][k]  C_{h,lk}

    Returns: scalar
    """
    L = len(D_ADC_list)

    inner = 0.0
    for l in range(L):
        D_ADC_lk = D_ADC_list[l][k]
        C_lk     = C_h_total_all[l][k]
        inner   += np.real(np.trace(D_ADC_lk @ C_lk))
    
    T = float(T)
    return alpha_list[k]**2 * T**3 * d * inner

def compute_kappa_DAC1(k, K, alpha_list, T, d,
                       U_list, V_list,
                       Sigma_list,       # [l][k]  Sigma^dir
                       C_h_ref_all,      # [l][k]  C^ref
                       R_list,           # [l][k]
                       h_bar_list,       # [l][k]  shape (N,1)
                       sigma_d2):   # [k]
    """
    kappa^(1)_{DAC,k} — full bracket, coefficient of P_tilde_k * P_bar_k * T^2

    dP_DAC_k / d(P_tilde_k) = P_bar_k * T^2 * kappa_DAC1
    i.e. P_DAC_k = P_tilde_k * P_bar_k * T^2 * kappa_DAC1

    All lists indexed [l][k].
    """
    L   = len(U_list)
    total = 0.0

    # ── Group 1: i == k self terms ────────────────────────────────
    for l in range(L):
        U_lk     = U_list[l][k]
        V_lk     = V_list[l][k]
        Sig_lk   = Sigma_list[l][k]       # Sigma^dir_{lk}
        Cref_lk  = C_h_ref_all[l][k]      # C^ref_{h,lk}
        R_lk     = R_list[l][k]
        h        = h_bar_list[l][k]        # (N,1)
        hH       = h.conj().T              # (1,N)

        t1  = np.real(np.trace(Sig_lk @ U_lk @ Sig_lk @ V_lk))
        t2  = np.real(np.trace(R_lk @ U_lk)) * np.real(np.trace(R_lk @ V_lk))
        t3  = np.real(np.trace(R_lk @ U_lk)) * np.real((hH @ V_lk @ h)[0, 0])
        t4  = np.real((hH @ U_lk @ h)[0, 0]) * np.real(np.trace(R_lk @ V_lk))
        t5  = np.real(np.trace(Sig_lk @ U_lk)) * np.real(np.trace(Cref_lk @ V_lk))
        t6  = np.real(np.trace(Cref_lk @ U_lk)) * np.real(np.trace(Sig_lk @ V_lk))
        t7  = np.real(np.trace(Sig_lk @ V_lk.conj().T @ Cref_lk @ V_lk))
        t8  = np.real(np.trace(Sig_lk @ U_lk.conj().T @ Cref_lk @ U_lk))
        t9  = np.real(np.trace(Cref_lk @ U_lk @ Cref_lk @ V_lk))
        t10 = np.real(np.trace(Cref_lk @ U_lk)) * np.real(np.trace(Cref_lk @ V_lk))

        total += sigma_d2[k] * (t1+t2+t3+t4+t5+t6+t7+t8+t9+t10)

    # ── Group 2: i != k other users ───────────────────────────────
    for l in range(L):
        U_lk    = U_list[l][k]
        V_lk    = V_list[l][k]
        Sig_lk  = Sigma_list[l][k]
        Cref_lk = C_h_ref_all[l][k]

        for i in range(K):
            if i == k:
                continue

            Sig_li  = Sigma_list[l][i]
            Cref_li = C_h_ref_all[l][i]

            t1 = np.real(np.trace(Sig_lk  @ U_lk @ Sig_li  @ V_lk))
            t2 = np.real(np.trace(Cref_lk @ U_lk @ Sig_li  @ V_lk))
            t3 = np.real(np.trace(Sig_lk  @ U_lk @ Cref_li @ V_lk))
            t4 = np.real(np.trace(Cref_lk @ U_lk @ Cref_li @ V_lk))

            total += sigma_d2[i] * (t1+t2+t3+t4)

    # ── Group 3: cross-panel l != m ───────────────────────────────
    for l in range(L):
        U_lk    = U_list[l][k]
        Sig_lk  = Sigma_list[l][k]
        Cref_lk = C_h_ref_all[l][k]
        tr_U    = np.real(np.trace(U_lk @ (Sig_lk + Cref_lk)))

        for m in range(L):
            if m == l:
                continue
            V_mk    = V_list[m][k]
            Sig_mk  = Sigma_list[m][k]
            Cref_mk = C_h_ref_all[m][k]
            tr_V    = np.real(np.trace(V_mk @ (Sig_mk + Cref_mk)))

            total += sigma_d2[k] * tr_U * tr_V
    T = float(T)
    return alpha_list[k]**2 * (T**3) * d * total

def compute_g_lik(l, i, k, U_list, V_list, Sigma_dir, R, C_h_total_all,C_h_ref_all, h_bar_dir, T):
    U_lk = U_list[l][k]
    V_lk = V_list[l][k]

    Sigma_dir_li = Sigma_dir[l][i]   # Σ̄^dir_{li}
    R_li         = R[l][i]           # R_{li}
    C_h_li       = C_h_total_all[l][i]
    h_bar        = h_bar_dir[l][i]   # h̄^dir_{li}
    C_ref_li = C_h_ref_all[l][i] 
    # Term 1: Tr(Σ̄^dir_{li} U_{lk} Σ̄^dir_{li} V_{lk})
    t1 = np.trace(Sigma_dir_li @ U_lk @ Sigma_dir_li @ V_lk)

    # Term 2: Tr(R_{li} U_{lk}) Tr(R_{li} V_{lk})
    t2 = np.trace(R_li @ U_lk) * np.trace(R_li @ V_lk)

    # Term 3: Tr(R_{li} U_{lk}) h̄^{dir,H}_{li} V_{lk} h̄^dir_{li}
    #        + h̄^{dir,H}_{li} U_{lk} h̄^dir_{li} Tr(R_{li} V_{lk})
    t3 = (np.trace(R_li @ U_lk) * (h_bar.conj().T @ V_lk @ h_bar).item()
        + (h_bar.conj().T @ U_lk @ h_bar).item() * np.trace(R_li @ V_lk))

    # Term 4: Tr(Σ̄^dir_{li} U_{lk}) Tr(C^ref_{h,li} V_{lk})
    #        + Tr(C^ref_{h,li} U_{lk}) Tr(Σ̄^dir_{li} V_{lk})
    t4 = (np.trace(Sigma_dir_li @ U_lk) * np.trace(C_ref_li @ V_lk)
        + np.trace(C_ref_li @ U_lk) * np.trace(Sigma_dir_li @ V_lk))

    # Term 5: Tr(Σ̄^dir_{li} V^H_{lk} C^ref_{h,li} V_{lk})
    #        + Tr(Σ̄^dir_{li} U^H_{lk} C^ref_{h,li} U_{lk})
    t5 = (np.trace(Sigma_dir_li @ V_lk.conj().T @ C_ref_li @ V_lk)
        + np.trace(Sigma_dir_li @ U_lk.conj().T @ C_ref_li @ U_lk))

    # Term 6: Tr(C^ref_{h,li} U_{lk} C^ref_{h,li} V_{lk})
    t6 = np.trace(C_ref_li @ U_lk @ C_ref_li @ V_lk)

    # Term 7: Tr(C^ref_{h,li} U_{lk}) Tr(C^ref_{h,li} V_{lk})
    t7 = np.trace(C_ref_li @ U_lk) * np.trace(C_ref_li @ V_lk)

    g_lik = np.real(t1 + t2 + t3 + t4 + t5 + t6 + t7)
    return g_lik
    
def compute_kappa_DAC0(k, K, alpha_list, T, d,
                       U_list,          # 2-D [l][k]
                       V_list,          # 2-D [l][k]
                       C_h_total_all,   # 2-D [l][k]
                       A_list,          # 1-D [l]     A_l  shape (N,N)
                       Theta_list,      # 1-D [l]     Theta_l shape (N,N)
                       sigma_d2,        # 1-D [K]     sigma_d^2 per user
                       sigma_w2,        # scalar
                       g_lik):      # 2-D [i][k] scalar 
                                       
    L = len(U_list)
    kappa = 0.0
 
    for l in range(L):
        U_lk = U_list[l][k]
        V_lk = V_list[l][k]
    
        # double-user cross term  i ≠ u
        for i in range(K):
            C_h_li = C_h_total_all[l][i]
            for u in range(K):
                if u == i:
                    continue
                C_h_lu = C_h_total_all[l][u]
                val = np.real(np.trace(C_h_lu @ U_lk @ C_h_li @ V_lk))
                kappa += T * sigma_d2[i] * sigma_d2[u] * val
 
        # α_i^4 g_{ik} sub-term (optional)
        for i in range(K):
            kappa += T * sigma_d2[i]**2 * g_lik[l, i, k]

    for l in range(L):
        U_lk   = U_list[l][k]
        for m in range(L):
            if m == l:
                continue
            V_mk = V_list[m][k]
            for i in range(K):
                C_h_li = C_h_total_all[l][i]
                C_h_mi = C_h_total_all[m][i]
                tr_U = np.real(np.trace(U_lk @ C_h_li))
                tr_V = np.real(np.trace(V_mk @ C_h_mi))
                kappa += T * (sigma_d2[i]**2) * tr_U * tr_V
 
    for l in range(L):
        U_lk  = U_list[l][k]
        V_lk  = V_list[l][k]
        A_l   = A_list[l]
        AAlH  = A_l @ A_l.conj().T          # A_l A_l^H
        for i in range(K):
            C_h_li = C_h_total_all[l][i]
            val = np.real(np.trace(AAlH @ U_lk @ C_h_li @ V_lk))
            kappa += sigma_w2 * (T ** 2) * d * sigma_d2[i] * val

    for l in range(L):
        U_lk  = U_list[l][k]
        V_lk  = V_list[l][k]
        Th_l  = Theta_list[l]
        for i in range(K):
            C_h_li = C_h_total_all[l][i]
            val = np.real(np.trace(Th_l @ U_lk @ C_h_li @ V_lk))
            kappa += (T ** 2) * d * sigma_d2[i] * val
 
    return float(np.real(kappa))
