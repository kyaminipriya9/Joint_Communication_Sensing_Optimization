import numpy as np #type: ignore

def compute_kappa_M1_DP(c, k, alpha_list, Z_list, Sigma_list, C_h_ref_all, U_list, V_list, T):
    """
    kappa^(1)_{M,k,c} — coefficient of P_bar_k
    
    Conventions (matching your builder):
      Sigma_list[l][k]   = Sigma^dir_{lk}
      C_h_ref_all[l][k]  = C^ref_{h,lk}
      U_list[l][k]       = U_{lk}
      V_list[l][k]       = V_{lk}
      Z_list[k]          = Z_k   shape (N, d)
      alpha_list[k]      = alpha_k
    """
    L = len(U_list)
    d = Z_list[k].shape[1]

    # ||Z_c^H Z_k||_F^2
    ZcHZk  = Z_list[c].conj().T @ Z_list[k]
    norm_sq = np.linalg.norm(ZcHZk, 'fro') ** 2

    inner = 0.0
    for l in range(L):
        U_lk    = U_list[l][k]
        V_lk    = V_list[l][k]

        # (Sigma^dir_lk + C^ref_lk)
        A = Sigma_list[l][k] + C_h_ref_all[l][k]

        # (Sigma^dir_lc + C^ref_lc)
        B = Sigma_list[l][c] + C_h_ref_all[l][c]

        # Tr( A  U_lk  B  V_lk )
        inner += np.real(np.trace(A @ U_lk @ B @ V_lk))
        
    alpha_c = float(alpha_list[c])
    alpha_k = float(alpha_list[k])
    T = float(T)
    d = float(d)
    coeff = (alpha_list[c] ** 2 * alpha_list[k] ** 2 * T ** 2) / d
    return coeff * norm_sq * inner


def compute_kappa_M0_DP(c, k, K,
               alpha_list, Z_list,
               Sigma_list,       # [l][k]  Sigma^dir
               C_h_ref_all,      # [l][k]  C^ref
               C_h_total_all,    # [l][k]  C_h total = Sigma^dir + C^ref
               U_list, V_list,   # [l][k]
               R_list,           # [l][k]
               hbar_dir_list,    # [l][k]  shape (N,1)
               sigma_d2_list,    # [k]     sigma_d^2
               sigma_w2,         # scalar
               T,                # scalar
               U_q_list, V_q_list, Theta_list):      # [l]     Theta_l
    """
    kappa^(0)_{M,k,c} — all terms NOT multiplied by P_bar_k

    Conventions match your builder exactly:
      All 2-D lists indexed [l][k].
      hbar_dir_list[l][k] has shape (N, 1).
      Theta_list[l]       has shape (N, N).
      sigma_d2_list[i]    = sigma_d^2 for user i  (scalar per user).
    """
    L = len(U_list)
    d = Z_list[k].shape[1]

    # ||Z_c^H Z_k||_F^2
    ZcHZk   = Z_list[c].conj().T @ Z_list[k]
    norm_sq  = np.linalg.norm(ZcHZk, 'fro') ** 2

    total = 0.0

    # ── Block A ──────────────────────────────────────────────────
    # sum_{i != c} sigma_d2_i * T *
    #   Tr( (Sig_li + C_li) U_lk (Sig_lc + C_lc) V_lk )
    # Note: the book inner sum is i != j = c, so all i except c
    # (we do NOT additionally exclude k here — book does not)
    for l in range(L):
        U_lk  = U_list[l][k]
        V_lk  = V_list[l][k]
        B_lc  = Sigma_list[l][c] + C_h_ref_all[l][c]   # "j-side" matrix

        for i in range(K):
            if i == c:
                continue                                 # exclude i == c only

            A_li = Sigma_list[l][i] + C_h_ref_all[l][i]  # "i-side" matrix

            # Tr( (Sig_li + C_li)  U_lk  (Sig_lc + C_lc)  V_lk )
            val = np.real(np.trace(A_li @ U_lk @ B_lc @ V_lk))
            total += sigma_d2_list[i] * T * val

    # ── Block B ──────────────────────────────────────────────────
    # sigma_d2_c * T * [
    #   Tr(Sig_lc U_lk Sig_lc V_lk)
    # + Tr(R_lc U_lk) Tr(R_lc V_lk)
    # + Tr(R_lc U_lk) hbar^H V_lk hbar
    # + hbar^H U_lk hbar Tr(R_lc V_lk)
    # ]
    for l in range(L):
        U_lk    = U_list[l][k]
        V_lk    = V_list[l][k]
        Sig_lc  = Sigma_list[l][c]
        R_lc    = R_list[l][c]
        h       = hbar_dir_list[l][c]        # shape (N, 1)
        hH      = h.conj().T                 # shape (1, N)

        t1 = np.real(np.trace(Sig_lc @ U_lk @ Sig_lc @ V_lk))
        t2 = np.real(np.trace(R_lc @ U_lk)) * np.real(np.trace(R_lc @ V_lk))
        t3 = np.real(np.trace(R_lc @ U_lk)) * np.real((hH @ V_lk @ h)[0, 0])
        t4 = np.real((hH @ U_lk @ h)[0, 0]) * np.real(np.trace(R_lc @ V_lk))

        total += sigma_d2_list[c] * T * (t1 + t2 + t3 + t4)

    # ── Block C ──────────────────────────────────────────────────
    # sigma_w2 * T * Tr(V_lk U_lk C_h_lc)
    # + T * Tr(U_lk^(q) C_h_lc V_lk^(q) Theta_l)
    # ── Block C ──────────────────────────────────────────────────────
    for l in range(L):
        U_lk   = U_list[l][k]
        V_lk   = V_list[l][k]
        C_h_lc = C_h_total_all[l][c]

        # Term 1: noise — uses regular U_lk, V_lk, no Theta
        total += sigma_w2 * T * np.real(np.trace(V_lk @ U_lk @ C_h_lc))

        # Term 2: quantization — uses U_q, V_q, Theta, no sigma_w2
        U_q_lk = U_q_list[l][k]
        V_q_lk = V_q_list[l][k]
        Th_l   = Theta_list[l]
        total += T * np.real(np.trace(U_q_lk @ C_h_lc @ V_q_lk @ Th_l))
    # ── Block D ──────────────────────────────────────────────────
    # sigma_d2_c * T *
    #   sum_{l} sum_{m != l} Tr(U_lk C_h_lc) * Tr(V_mk C_h_mc)
    for l in range(L):
        U_lk   = U_list[l][k]
        C_h_lc = C_h_total_all[l][c]
        tr_U   = np.real(np.trace(U_lk @ C_h_lc))

        for m in range(L):
            if m == l:
                continue
            V_mk   = V_list[m][k]
            C_h_mc = C_h_total_all[m][c]
            tr_V   = np.real(np.trace(V_mk @ C_h_mc))

            total += sigma_d2_list[c] * T * tr_U * tr_V

    return (alpha_list[c] ** 2 / d) * norm_sq * total


# ── Wrapper: compute full K×K kappa matrices ──────────────────────────────────
def build_kappa_MUI(K, alpha_list, Z_list,
                    Sigma_list, C_h_ref_all, C_h_total_all,
                    U_list, V_list, R_list, hbar_dir_list,
                    sigma_d2_list, sigma_w2, T,
                    U_q_list, V_q_list, Theta_list):
    """
    Returns:
      kappa1_dp[k, c] = kappa^(1)_{M,k,c}   (0 when c == k)
      kappa0_dp[k, c] = kappa^(0)_{M,k,c}   (0 when c == k)

    Gradient of P_MUI_k w.r.t. P_tilde_c:
      dP_MUI_k / dP_tilde_c = P_bar_k * kappa1_dp[k,c] + kappa0_dp[k,c]
    """
    kappa1_dp = np.zeros((K, K))
    kappa0_dp = np.zeros((K, K))

    for k in range(K):
        for c in range(K):
            if c == k:
                continue      # P_tilde_k does not appear in P_MUI_k

            kappa1_dp[k, c] = compute_kappa_M1_DP(
                c, k,
                alpha_list, Z_list,
                Sigma_list, C_h_ref_all,
                U_list, V_list,
                T
            )

            kappa0_dp[k, c] = compute_kappa_M0_DP(
                c, k, K,
                alpha_list, Z_list,
                Sigma_list,
                C_h_ref_all,
                C_h_total_all,
                U_list, V_list,
                R_list,
                hbar_dir_list,
                sigma_d2_list,
                sigma_w2,
                T,
                U_q_list,
                V_q_list,
                Theta_list
            )

    return kappa1_dp, kappa0_dp