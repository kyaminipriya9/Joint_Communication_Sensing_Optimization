import numpy as np

def compute_Uq_lk(A_l,G_lk,W_lk):
    return G_lk.conj().T @ W_lk.conj().T @ A_l

def compute_Vq_lk(A_l, G_lk, W_lk):
    return A_l.conj().T @ W_lk @ G_lk
# U_lk = A_l^H G_lk^H W_lk^H A_l
def compute_U_lk(A_l, G_lk, W_lk):
    return A_l.conj().T @ G_lk.conj().T @ W_lk.conj().T @ A_l
# V_lk = A_l^H W_lk G_lk A_l
def compute_V_lk(A_l, G_lk, W_lk):
    return A_l.conj().T @ W_lk @ G_lk @ A_l


# Sigmabar_lk^dir = R_lk + hbar_dir_lk hbar_dir_lk^H
def compute_Sigmabar_dir(R_lk, hbar_dir_lk):
    return R_lk + hbar_dir_lk @ hbar_dir_lk.conj().T


# C_h_ref_lk = E[h_ref_lk h_ref_lk^H]  (sample average)
# Input: h_ref_samples shape (num_samples, N, 1)
# def compute_C_h_ref(h_ref_samples):
#     num_samples = h_ref_samples.shape[0]
#     N = h_ref_samples.shape[1]
#     C = np.zeros((N, N), dtype=complex)
#     for h in h_ref_samples:
#         C += h @ h.conj().T
#     return C / num_samples

def build_C_h_ref_all_from_targets(
    b_target_all,
    alpha_var_matrix_all,
    K,
    M,
    N,
    make_hermitian=True
):
    """
    Builds C_h_ref_all[l][k] = E[h_ref_lk h_ref_lk^H].

    Parameters
    ----------
    b_target_all : ndarray, shape (N, M, Tg)
        Target steering vectors b_target[:, l, t].

    alpha_var_matrix_all : ndarray, shape (K, M, Tg)
        Reflection coefficient variances alpha_var_matrix[k, l, t].

    K : int
        Number of users.

    M : int
        Number of APs / panels.

    N : int
        Number of antennas per AP.

    Returns
    -------
    C_h_ref_all : list of list
        C_h_ref_all[l][k] has shape (N, N).
    """

    b_target_all = np.asarray(b_target_all)
    alpha_var_matrix_all = np.asarray(alpha_var_matrix_all)

    if b_target_all.shape[0] != N:
        raise ValueError(f"Expected b_target_all.shape[0] = {N}, got {b_target_all.shape[0]}")

    if b_target_all.shape[1] != M:
        raise ValueError(f"Expected b_target_all.shape[1] = {M}, got {b_target_all.shape[1]}")

    if alpha_var_matrix_all.shape[0] != K:
        raise ValueError(f"Expected alpha_var_matrix_all.shape[0] = {K}, got {alpha_var_matrix_all.shape[0]}")

    if alpha_var_matrix_all.shape[1] != M:
        raise ValueError(f"Expected alpha_var_matrix_all.shape[1] = {M}, got {alpha_var_matrix_all.shape[1]}")

    Tg = b_target_all.shape[2]

    if alpha_var_matrix_all.shape[2] != Tg:
        raise ValueError(
            f"Tg mismatch: b_target_all has {Tg}, "
            f"alpha_var_matrix_all has {alpha_var_matrix_all.shape[2]}"
        )

    C_h_ref_all = [
        [np.zeros((N, N), dtype=complex) for _ in range(K)]
        for _ in range(M)
    ]

    for l in range(M):
        # B_l shape: N x Tg
        B_l = b_target_all[:, l, :]

        for k in range(K):
            # weights shape: Tg
            weights = np.real(alpha_var_matrix_all[k, l, :])

            # C_ref = sum_t weights[t] * b_t b_t^H
            C_ref = (B_l * weights.reshape(1, Tg)) @ B_l.conj().T

            if make_hermitian:
                C_ref = 0.5 * (C_ref + C_ref.conj().T)

            C_h_ref_all[l][k] = C_ref

    return C_h_ref_all

# C_h_lk = Sigmabar_dir_lk + C_h_ref_lk
def compute_C_h_total(Sigmabar_dir_lk, C_h_ref_lk):
    return Sigmabar_dir_lk + C_h_ref_lk


# sigma_d^2_i = alpha_i (1 - alpha_i) P_tot_i
def compute_sigma_d2(alpha_i, P_tot_i):
    return alpha_i * (1 - alpha_i) * P_tot_i


# S_l = sum_i [ alpha_i * P_tot_i * C_h_li ] + sigma_w2 * I
# Per panel l only (summed over ALL users i — no k index)
# C_h_total_l : list of length K → C_h_total_l[i] = C_{h,li}
# def compute_S_l(alpha_list, P_tot_list, C_h_total_l, sigma_w2):
#     N = C_h_total_l[0].shape[0]
#     S = np.zeros((N, N), dtype=complex)
#     for i in range(len(alpha_list)):
#         S += alpha_list[i] * P_tot_list[i] * C_h_total_l[i]
#     S += sigma_w2 * np.eye(N)
#     return S


# # Theta_l = A_l (I_N - A_l) S_l
# # Per panel l only (no k index)
# def compute_theta_l(A_l, S_l):
#     N = A_l.shape[0]
#     I = np.eye(N, dtype=A_l.dtype)
#     return A_l @ (I - A_l) @ S_l
def compute_S_l(alpha_list, P_tot_list, C_h_total_l, sigma_w2):
    N = C_h_total_l[0].shape[0]
    S_full = np.zeros((N, N), dtype=complex)

    for i in range(len(alpha_list)):
        S_full += alpha_list[i] * P_tot_list[i] * C_h_total_l[i]

    S_full += sigma_w2 * np.eye(N)

    S_diag = np.diag(np.diag(S_full))

    return S_diag


def compute_theta_l(A_l, S_l):
    N = A_l.shape[0]
    I = np.eye(N, dtype=complex)

    Theta_l = A_l @ (I - A_l) @ S_l

    # Match MATLAB symmetrization
    Theta_l = 0.5 * (Theta_l + Theta_l.conj().T)

    return Theta_l
def compute_D_Th_lk(A_l, G_lk, W_lk):
    Rl = A_l.conj().T @ A_l
    return (
        A_l.conj().T
        @ G_lk.conj().T
        @ W_lk.conj().T
        @ Rl
        @ W_lk
        @ G_lk
        @ A_l
    )


def compute_D_ADC_lk(A_l, G_lk, W_lk, Theta_l):
    return (
        A_l.conj().T
        @ G_lk.conj().T
        @ W_lk.conj().T
        @ Theta_l
        @ W_lk
        @ G_lk
        @ A_l
    )
# def compute_D_Th_lk(A_l, G_lk, R_lk):
#     """
#     D^Th_{lk} = A_l^H G_lk^H R_lk G_lk A_l

#     Inputs:
#         A_l   : (N, N)  analog combiner for panel l
#         G_lk  : (N, N)  effective channel matrix for user k, panel l
#         R_lk  : (N, N)  spatial covariance R_{lk}

#     Returns: (N, N)
#     """
#     return A_l.conj().T @ G_lk.conj().T @ R_lk @ G_lk @ A_l
# def compute_D_Th_lk(A_l, G_lk):
#     Rl = A_l @ A_l.conj().T
#     return A_l.conj().T @ G_lk.conj().T @ Rl @ G_lk @ A_l

# def compute_D_ADC_lk(A_l, G_lk, Theta_l):
#     """
#     D^ADC_{lk} = A_l^H  G_lk^H  Theta_l  G_lk  A_l

#     Compare with D^Th_{lk} = A_l^H G_lk^H R_lk G_lk A_l
#     — only difference is Theta_l (per panel) replaces R_lk (per user)

#     Inputs:
#         A_l     : (N, N)  analog combiner for panel l
#         G_lk    : (N, N)  effective channel matrix, user k panel l
#         Theta_l : (N, N)  ADC distortion matrix = A_l (I - A_l) S_l

#     Returns: (N, N)
#     """
#     return A_l.conj().T @ G_lk.conj().T @ Theta_l @ G_lk @ A_l


# Build U[l][k], V[l][k], Sigmabar_dir[l][k] for all l, k
#
# Inputs  — all 2-D [l][k]:
#   A_list     : [A_l]           length L         (per panel only)
#   G_list     : [l][k] G_lk     shape (N, N)
#   W_list     : [l][k] W_lk     shape (N, N)
#   R_list     : [l][k] R_lk     shape (N, N)
#   hbar_dir_list : [l][k] h̄_lk    shape (N, 1)
#
# Returns — all 2-D [l][k]:
#   U_list, V_list, Sigma_list

def build_all_matrices(A_list, G_list, W_list, R_list, hbar_dir_list,Theta_list):
    L = len(A_list)
    K = len(G_list[0])

    U_list     = [[] for _ in range(L)]
    U_q_list     = [[] for _ in range(L)]
    V_list     = [[] for _ in range(L)]
    V_q_list     = [[] for _ in range(L)]
    Sigma_list = [[] for _ in range(L)]
    D_Th_list  = [[] for _ in range(L)]
    D_ADC_list = [[] for _ in range(L)]
    for l in range(L):
        for k in range(K):
            U_list[l].append(compute_U_lk(A_list[l], G_list[l][k], W_list[l][k]))
            V_list[l].append(compute_V_lk(A_list[l], G_list[l][k], W_list[l][k]))
            U_q_list[l].append(compute_Uq_lk(A_list[l], G_list[l][k], W_list[l][k]))
            V_q_list[l].append(compute_Vq_lk(A_list[l], G_list[l][k], W_list[l][k]))
            Sigma_list[l].append(compute_Sigmabar_dir(R_list[l][k], hbar_dir_list[l][k]))
            D_Th_list[l].append(compute_D_Th_lk(A_list[l], G_list[l][k], W_list[l][k]))
            D_ADC_list[l].append(compute_D_ADC_lk(A_list[l], G_list[l][k], W_list[l][k], Theta_list[l]))
    return U_list, V_list,U_q_list,V_q_list, Sigma_list, D_Th_list, D_ADC_list


# Build Theta[l] for all panels l
# Requires C_h_total[l] = list over users i → C_h_total[l][i]
def build_theta_list(A_list, alpha_list, P_tot_list, C_h_total, sigma_w2):
    Theta_list = []
    for l in range(len(A_list)):
        # C_h_total[l] is a list over i, so compute_S_l gets C_{h,li} for all i
        S_l     = compute_S_l(alpha_list, P_tot_list, C_h_total[l], sigma_w2)
        Theta_l = compute_theta_l(A_list[l], S_l)
        Theta_list.append(Theta_l)
    return Theta_list


import numpy as np

def compute_R_tilda_li(R_li, h_bar_li):
    """
    R̃_li = R_li + h̄_li h̄_li^H

    This is used throughout as a shorthand for the total
    mean-inclusive spatial covariance (Eq. 23 notation).

    Inputs:
        R_li     : (N, N)  spatial covariance matrix for user i, panel l
        h_bar_li : (N, 1)  mean channel vector for user i, panel l

    Returns: (N, N)
    """
    return R_li + h_bar_li @ h_bar_li.conj().T


def design_quantizer_gains(b_dac, b_adc_options, adc_mode, UE, AP, N):
    """
    Python version of MATLAB design_quantizer_gains.

    b_dac:
        int in {1,...,6} or "ideal"

    b_adc_options:
        int for adc_mode="low"
        list/array for adc_mode="dynamic"

    adc_mode:
        "ideal", "low", or "dynamic"

    UE:
        number of users K

    AP:
        number of APs L

    N:
        antennas per AP

    Returns
    -------
    alpha_dac : shape (K,)
        DAC gain per UE. This is your alpha_list.

    A_list : list length L
        A_list[l] is N x N ADC gain matrix for AP l.

    alpha_adc_per_ap : shape (L,)
        Scalar ADC gain at each AP, useful for debugging.
    """

    rho_lookup = {
        1: 0.3634,
        2: 0.1175,
        3: 0.0345,
        4: 0.0095,
        5: 0.0025,
        6: 0.0007,
    }

    # DAC gain, user side
    if isinstance(b_dac, str) and b_dac.lower() == "ideal":
        rho_dac_val = 0.0
    else:
        b_dac = int(b_dac)
        if b_dac not in rho_lookup:
            raise ValueError("b_dac must be one of {1,2,3,4,5,6} or 'ideal'")
        rho_dac_val = rho_lookup[b_dac]

    alpha_dac_scalar = 1.0 - rho_dac_val
    alpha_dac = alpha_dac_scalar * np.ones(UE, dtype=float)

    # ADC gain, AP side
    adc_mode = adc_mode.lower()
    alpha_adc_per_ap = np.zeros(AP, dtype=float)

    if adc_mode == "ideal":
        alpha_adc_per_ap[:] = 1.0

    elif adc_mode == "low":
        if not np.isscalar(b_adc_options):
            raise ValueError('For adc_mode="low", b_adc_options must be a scalar.')

        b_adc = int(b_adc_options)
        if b_adc not in rho_lookup:
            raise ValueError("b_adc_options must be one of {1,2,3,4,5,6}")

        alpha_adc_per_ap[:] = 1.0 - rho_lookup[b_adc]

    elif adc_mode == "dynamic":
        b_adc_options = np.asarray(b_adc_options, dtype=int)
        num_options = len(b_adc_options)

        if AP % num_options != 0:
            raise ValueError(
                'For adc_mode="dynamic", AP must be divisible by len(b_adc_options).'
            )

        aps_per_option = AP // num_options

        for idx, b_adc in enumerate(b_adc_options):
            if int(b_adc) not in rho_lookup:
                raise ValueError("All ADC bit options must be in {1,2,3,4,5,6}")

            alpha_val = 1.0 - rho_lookup[int(b_adc)]
            start = idx * aps_per_option
            end = (idx + 1) * aps_per_option
            alpha_adc_per_ap[start:end] = alpha_val

    else:
        raise ValueError('adc_mode must be "ideal", "low", or "dynamic".')

    A_list = [
        alpha_adc_per_ap[l] * np.eye(N, dtype=complex)
        for l in range(AP)
    ]

    return alpha_dac, A_list, alpha_adc_per_ap
import numpy as np

# def right_solve(B, C):
#     """
#     Return B @ inv(C) without explicitly forming inv(C).
#     """
#     return np.linalg.solve(C, B.conj().T).conj().T


# def compute_G_lk(l, k,
#                  A_list,
#                  R_tilda_all,
#                  C_yy_all):
#     """
#     Power-independent part of the LMMSE estimator:
#         G_lk = R_tilda_lk A_l^H C_yy_lk^{-1}

#     This convention is consistent with kappas that already contain
#     alpha_k, P_bar[k], and tau factors explicitly.
#     """
#     A_l = A_list[l]
#     R_tilda_lk = R_tilda_all[l][k]
#     C_yy_lk = C_yy_all[l][k]

#     B = R_tilda_lk @ A_l.conj().T

#     return right_solve(B, C_yy_lk)


# def build_G_list(A_list, R_tilda_all, C_yy_all):
#     L = len(A_list)
#     K = len(R_tilda_all[0])

#     G_list = [[] for _ in range(L)]

#     for l in range(L):
#         for k in range(K):
#             G_list[l].append(
#                 compute_G_lk(
#                     l, k,
#                     A_list,
#                     R_tilda_all,
#                     C_yy_all
#                 )
#             )

#     return G_list

def right_solve(B, C):
    """
    Return B @ inv(C) without explicitly forming inv(C).
    Works also when C is not exactly Hermitian.
    """
    return np.linalg.solve(C.T, B.T).T


def compute_G_lk(l, k,
                      alpha_list,
                      P_bar_ref,
                      tau,
                      A_list,
                      R_tilda_all,
                      C_yy_all):
    """
    Full LMMSE matrix:

        G_lk = C_hy C_yy^{-1}

    where

        C_hy = alpha_k * sqrt(P_bar_k) * tau
               * R_tilda_lk * A_l^H
    """
    A_l = A_list[l]
    R_tilda_lk = R_tilda_all[l][k]
    C_yy_lk = C_yy_all[l][k]

    C_hy_lk = (
        alpha_list[k]
        * np.sqrt(P_bar_ref[k])
        * tau
        * R_tilda_lk
        @ A_l.conj().T
    )

    return right_solve(C_hy_lk, C_yy_lk)


def build_G_list(alpha_list,
                      P_bar_ref,
                      tau,
                      A_list,
                      R_tilda_all,
                      C_yy_all):
    L = len(A_list)
    K = len(alpha_list)

    G_list = [[] for _ in range(L)]

    for l in range(L):
        for k in range(K):
            G_list[l].append(
                compute_G_lk(
                    l, k,
                    alpha_list,
                    P_bar_ref,
                    tau,
                    A_list,
                    R_tilda_all,
                    C_yy_all
                )
            )

    return G_list

def compute_C_yy(l, k,
                 alpha_list,
                 P_bar_ref,
                 tau,
                 A_list,
                 R_tilda_all,
                 C_h_total_all,
                 sigma_d2,
                 sigma_w2,
                 Theta_list):
    """
    C_yy_lk =
        alpha_k^2 * Pbar_k * tau^2 * A_l R_tilda_lk A_l^H
        + tau * sum_i sigma_d2_i * A_l C_h_li A_l^H
        + tau * sigma_w2 * A_l A_l^H
        + tau * Theta_l
    """
    A_l = A_list[l]
    N = A_l.shape[0]
    K = len(alpha_list)

    term1 = (
        alpha_list[k] ** 2
        * P_bar_ref[k]
        * tau ** 2
        * (A_l @ C_h_total_all[l][k] @ A_l.conj().T)
    )

    term2 = np.zeros((N, N), dtype=complex)
    for i in range(K):
        term2 += sigma_d2[i] * (
            A_l @ C_h_total_all[l][i] @ A_l.conj().T
        )
    term2 *= tau

    term3 = tau * sigma_w2 * (A_l @ A_l.conj().T)

    term4 = tau * Theta_list[l]

    return term1 + term2 + term3 + term4


def build_C_yy_all(alpha_list,
                   P_bar_ref,
                   tau,
                   A_list,
                   R_tilda_all,
                   C_h_total_all,
                   sigma_d2,
                   sigma_w2,
                   Theta_list):
    L = len(A_list)
    K = len(alpha_list)

    C_yy_all = [[] for _ in range(L)]

    for l in range(L):
        for k in range(K):
            C_yy_all[l].append(
                compute_C_yy(
                    l, k,
                    alpha_list,
                    P_bar_ref,
                    tau,
                    A_list,
                    R_tilda_all,
                    C_h_total_all,
                    sigma_d2,
                    sigma_w2,
                    Theta_list
                )
            )

    return C_yy_all

def compute_C_hh_dir(l, k,
                     alpha_list,
                     P_bar_ref,
                     tau,
                     A_list,
                     R_tilda_all,
                     C_yy_all):
    """
    C_hh_dir_lk =
        alpha_k^2 * Pbar_k * tau^2
        * R_tilda_lk A_l^H C_yy_lk^{-1} A_l R_tilda_lk
    """
    A_l = A_list[l]
    R_tilda_lk = R_tilda_all[l][k]
    C_yy_lk = C_yy_all[l][k]

    X = np.linalg.solve(C_yy_lk, A_l @ R_tilda_lk)

    return (
        alpha_list[k] ** 2
        * P_bar_ref[k]
        * tau ** 2
        * (R_tilda_lk @ A_l.conj().T @ X)
    )


def build_C_hh_dir_all(alpha_list,
                       P_bar_ref,
                       tau,
                       A_list,
                       R_tilda_all,
                       C_yy_all):
    L = len(A_list)
    K = len(alpha_list)

    C_hh_dir_all = [[] for _ in range(L)]

    for l in range(L):
        for k in range(K):
            C_hh_dir_all[l].append(
                compute_C_hh_dir(
                    l, k,
                    alpha_list,
                    P_bar_ref,
                    tau,
                    A_list,
                    R_tilda_all,
                    C_yy_all
                )
            )

    return C_hh_dir_all
def build_R_tilda_all(R_list, h_bar_list):
    """
    Build R̃_li for all panels l and users i.

    Inputs:
        R_list     : [L][K]  R_li   shape (N, N)
        h_bar_list : [L][K]  h̄_li   shape (N, 1)

    Returns: [L][K]  R̃_li
    """
    L = len(R_list)
    K = len(R_list[0])
    R_tilda_all = [[] for _ in range(L)]
    for l in range(L):
        for i in range(K):
            R_tilda_all[l].append(compute_R_tilda_li(R_list[l][i], h_bar_list[l][i]))
    return R_tilda_all
