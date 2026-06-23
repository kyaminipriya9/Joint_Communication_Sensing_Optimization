"""
Run ONLY the communication D/lambda-QT code for fixed-power fraction sweeps.

Cases implemented
-----------------
Case 1: keep P_bar fixed and optimize only P_tilda.
Case 2: keep P_tilda fixed and optimize only P_bar.

This file does NOT run classical QT and does NOT run deep unfolding.
It uses your existing main code only for dataset loading, kappa building,
SINR/WSR evaluation, and auxiliary-variable update. It uses
Communication_Accelerated_QT.py only for the D/lambda single-block updates.

How to use
----------
1) Put this file in the same folder as your existing code files.
2) Change MAIN_MODULE below to the filename of your main script WITHOUT .py.
   Example: if your main file is Joint_Accelerated_QT.py, use
       MAIN_MODULE = "Joint_Accelerated_QT"
3) Ensure Communication_Accelerated_QT.py is also in the same folder.
4) Change MAT_FILE if needed.
5) Run:
       python run_comm_lambda_fixed_cases.py

Outputs
-------
For each Pmax/setup/fraction-pair/case:
    - convergence plot PNG
    - step-size text file with per-user alpha/lambda values
    - step-size NPZ file with full histories
For the full sweep:
    - final WSR comparison plot PNG
    - summary CSV
"""

# Keep postponed evaluation of type hints so annotations do not affect runtime imports.
from __future__ import annotations

# Standard-library imports for dynamic module loading, file paths, and output folders.
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Numerical and plotting libraries used for power arrays, histories, CSV data, and figures.
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Main file that contains load_matlab_direct_values, build_all_kappas,
# compute_SINR_exact, compute_WSR_exact, update_auxiliary, etc.
# Use either module name without .py or a direct path ending with .py.
# Name/path of the main script used as a helper library. It is imported, not executed as __main__.
MAIN_MODULE = "communication_classical_unfolding"       

# Lambda/D module. This is the file you currently import as Communication_Accelerated_QT.
# Name/path of the accelerated QT implementation that contains the D/lambda block updates.
LAMBDA_MODULE = "Communication_Accelerated_QT"

# MAT_FILE = "12th_may_ideal_dataset_with_alpha_var.mat"
# Dataset used to build communication channel statistics and kappa coefficients.
MAT_FILE = "JointOpt_Dataset_with_V.mat"
# Root folder where CSV summaries, plots, and step-size text files are written.
OUTPUT_DIR = "lambda_fraction_sweep_outputs"

# Problem dimensions selected from the MATLAB dataset.
K_USE = 8
M_USE = 16
N_USE = 16
# Sweep controls: setup indices and transmit-power values to test.
SETUP_LIST = [0]
PMAX_DB_LIST = list(np.arange(-20, 31, 5))

# Numerical/algorithmic settings for the lambda-QT iterations.
SIGMA_W2 = 1.0
MAX_ITERS = 500
EPSILON = 1e-4
EPS_POWER = 1e-12
EPS_LAMBDA = 1e-30
LAMBDA_MODE_BAR = "actual"
USE_BACKTRACKING = True
VERBOSE = True

# Fraction sweep.
# Default: paired complementary fractions:
#   (Pbar, Ptilda) = (0.1, 0.9), (0.2, 0.8), ..., (0.9, 0.1)
# These satisfy Pbar_fraction + Ptilda_fraction = 1.
# PBAR_FRACTION_LIST = np.asarray([0.5], dtype=float)
# PTILDA_FRACTION_LIST = np.asarray([0.5], dtype=float)

# Fixed initial/frozen fractions for pilot power and data power.
# With RUN_CARTESIAN_PRODUCT=False, entries are paired by index.
PBAR_FRACTION_LIST = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)
PTILDA_FRACTION_LIST = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=float)

# Keep this False for the 9 paired cases above.
# Set True only if you want a full feasible grid from the two lists.
RUN_CARTESIAN_PRODUCT = False

# Build the list of fraction pairs once at import time.
# Each pair defines one fixed-fraction experiment.
if RUN_CARTESIAN_PRODUCT:
    FRACTION_PAIRS = [
        (float(pb), float(pt))
        for pb in PBAR_FRACTION_LIST
        for pt in PTILDA_FRACTION_LIST
        if pb + pt <= 1.0 + 1e-12
    ]
else:
    if len(PBAR_FRACTION_LIST) != len(PTILDA_FRACTION_LIST):
        raise ValueError("PBAR_FRACTION_LIST and PTILDA_FRACTION_LIST must have the same length for paired sweep.")
    FRACTION_PAIRS = list(zip(PBAR_FRACTION_LIST.astype(float), PTILDA_FRACTION_LIST.astype(float)))

# Hardware/estimation reference power used when building Theta, sigma_d2, Cyy, G.
# True matches your current main script's communication-lambda path:
#   P_comm_ref_hw = 0.5 * Pmax, P_sense_ref_hw = 0.5 * Pmax.
# False uses pure communication reference:
#   P_comm_ref_hw = Pmax, P_sense_ref_hw = 0.
# Controls only the hardware/reference power used to build fixed matrices.
# It does not change the actual fixed fractions swept below.
MATCH_MAIN_REFERENCE_POWER = False


# =============================================================================
# IMPORT HELPERS
# =============================================================================


def import_module_from_name_or_path(module_name_or_path: str):
    """Import a module from a normal module name or from a .py file path."""
    # Treat the input either as a Python module name or as a filesystem path.
    module_path = Path(module_name_or_path)

    # Path case: load the file directly without requiring it to be on PYTHONPATH.
    if module_name_or_path.endswith(".py") or module_path.exists():
        module_path = module_path.resolve()
        module_name = module_path.stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import module from path: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Module-name case: import normally from the current Python environment.
    return importlib.import_module(module_name_or_path)


# Import the user-selected modules after the configuration is defined.
main = import_module_from_name_or_path(MAIN_MODULE)
lam = import_module_from_name_or_path(LAMBDA_MODULE)


# =============================================================================
# COMMON PROBLEM BUILDING
# =============================================================================


def db2lin(db: float) -> float:
    return float(10.0 ** (float(db) / 10.0))




def build_communication_problem(
    *,
    mat_file: str,
    pmax_db: float,
    setup_idx: int,
    K_use: int = K_USE,
    M_use: int = M_USE,
    N_use: int = N_USE,
    sigma_w2: float = SIGMA_W2,
) -> Dict[str, Any]:
    """
    Build only the communication quantities required by D/lambda-QT.

    This is a trimmed version of your main setup path. It does not build sensing
    coefficients and does not call classical/DU runners.
    """
    # Load one setup from the MATLAB dataset using the existing main-file loader.
    data = main.load_matlab_direct_values(
        mat_file,
        setup_idx=setup_idx,
        K_use=K_use,
        M_use=M_use,
        N_use=N_use,
    )

    # Extract dimensions and fixed communication objects from the loaded setup.
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

    # Convert this sweep power from dB to linear scale.
    pmax_lin = db2lin(pmax_db)
    P_total_max = pmax_lin

    # Choose the reference power split used only for hardware/estimation matrices.
    if MATCH_MAIN_REFERENCE_POWER:
        # Matches your current comm_lambda/comm_only_compare path.
        P_comm_ref_hw = 0.5 * pmax_lin
        P_sense_ref_hw = 0.5 * pmax_lin
    else:
        # Pure communication-only reference, if you want zero sensing reference.
        P_comm_ref_hw = pmax_lin
        P_sense_ref_hw = 0.0

    # Reference powers used to build Theta, sigma_d2, C_yy, and G.
    P_bar_ref = 0.5 * P_comm_ref_hw * np.ones(K, dtype=float)
    P_tilda_ref = 0.5 * P_comm_ref_hw * np.ones(K, dtype=float)
    P_prime_ref = P_sense_ref_hw * np.ones(K, dtype=float)
    P_tot_list = P_bar_ref + P_tilda_ref + P_prime_ref

    # Build ADC quantization covariance matrices using the reference total power.
    Theta_list = main.build_theta_list_for_power(
        A_list=A_list,
        C_h_total_all=C_h_total_all,
        alpha_list=alpha_list,
        P_tot_list=P_tot_list,
        sigma_w2=sigma_w2,
    )

    # DAC distortion variance depends on quantization factor and reference power.
    sigma_d2 = alpha_list * (1.0 - alpha_list) * P_tot_list
    # Use the same MR combining structure as the main code.
    W_list = main.build_W_list_MR(M, K, N)

    # Build channel second-moment matrices and LMMSE-related matrices.
    R_tilda_all = main.build_R_tilda_all(R_list, h_bar_list)

    C_yy_all = main.build_C_yy_all(
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

    G_list = main.build_G_list(
        alpha_list=alpha_list,
        P_bar_ref=P_bar_ref,
        tau=tau,
        A_list=A_list,
        R_tilda_all=R_tilda_all,
        C_yy_all=C_yy_all,
    )

    # Build matrices needed by all kappa expressions.
    U_list, V_list, U_q_list, V_q_list, _, D_Th_list, D_ADC_list = main.build_all_matrices(
        A_list,
        G_list,
        W_list,
        R_list,
        h_bar_list,
        Theta_list,
    )

    # Precompute g_lik for every AP/user/user triple used in DAC-related terms.
    g_lik = np.zeros((M, K, K), dtype=float)
    for l_idx in range(M):
        for i in range(K):
            for k in range(K):
                g_lik[l_idx, i, k] = main.compute_g_lik(
                    l_idx,
                    i,
                    k,
                    U_list,
                    V_list,
                    Sigma_list,
                    R_list,
                    C_h_total_all,
                    C_h_ref_all,
                    h_bar_list,
                    T=tau,
                )

    # Build all scalar kappa coefficients used by the communication SINR.
    (
        kappa_S,
        kappa_V1,
        kappa_V0,
        kappa_K1,
        kappa_K0,
        kappa_DAC1,
        kappa_DAC0,
        kappa_Th1,
        kappa_ADC1,
        kappa_M1,
        kappa_M0,
    ) = main.build_all_kappas(
        K,
        alpha_list,
        tau,
        d,
        U_list,
        V_list,
        U_q_list,
        V_q_list,
        Sigma_list,
        R_list,
        h_bar_list,
        C_h_ref_all,
        C_h_total_all,
        sigma_d2,
        sigma_w2,
        D_Th_list,
        D_ADC_list,
        A_list,
        Theta_list,
        Z_list,
        g_lik,
    )

    # Build pilot-power-independent thermal and ADC terms.
    kappa_Th0, kappa_ADC0 = main.build_kappa_Th0_ADC0(
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

    # Return a compact problem dictionary used by all subsequent runners.
    return {
        "K": K,
        "M": M,
        "N": N,
        "tau": tau,
        "d": d,
        "w": np.ones(K, dtype=float),
        "P_total_max": P_total_max,
        "Pmax_lin": pmax_lin,
        "Pmax_db": float(pmax_db),
        "setup_idx": int(setup_idx),
        "Z_list": Z_list,
        "kappa_S": kappa_S,
        "kappa_V1": kappa_V1,
        "kappa_K1": kappa_K1,
        "kappa_V0": kappa_V0,
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


def evaluate(problem: Dict[str, Any], P_bar: np.ndarray, P_tilda: np.ndarray) -> Dict[str, Any]:
    """Evaluate communication WSR/SINR for a given P_bar, P_tilda."""
    # Delegate objective/SINR evaluation to the accelerated-QT helper module.
    return lam.evaluate_comm_only_state(
        K=problem["K"],
        w=problem["w"],
        d=problem["d"],
        tau=problem["tau"],
        kappa_S=problem["kappa_S"],
        kappa_V1=problem["kappa_V1"],
        kappa_K1=problem["kappa_K1"],
        kappa_V0=problem["kappa_V0"],
        kappa_K0=problem["kappa_K0"],
        kappa_DAC1=problem["kappa_DAC1"],
        kappa_DAC0=problem["kappa_DAC0"],
        kappa_Th1=problem["kappa_Th1"],
        kappa_ADC1=problem["kappa_ADC1"],
        kappa_M1=problem["kappa_M1"],
        kappa_M0=problem["kappa_M0"],
        kappa_Th0=problem["kappa_Th0"],
        kappa_ADC0=problem["kappa_ADC0"],
        P_bar=P_bar,
        P_tilda=P_tilda,
        compute_sinr_exact_fn=main.compute_SINR_exact,
        compute_wsr_exact_fn=main.compute_WSR_exact,
    )


def compute_mu(problem: Dict[str, Any], state: Dict[str, Any], P_bar: np.ndarray, P_tilda: np.ndarray) -> np.ndarray:
    """Compute communication auxiliary variable mu from the main code."""
    # Reuse the main code auxiliary-variable update for consistency.
    return main.update_auxiliary(
        problem["K"],
        problem["w"],
        state["gamma"],
        problem["kappa_S"],
        problem["kappa_V1"],
        problem["kappa_K1"],
        problem["kappa_V0"],
        problem["kappa_K0"],
        problem["kappa_DAC1"],
        problem["kappa_DAC0"],
        problem["kappa_M1"],
        problem["kappa_M0"],
        problem["kappa_Th1"],
        problem["kappa_ADC1"],
        P_bar,
        P_tilda,
        state["P_th"],
        state["P_adc"],
    )


# =============================================================================
# SINGLE-BLOCK D/LAMBDA RUNNERS
# =============================================================================


def run_lambda_fixed_pbar_opt_ptilda(
    *,
    problem: Dict[str, Any],
    P_bar_fixed: np.ndarray,
    P_tilda_init: np.ndarray,
    max_iters: int = MAX_ITERS,
    epsilon: float = EPSILON,
    eps_power: float = EPS_POWER,
    eps_lambda: float = EPS_LAMBDA,
    use_backtracking: bool = USE_BACKTRACKING,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Case 1: keep P_bar fixed and optimize only P_tilda using D/lambda-QT."""
    # This sweep is communication-only, so sensing power is fixed to zero.
    # Convert the selected fractions into actual per-user fixed initial powers.
    K = problem["K"]
    P_prime_fixed = np.zeros(K, dtype=float)
    P_bar_fixed = np.asarray(P_bar_fixed, dtype=float).reshape(K)
    P_tilda = np.asarray(P_tilda_init, dtype=float).reshape(K)

    # Ensure the initial data-power vector is feasible under fixed P_bar.
    P_tilda = lam.clip_P_tilda_joint(
        P_tilda,
        P_bar=P_bar_fixed,
        P_prime=P_prime_fixed,
        P_total_max=problem["P_total_max"],
        eps=eps_power,
    )

    # Store the initial WSR before applying any lambda-QT updates.
    initial_state = evaluate(problem, P_bar_fixed, P_tilda)
    full_history = [float(initial_state["comm_wsr"])]

    # Histories are saved for diagnostics and step-size reports.
    P_tilda_history = [P_tilda.copy()]
    alpha_history = []
    lambda_history = []
    D_update_history = []
    D_matrix_history = []
    grad_history = []
    eta_history = []
    delta_history = []

    converged = False

    # Main single-block loop: only P_tilda is updated in this case.
    for t in range(max_iters):
        # Keep the old iterate so backtracking can interpolate from it.
        old_P_tilda = P_tilda.copy()
        old_wsr = float(full_history[-1])
        old_state = evaluate(problem, P_bar_fixed, old_P_tilda)
        # Update QT auxiliary variable using the current fixed-Pbar state.
        mu = compute_mu(problem, old_state, P_bar_fixed, old_P_tilda)

        # Compute the candidate D/lambda update for the data-power block.
        P_tilda_target, D_info = lam.update_P_tilda_using_D_lambda(
            K=K,
            w=problem["w"],
            gamma=old_state["gamma"],
            mu=mu,
            P_bar=P_bar_fixed,
            P_tilda=old_P_tilda,
            P_prime_fixed=P_prime_fixed,
            P_total_max=problem["P_total_max"],
            kappa_S=problem["kappa_S"],
            kappa_V1=problem["kappa_V1"],
            kappa_K1=problem["kappa_K1"],
            kappa_V0=problem["kappa_V0"],
            kappa_K0=problem["kappa_K0"],
            kappa_M1=problem["kappa_M1"],
            kappa_M0=problem["kappa_M0"],
            eps_power=eps_power,
            eps_lambda=eps_lambda,
        )

        # Backtracking scale. eta=1 means accepting the full D/lambda step.
        eta = 1.0
        accepted = False

        # Accept only a candidate that does not reduce the communication WSR.
        if use_backtracking:
            while eta >= 1e-6:
                P_candidate = old_P_tilda + eta * (P_tilda_target - old_P_tilda)
                candidate_state = evaluate(problem, P_bar_fixed, P_candidate)
                if candidate_state["comm_wsr"] >= old_wsr - 1e-12:
                    P_tilda = P_candidate
                    new_wsr = float(candidate_state["comm_wsr"])
                    accepted = True
                    break
                eta *= 0.5

            if not accepted:
                P_tilda = old_P_tilda
                new_wsr = old_wsr
                eta = 0.0
        else:
            P_tilda = P_tilda_target
            new_wsr = float(evaluate(problem, P_bar_fixed, P_tilda)["comm_wsr"])
            accepted = True

        # Record objective change and all per-iteration lambda/alpha values.
        delta = float(new_wsr - old_wsr)
        full_history.append(new_wsr)
        P_tilda_history.append(P_tilda.copy())
        alpha_history.append(np.asarray(D_info["alpha"], dtype=float).copy())
        lambda_history.append(np.asarray(D_info["lambda"], dtype=float).copy())
        D_update_history.append(np.asarray(D_info["D_update"], dtype=float).copy())
        D_matrix_history.append(np.asarray(D_info["D_matrix"], dtype=float).copy())
        grad_history.append(np.asarray(D_info["gradient"], dtype=float).copy())
        eta_history.append(float(eta))
        delta_history.append(delta)

        if verbose:
            print(
                f"fixed-Pbar opt-Ptilda iter={t + 1:03d} | "
                f"WSR={new_wsr:.8f} | delta={delta:.3e} | eta={eta:.3e} | accepted={accepted}"
            )

        # Stop when the WSR change becomes smaller than the requested tolerance.
        if t > 0 and abs(delta) < epsilon:
            converged = True
            break

    # Package the final powers and all histories for saving/plotting.
    full_history_arr = np.asarray(full_history, dtype=float)
    return {
        "case": "fixed_Pbar_opt_Ptilda",
        "optimized_block": "P_tilda",
        "fixed_block": "P_bar",
        "P_bar_fixed": P_bar_fixed,
        "P_tilda_initial": np.asarray(P_tilda_init, dtype=float).reshape(K),
        "P_tilda_opt": P_tilda,
        "P_bar_opt": P_bar_fixed,
        "P_prime_fixed": P_prime_fixed,
        "initial_comm_wsr": float(full_history_arr[0]),
        "final_comm_wsr": float(full_history_arr[-1]),
        "full_comm_history": full_history_arr,
        "P_tilda_history": np.asarray(P_tilda_history, dtype=float),
        "alpha_history": np.asarray(alpha_history, dtype=float),
        "lambda_history": np.asarray(lambda_history, dtype=float),
        "D_update_history": np.asarray(D_update_history, dtype=float),
        "D_matrix_history": np.asarray(D_matrix_history, dtype=float),
        "grad_history": np.asarray(grad_history, dtype=float),
        "eta_history": np.asarray(eta_history, dtype=float),
        "delta_history": np.asarray(delta_history, dtype=float),
        "converged": converged,
        "iterations": len(full_history_arr) - 1,
        "monotonic": bool(np.all(np.diff(full_history_arr) >= -1e-10)),
    }


def run_lambda_fixed_ptilda_opt_pbar(
    *,
    problem: Dict[str, Any],
    P_tilda_fixed: np.ndarray,
    P_bar_init: np.ndarray,
    max_iters: int = MAX_ITERS,
    epsilon: float = EPSILON,
    eps_power: float = EPS_POWER,
    eps_lambda: float = EPS_LAMBDA,
    lambda_mode_bar: str = LAMBDA_MODE_BAR,
    use_backtracking: bool = USE_BACKTRACKING,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Case 2: keep P_tilda fixed and optimize only P_bar using D/lambda-QT."""
    # This sweep is communication-only, so sensing power is fixed to zero.
    # In this case P_tilda is fixed and only P_bar is optimized.
    K = problem["K"]
    P_prime_fixed = np.zeros(K, dtype=float)
    P_tilda_fixed = np.asarray(P_tilda_fixed, dtype=float).reshape(K)
    P_bar = np.asarray(P_bar_init, dtype=float).reshape(K)

    # Ensure the initial pilot-power vector is feasible under fixed P_tilda.
    P_bar = lam.clip_P_bar_joint(
        P_bar,
        P_tilda=P_tilda_fixed,
        P_prime=P_prime_fixed,
        P_total_max=problem["P_total_max"],
        eps=eps_power,
    )

    # Store the initial WSR before applying any lambda-QT updates.
    initial_state = evaluate(problem, P_bar, P_tilda_fixed)
    full_history = [float(initial_state["comm_wsr"])]

    # Histories are saved for diagnostics and step-size reports.
    P_bar_history = [P_bar.copy()]
    alpha_history = []
    lambda_history = []
    D_update_history = []
    D_aug_history = []
    grad_history = []
    eta_history = []
    delta_history = []

    converged = False

    for t in range(max_iters):
        # Keep the old iterate so backtracking can interpolate from it.
        old_P_bar = P_bar.copy()
        old_wsr = float(full_history[-1])
        old_state = evaluate(problem, old_P_bar, P_tilda_fixed)
        # Update QT auxiliary variable using the current fixed-Ptilda state.
        mu = compute_mu(problem, old_state, old_P_bar, P_tilda_fixed)

        # Compute the candidate D/lambda update for the pilot-power block.
        P_bar_target, D_info = lam.update_P_bar_using_D_lambda(
            K=K,
            w=problem["w"],
            gamma=old_state["gamma"],
            mu=mu,
            P_bar=old_P_bar,
            P_tilda=P_tilda_fixed,
            P_prime_fixed=P_prime_fixed,
            P_total_max=problem["P_total_max"],
            kappa_S=problem["kappa_S"],
            kappa_V1=problem["kappa_V1"],
            kappa_K1=problem["kappa_K1"],
            kappa_V0=problem["kappa_V0"],
            kappa_K0=problem["kappa_K0"],
            kappa_DAC1=problem["kappa_DAC1"],
            kappa_DAC0=problem["kappa_DAC0"],
            kappa_Th1=problem["kappa_Th1"],
            kappa_ADC1=problem["kappa_ADC1"],
            kappa_M1=problem["kappa_M1"],
            kappa_M0=problem["kappa_M0"],
            kappa_Th0=problem["kappa_Th0"],
            kappa_ADC0=problem["kappa_ADC0"],
            eps_power=eps_power,
            eps_lambda=eps_lambda,
            lambda_mode=lambda_mode_bar,
        )

        eta = 1.0
        accepted = False

        if use_backtracking:
            while eta >= 1e-6:
                P_candidate = old_P_bar + eta * (P_bar_target - old_P_bar)
                candidate_state = evaluate(problem, P_candidate, P_tilda_fixed)
                if candidate_state["comm_wsr"] >= old_wsr - 1e-12:
                    P_bar = P_candidate
                    new_wsr = float(candidate_state["comm_wsr"])
                    accepted = True
                    break
                eta *= 0.5

            if not accepted:
                P_bar = old_P_bar
                new_wsr = old_wsr
                eta = 0.0
        else:
            P_bar = P_bar_target
            new_wsr = float(evaluate(problem, P_bar, P_tilda_fixed)["comm_wsr"])
            accepted = True

        delta = float(new_wsr - old_wsr)
        full_history.append(new_wsr)
        P_bar_history.append(P_bar.copy())
        alpha_history.append(np.asarray(D_info["alpha"], dtype=float).copy())
        lambda_history.append(np.asarray(D_info["lambda"], dtype=float).copy())
        D_update_history.append(np.asarray(D_info["D_update"], dtype=float).copy())
        D_aug_history.append(np.asarray(D_info["D_aug"], dtype=float).copy())
        grad_history.append(np.asarray(D_info["gradient"], dtype=float).copy())
        eta_history.append(float(eta))
        delta_history.append(delta)

        if verbose:
            print(
                f"fixed-Ptilda opt-Pbar iter={t + 1:03d} | "
                f"WSR={new_wsr:.8f} | delta={delta:.3e} | eta={eta:.3e} | accepted={accepted}"
            )

        if t > 0 and abs(delta) < epsilon:
            converged = True
            break

    full_history_arr = np.asarray(full_history, dtype=float)
    return {
        "case": "fixed_Ptilda_opt_Pbar",
        "optimized_block": "P_bar",
        "fixed_block": "P_tilda",
        "P_tilda_fixed": P_tilda_fixed,
        "P_bar_initial": np.asarray(P_bar_init, dtype=float).reshape(K),
        "P_bar_opt": P_bar,
        "P_tilda_opt": P_tilda_fixed,
        "P_prime_fixed": P_prime_fixed,
        "initial_comm_wsr": float(full_history_arr[0]),
        "final_comm_wsr": float(full_history_arr[-1]),
        "full_comm_history": full_history_arr,
        "P_bar_history": np.asarray(P_bar_history, dtype=float),
        "alpha_history": np.asarray(alpha_history, dtype=float),
        "lambda_history": np.asarray(lambda_history, dtype=float),
        "D_update_history": np.asarray(D_update_history, dtype=float),
        "D_aug_history": np.asarray(D_aug_history, dtype=float),
        "grad_history": np.asarray(grad_history, dtype=float),
        "eta_history": np.asarray(eta_history, dtype=float),
        "delta_history": np.asarray(delta_history, dtype=float),
        "converged": converged,
        "iterations": len(full_history_arr) - 1,
        "monotonic": bool(np.all(np.diff(full_history_arr) >= -1e-10)),
    }


# =============================================================================
# SAVE / PLOT HELPERS
# =============================================================================


def save_step_sizes_txt(result: Dict[str, Any], save_path: str) -> None:
    """Save per-iteration per-user alpha/lambda values to a text file."""
    # Convert histories to arrays so each row can be written as numeric columns.
    alpha = np.asarray(result["alpha_history"], dtype=float)
    lamb = np.asarray(result["lambda_history"], dtype=float)
    eta = np.asarray(result["eta_history"], dtype=float).reshape(-1)
    wsr = np.asarray(result["full_comm_history"], dtype=float)[1:]
    delta = np.asarray(result["delta_history"], dtype=float).reshape(-1)

    # Protect against saving an empty run with no lambda-QT iterations.
    if alpha.size == 0:
        raise ValueError(f"No step-size history to save for case {result['case']}.")

    # One row per iteration; each row contains summary stats and user-wise values.
    n_iter, K = alpha.shape
    rows = []
    for i in range(n_iter):
        rows.append(
            np.concatenate(
                [
                    np.array(
                        [
                            i + 1,
                            np.mean(alpha[i]),
                            np.min(alpha[i]),
                            np.max(alpha[i]),
                            np.mean(lamb[i]),
                            np.min(lamb[i]),
                            np.max(lamb[i]),
                            eta[i],
                            wsr[i],
                            delta[i],
                        ],
                        dtype=float,
                    ),
                    alpha[i],
                    lamb[i],
                ]
            )
        )

    # Header names match the numeric columns written by np.savetxt.
    header_cols = [
        "iter",
        "mean_alpha",
        "min_alpha",
        "max_alpha",
        "mean_lambda",
        "min_lambda",
        "max_lambda",
        "eta",
        "comm_wsr",
        "delta_wsr",
    ]
    header_cols += [f"alpha_u{k}" for k in range(K)]
    header_cols += [f"lambda_u{k}" for k in range(K)]

    # Create the destination folder before writing the file.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savetxt(
        save_path,
        np.asarray(rows, dtype=float),
        header=" ".join(header_cols),
        fmt="%.12e",
    )
    print(f"[SAVE] step sizes TXT: {save_path}")


def save_result_npz(result: Dict[str, Any], save_path: str) -> None:
    """Save full numerical histories for later checking."""
    # Store full arrays in NPZ format for later debugging or plotting.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        case=result["case"],
        optimized_block=result["optimized_block"],
        full_comm_history=result["full_comm_history"],
        alpha_history=result["alpha_history"],
        lambda_history=result["lambda_history"],
        D_update_history=result["D_update_history"],
        grad_history=result["grad_history"],
        eta_history=result["eta_history"],
        delta_history=result["delta_history"],
        P_bar_opt=result["P_bar_opt"],
        P_tilda_opt=result["P_tilda_opt"],
        P_prime_fixed=result["P_prime_fixed"],
        pbar_fraction=np.asarray([result.get("pbar_fraction", np.nan)], dtype=float),
        ptilda_fraction=np.asarray([result.get("ptilda_fraction", np.nan)], dtype=float),
    )
    print(f"[SAVE] full histories NPZ: {save_path}")


def plot_convergence(
    *,
    result_pbar_fixed: Dict[str, Any],
    result_ptilda_fixed: Dict[str, Any],
    title: str,
    save_path: str,
) -> None:
    # Plot both single-block convergence histories on one figure.
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.asarray(result_pbar_fixed["full_comm_history"], dtype=float),
        marker="o",
        linewidth=2,
        label="Fixed P_bar, optimize P_tilda",
    )
    plt.plot(
        np.asarray(result_ptilda_fixed["full_comm_history"], dtype=float),
        marker="s",
        linewidth=2,
        label="Fixed P_tilda, optimize P_bar",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Communication WSR")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] convergence plot: {save_path}")


def plot_mean_alpha(
    *,
    result_pbar_fixed: Dict[str, Any],
    result_ptilda_fixed: Dict[str, Any],
    title: str,
    save_path: str,
) -> None:
    # Plot the mean learned/derived alpha value across users per iteration.
    alpha_1 = np.asarray(result_pbar_fixed["alpha_history"], dtype=float)
    alpha_2 = np.asarray(result_ptilda_fixed["alpha_history"], dtype=float)

    plt.figure(figsize=(8, 5))
    if alpha_1.size:
        plt.plot(alpha_1.mean(axis=1), marker="o", linewidth=2, label="alpha_tilda: fixed P_bar")
    if alpha_2.size:
        plt.plot(alpha_2.mean(axis=1), marker="s", linewidth=2, label="alpha_bar: fixed P_tilda")
    plt.xlabel("Iteration")
    plt.ylabel("Mean alpha = 1 / lambda")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] mean-alpha plot: {save_path}")


def plot_power_sweep_summary(summary_rows: Iterable[Dict[str, Any]], save_path: str) -> None:
    """Plot final WSR vs Pmax for every fraction pair and case."""
    # Materialize summary rows because they are scanned several times for plotting.
    rows = list(summary_rows)
    pmax_vals = sorted(set(float(r["Pmax_db"]) for r in rows))
    fraction_pairs = sorted(
        set((float(r["pbar_fraction"]), float(r["ptilda_fraction"])) for r in rows)
    )

    # Average across setups for one case/fraction/power combination.
    def mean_final(case_name: str, pmax_db: float, pb_frac: float, pt_frac: float) -> float:
        vals = [
            r["final_comm_wsr"]
            for r in rows
            if r["case"] == case_name
            and float(r["Pmax_db"]) == pmax_db
            and abs(float(r["pbar_fraction"]) - pb_frac) < 1e-12
            and abs(float(r["ptilda_fraction"]) - pt_frac) < 1e-12
        ]
        return float(np.mean(vals)) if vals else np.nan

    plt.figure(figsize=(11, 7))
    for pb_frac, pt_frac in fraction_pairs:
        y_case1 = [mean_final("fixed_Pbar_opt_Ptilda", p, pb_frac, pt_frac) for p in pmax_vals]
        y_case2 = [mean_final("fixed_Ptilda_opt_Pbar", p, pb_frac, pt_frac) for p in pmax_vals]
        plt.plot(
            pmax_vals,
            y_case1,
            marker="o",
            linewidth=1.5,
            label=f"fixed Pbar={pb_frac:.1f}, opt Ptilda",
        )
        plt.plot(
            pmax_vals,
            y_case2,
            marker="s",
            linewidth=1.5,
            linestyle="--",
            label=f"fixed Ptilda={pt_frac:.1f}, opt Pbar",
        )

    plt.xlabel("Pmax (dB)")
    plt.ylabel("Final communication WSR")
    plt.title("Lambda-QT fraction sweep")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] power-sweep plot: {save_path}")


def save_summary_csv(summary_rows: Iterable[Dict[str, Any]], save_path: str) -> None:
    rows = list(summary_rows)
    # Fixed CSV column order for easy loading in Excel/MATLAB/Python.
    cols = [
        "case",
        "setup_idx",
        "Pmax_db",
        "pbar_fraction",
        "ptilda_fraction",
        "initial_comm_wsr",
        "final_comm_wsr",
        "gain",
        "iterations",
        "converged",
        "monotonic",
        "mean_P_bar_opt",
        "mean_P_tilda_opt",
        "sum_P_bar_opt",
        "sum_P_tilda_opt",
    ]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"[SAVE] summary CSV: {save_path}")


# =============================================================================
# MAIN SWEEP
# =============================================================================


def run_for_one_setup_power(
    *,
    pmax_db: float,
    setup_idx: int,
    pbar_fraction: float,
    ptilda_fraction: float,
    out_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Build all kappa coefficients and helper arrays for this setup and Pmax.
    problem = build_communication_problem(
        mat_file=MAT_FILE,
        pmax_db=pmax_db,
        setup_idx=setup_idx,
        K_use=K_USE,
        M_use=M_USE,
        N_use=N_USE,
        sigma_w2=SIGMA_W2,
    )

    K = problem["K"]
    pmax_lin = problem["Pmax_lin"]

    pbar_fraction = float(pbar_fraction)
    ptilda_fraction = float(ptilda_fraction)

    # Fractions must be feasible because P_bar + P_tilda cannot exceed Pmax.
    if pbar_fraction < 0.0 or ptilda_fraction < 0.0 or pbar_fraction + ptilda_fraction > 1.0 + 1e-12:
        raise ValueError(
            f"Infeasible fractions: pbar_fraction={pbar_fraction}, "
            f"ptilda_fraction={ptilda_fraction}; sum must be <= 1."
        )

    P_bar_fixed = pbar_fraction * pmax_lin * np.ones(K, dtype=float)
    P_tilda_init_for_case1 = ptilda_fraction * pmax_lin * np.ones(K, dtype=float)

    P_tilda_fixed = ptilda_fraction * pmax_lin * np.ones(K, dtype=float)
    P_bar_init_for_case2 = pbar_fraction * pmax_lin * np.ones(K, dtype=float)

    # Case 1: freeze P_bar at its fraction and optimize P_tilda only.
    result_case1 = run_lambda_fixed_pbar_opt_ptilda(
        problem=problem,
        P_bar_fixed=P_bar_fixed,
        P_tilda_init=P_tilda_init_for_case1,
        max_iters=MAX_ITERS,
        epsilon=EPSILON,
        eps_power=EPS_POWER,
        eps_lambda=EPS_LAMBDA,
        use_backtracking=USE_BACKTRACKING,
        verbose=VERBOSE,
    )

    # Case 2: freeze P_tilda at its fraction and optimize P_bar only.
    result_case2 = run_lambda_fixed_ptilda_opt_pbar(
        problem=problem,
        P_tilda_fixed=P_tilda_fixed,
        P_bar_init=P_bar_init_for_case2,
        max_iters=MAX_ITERS,
        epsilon=EPSILON,
        eps_power=EPS_POWER,
        eps_lambda=EPS_LAMBDA,
        lambda_mode_bar=LAMBDA_MODE_BAR,
        use_backtracking=USE_BACKTRACKING,
        verbose=VERBOSE,
    )

    # Attach metadata so saved files and CSV rows identify the fraction pair.
    for res in (result_case1, result_case2):
        res["pbar_fraction"] = pbar_fraction
        res["ptilda_fraction"] = ptilda_fraction

    # Build a filesystem-safe tag for this setup/power/fraction combination.
    frac_tag = f"PbarFrac_{pbar_fraction:.1f}_PtildaFrac_{ptilda_fraction:.1f}".replace(".", "p")
    tag = f"setup_{setup_idx}_Pmax_{pmax_db:g}dB_{frac_tag}"

    # Always save compact per-iteration alpha/lambda text summaries.
    save_step_sizes_txt(
        result_case1,
        os.path.join(out_dir, "step_sizes", f"step_sizes_fixed_Pbar_opt_Ptilda_{tag}.txt"),
    )
    save_step_sizes_txt(
        result_case2,
        os.path.join(out_dir, "step_sizes", f"step_sizes_fixed_Ptilda_opt_Pbar_{tag}.txt"),
    )

    # save_result_npz(
    #     result_case1,
    #     os.path.join(out_dir, "npz", f"histories_fixed_Pbar_opt_Ptilda_{tag}.npz"),
    # )
    # save_result_npz(
    #     result_case2,
    #     os.path.join(out_dir, "npz", f"histories_fixed_Ptilda_opt_Pbar_{tag}.npz"),
    # )

    # plot_convergence(
    #     result_pbar_fixed=result_case1,
    #     result_ptilda_fixed=result_case2,
    #     title=f"Lambda-QT convergence, setup {setup_idx}, Pmax={pmax_db:g} dB, Pbar={pbar_fraction:.1f}, Ptilda={ptilda_fraction:.1f}",
    #     save_path=os.path.join(out_dir, "plots", f"convergence_{tag}.png"),
    # )

    # plot_mean_alpha(
    #     result_pbar_fixed=result_case1,
    #     result_ptilda_fixed=result_case2,
    #     title=f"Mean alpha=1/lambda, setup {setup_idx}, Pmax={pmax_db:g} dB, Pbar={pbar_fraction:.1f}, Ptilda={ptilda_fraction:.1f}",
    #     save_path=os.path.join(out_dir, "plots", f"mean_alpha_{tag}.png"),
    # )

    # Print a compact terminal summary for the two cases.
    print("\n" + "=" * 90)
    print(f"Summary for setup={setup_idx}, Pmax={pmax_db:g} dB, Pbar frac={pbar_fraction:.1f}, Ptilda frac={ptilda_fraction:.1f}")
    print("=" * 90)
    for res in [result_case1, result_case2]:
        print(
            f"{res['case']:<28} | initial={res['initial_comm_wsr']:.8f} | "
            f"final={res['final_comm_wsr']:.8f} | gain={res['final_comm_wsr'] - res['initial_comm_wsr']:.8f} | "
            f"iters={res['iterations']} | converged={res['converged']} | monotonic={res['monotonic']} | "
            f"mean Pbar={np.mean(res['P_bar_opt']):.6e} | mean Ptilda={np.mean(res['P_tilda_opt']):.6e}"
        )

    return result_case1, result_case2


def main_sweep() -> None:
    # Main driver: loop over setups, powers, and fraction pairs.
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Each completed case contributes one row to the final CSV summary.
    summary_rows = []

    # Full nested sweep: setup -> Pmax -> fraction pair.
    for setup_idx in SETUP_LIST:
        for pmax_db in PMAX_DB_LIST:
            for pbar_fraction, ptilda_fraction in FRACTION_PAIRS:
                result_case1, result_case2 = run_for_one_setup_power(
                    pmax_db=float(pmax_db),
                    setup_idx=int(setup_idx),
                    pbar_fraction=float(pbar_fraction),
                    ptilda_fraction=float(ptilda_fraction),
                    out_dir=out_dir,
                )

                # Store only scalar summaries in the CSV; full histories are saved separately.
                for res in [result_case1, result_case2]:
                    summary_rows.append(
                        {
                            "case": res["case"],
                            "setup_idx": int(setup_idx),
                            "Pmax_db": float(pmax_db),
                            "pbar_fraction": float(res["pbar_fraction"]),
                            "ptilda_fraction": float(res["ptilda_fraction"]),
                            "initial_comm_wsr": float(res["initial_comm_wsr"]),
                            "final_comm_wsr": float(res["final_comm_wsr"]),
                            "gain": float(res["final_comm_wsr"] - res["initial_comm_wsr"]),
                            "iterations": int(res["iterations"]),
                            "converged": bool(res["converged"]),
                            "monotonic": bool(res["monotonic"]),
                            "mean_P_bar_opt": float(np.mean(res["P_bar_opt"])),
                            "mean_P_tilda_opt": float(np.mean(res["P_tilda_opt"])),
                            "sum_P_bar_opt": float(np.sum(res["P_bar_opt"])),
                            "sum_P_tilda_opt": float(np.sum(res["P_tilda_opt"])),
                        }
                    )

    # Save aggregate outputs after all runs finish.
    save_summary_csv(summary_rows, os.path.join(out_dir, "fixed_power_cases_summary.csv"))
    plot_power_sweep_summary(summary_rows, os.path.join(out_dir, "plots", "fixed_power_cases_power_sweep.png"))

    print("\nDone. Main outputs are in:")
    print(f"  {out_dir}")


# Execute the sweep only when this file is run directly, not when imported.
if __name__ == "__main__":
    main_sweep()
