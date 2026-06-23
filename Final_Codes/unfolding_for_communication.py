"""
unfolding_for_communication.py

Pure communication-only deep-unfolding PGD module.

This version has NO sensing optimization and NO sensing-power update.
It optimizes only:
    P_bar[k]   : communication pilot / channel-estimation power
    P_tilda[k] : communication data power

Communication-only constraint per user:
    P_bar[k] + P_tilda[k] <= P_total_max[k]

Important compatibility behavior:
    - P_prime is not optimized and is returned as exact zeros.
    - Optional old joint-style arguments such as P_prime_init, a_sens, b_sens,
      n_sens, target_weights, w_c and w_s are accepted only so older main files
      do not crash. They are ignored and no sensing quantities are computed.
    - Joint mode must continue to use the separate joint unfolding file, where
      P_prime >= eps is still enforced.

Clipping / update style:
    This keeps the sequential clipping style of the previous
    unfolding_for_communication.py:
        1) update/clip P_bar first using old P_tilda
        2) recompute P_tilda gradient using updated P_bar
        3) update/clip P_tilda using updated P_bar
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorLike = Union[torch.Tensor, float, int]

# This alias is used so functions can accept either plain Python scalars
# or torch tensors while keeping dtype/device conversion consistent.


# -----------------------------------------------------------------------------
# Tensor helpers used by the main script
# -----------------------------------------------------------------------------
def to_torch_1d(x, device=None, dtype=torch.float64) -> torch.Tensor:
    """Convert vector-like input to shape [1, K]."""
    return torch.as_tensor(x, dtype=dtype, device=device).reshape(1, -1)


def to_torch_2d(x, device=None, dtype=torch.float64) -> torch.Tensor:
    """Convert matrix-like input to shape [1, dim1, dim2]."""
    return torch.as_tensor(x, dtype=dtype, device=device).unsqueeze(0)


def scalar_to_torch(x, device=None, dtype=torch.float64) -> torch.Tensor:
    """Convert scalar input to a torch scalar."""
    return torch.as_tensor(x, dtype=dtype, device=device)


def _batch_2d(x: torch.Tensor) -> torch.Tensor:
    """Accept [K] or [B,K], return [B,K]."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1-D or 2-D tensor, got shape {tuple(x.shape)}")


def _batch_3d(x: torch.Tensor) -> torch.Tensor:
    """Accept [K,K] or [B,K,K], return [B,K,K]."""
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2-D or 3-D tensor, got shape {tuple(x.shape)}")


def _scalar_like(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
    """Convert scalar/tensor to same dtype and device as ref."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def _positive_raw_init(value: float, dtype=torch.float64) -> torch.Tensor:
    """Return raw r such that softplus(r) is approximately value."""
    v = torch.as_tensor(value, dtype=dtype)
    return torch.log(torch.expm1(v))


class CommunicationUnfoldingPGD(nn.Module):
    """
    Pure communication-only deep-unfolded PGD.

    Learned step sizes:
        step_Pbar
        step_Ptilda

    A zero step_Pprime buffer is exposed only for compatibility with older
    logging/checkpoint code that expects a Pprime step-size array.
    """

    def __init__(
        self,
        K: int,
        num_layers: int = 20,
        num_pgd_steps: int = 20,
        init_step_bar: float = 0.1,
        init_step_tilda: float = 0.1,
        init_step_prime: Optional[float] = None,  # ignored; compatibility only
        enforce_full_power: bool = False,
        use_softplus_steps: bool = True,
    ):
        super().__init__()

        # Store model dimensions. Each layer contains num_pgd_steps inner
        # projected-gradient updates, so total learnable steps = layers × steps.
        self.K = int(K)
        self.num_layers = int(num_layers)
        self.num_pgd_steps = int(num_pgd_steps)
        self.enforce_full_power = bool(enforce_full_power)
        self.use_softplus_steps = bool(use_softplus_steps)

        total_steps = self.num_layers * self.num_pgd_steps

        if self.use_softplus_steps:
            # Learn unconstrained raw parameters and pass them through softplus
            # so the actual step sizes remain positive during training.
            self.raw_step_Pbar = nn.Parameter(
                _positive_raw_init(init_step_bar) * torch.ones(total_steps, dtype=torch.float64)
            )
            self.raw_step_Ptilda = nn.Parameter(
                _positive_raw_init(init_step_tilda) * torch.ones(total_steps, dtype=torch.float64)
            )
        else:
            # Optional mode: learn step sizes directly without positivity mapping.
            self.raw_step_Pbar = nn.Parameter(
                init_step_bar * torch.ones(total_steps, dtype=torch.float64)
            )
            self.raw_step_Ptilda = nn.Parameter(
                init_step_tilda * torch.ones(total_steps, dtype=torch.float64)
            )

        # Not used in communication mode. Kept as a buffer so old save/load code
        # that reads/copies model.step_Pprime will still work.
        self.register_buffer("_step_Pprime_zero", torch.zeros(total_steps, dtype=torch.float64))

    @property
    def step_Pbar(self) -> torch.Tensor:
        return F.softplus(self.raw_step_Pbar) if self.use_softplus_steps else self.raw_step_Pbar

    @property
    def step_Ptilda(self) -> torch.Tensor:
        return F.softplus(self.raw_step_Ptilda) if self.use_softplus_steps else self.raw_step_Ptilda

    @property
    def step_Pprime(self) -> torch.Tensor:
        # Compatibility only. P_prime is exactly zero and never optimized here.
        return self._step_Pprime_zero

    # ------------------------------------------------------------------
    # MUI helpers
    # ------------------------------------------------------------------
    def mui_row_sum(self, kappa_M: torch.Tensor, P_tilda: torch.Tensor) -> torch.Tensor:
        """sum_{i != k} P_tilda[i] * kappa_M[k,i].

        Row-wise MUI: interference caused by all other users i to victim user k.
        """
        K = kappa_M.shape[-1]
        mask = (1.0 - torch.eye(K, dtype=kappa_M.dtype, device=kappa_M.device)).unsqueeze(0)
        return torch.sum(P_tilda.unsqueeze(1) * kappa_M * mask, dim=2)

    def mui_col_sum(self, kappa_M: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        """sum_{i != k} coeff[i] * kappa_M[i,k].

        Column-wise MUI: effect of variable user k on other users' denominators.
        """
        K = kappa_M.shape[-1]
        mask = (1.0 - torch.eye(K, dtype=kappa_M.dtype, device=kappa_M.device)).unsqueeze(0)
        return torch.sum(coeff.unsqueeze(2) * kappa_M * mask, dim=1)

    # ------------------------------------------------------------------
    # Communication SINR / WSR
    # ------------------------------------------------------------------
    def compute_Ak(self, k: int, kappa_S: torch.Tensor, P_tilda: torch.Tensor) -> torch.Tensor:
        """Desired-signal coefficient A_k = P_tilda[k] * kappa_S[k]."""
        return P_tilda[:, k] * kappa_S[:, k]

    def compute_Bk(
        self,
        k: int,
        kappa_V1: torch.Tensor,
        kappa_K1: torch.Tensor,
        kappa_DAC1: torch.Tensor,
        kappa_M1: torch.Tensor,
        P_tilda: torch.Tensor,
    ) -> torch.Tensor:
        """Pilot-power-dependent denominator coefficient B_k.

        Includes self error/distortion terms, multi-user interference terms,
        and the pilot-dependent DAC distortion coefficient.
        """
        self_term = P_tilda[:, k] * (kappa_V1[:, k] + kappa_K1[:, k])
        mui_term = torch.zeros_like(self_term)
        for i in range(self.K):
            if i != k:
                mui_term = mui_term + P_tilda[:, i] * kappa_M1[:, k, i]
        return self_term + mui_term + kappa_DAC1[:, k]

    def compute_Ck(
        self,
        k: int,
        kappa_V0: torch.Tensor,
        kappa_K0: torch.Tensor,
        kappa_DAC0: torch.Tensor,
        kappa_M0: torch.Tensor,
        P_tilda: torch.Tensor,
    ) -> torch.Tensor:
        """Pilot-power-independent denominator coefficient C_k.

        Includes terms that remain in the SINR denominator even without the
        multiplicative P_bar[k] factor.
        """
        self_term = P_tilda[:, k] * (kappa_V0[:, k] + kappa_K0[:, k])
        mui_term = torch.zeros_like(self_term)
        for i in range(self.K):
            if i != k:
                mui_term = mui_term + P_tilda[:, i] * kappa_M0[:, k, i]
        return self_term + mui_term + kappa_DAC0[:, k]

    def compute_comm_sinr_torch(
        self,
        P_bar: torch.Tensor,
        P_tilda: torch.Tensor,
        P_th: torch.Tensor,
        P_adc: torch.Tensor,
        kappa_S: torch.Tensor,
        kappa_V1: torch.Tensor,
        kappa_K1: torch.Tensor,
        kappa_V0: torch.Tensor,
        kappa_K0: torch.Tensor,
        kappa_DAC1: torch.Tensor,
        kappa_DAC0: torch.Tensor,
        kappa_M1: torch.Tensor,
        kappa_M0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute communication SINR for every user in the batch.

        SINR_k = A_k P_bar[k] / (B_k P_bar[k] + C_k + P_th[k] + P_adc[k]).
        """
        gamma_all = []
        for k in range(self.K):
            Ak = self.compute_Ak(k, kappa_S, P_tilda)
            Bk = self.compute_Bk(k, kappa_V1, kappa_K1, kappa_DAC1, kappa_M1, P_tilda)
            Ck = self.compute_Ck(k, kappa_V0, kappa_K0, kappa_DAC0, kappa_M0, P_tilda)
            den = Bk * P_bar[:, k] + Ck + P_th[:, k] + P_adc[:, k]
            gamma_all.append(Ak * P_bar[:, k] / (den + 1e-30))
        return torch.stack(gamma_all, dim=1)

    # Backward-compatible name used by some older snippets.
    compute_SINR_torch = compute_comm_sinr_torch

    def compute_comm_wsr_torch(
        self,
        gamma: torch.Tensor,
        w: torch.Tensor,
        d: TensorLike,
        tau: TensorLike,
    ) -> torch.Tensor:
        d_t = _scalar_like(d, gamma)
        tau_t = _scalar_like(tau, gamma)
        comm_wsr = torch.sum(w * torch.log2(1.0 + torch.clamp(gamma, min=0.0)), dim=1)
        return comm_wsr * (d_t / tau_t)

    # Backward-compatible name used by the standalone communication file.
    compute_WSR_torch = compute_comm_wsr_torch

    def update_auxiliary_torch(
        self,
        gamma: torch.Tensor,
        P_bar: torch.Tensor,
        P_tilda: torch.Tensor,
        P_th: torch.Tensor,
        P_adc: torch.Tensor,
        w_eff: torch.Tensor,
        kappa_S: torch.Tensor,
        kappa_V1: torch.Tensor,
        kappa_K1: torch.Tensor,
        kappa_V0: torch.Tensor,
        kappa_K0: torch.Tensor,
        kappa_DAC1: torch.Tensor,
        kappa_DAC0: torch.Tensor,
        kappa_M1: torch.Tensor,
        kappa_M0: torch.Tensor,
    ) -> torch.Tensor:
        """Communication QT auxiliary mu. w_eff is the effective comm weight."""
        mu_all = []
        for k in range(self.K):
            Ak = self.compute_Ak(k, kappa_S, P_tilda)
            Bk = self.compute_Bk(k, kappa_V1, kappa_K1, kappa_DAC1, kappa_M1, P_tilda)
            Ck = self.compute_Ck(k, kappa_V0, kappa_K0, kappa_DAC0, kappa_M0, P_tilda)

            numer = w_eff[:, k] * (1.0 + gamma[:, k]) * Ak * P_bar[:, k]
            denom = (Ak + Bk) * P_bar[:, k] + Ck + P_th[:, k] + P_adc[:, k]
            mu_all.append(torch.sqrt(torch.clamp(numer, min=0.0)) / (denom + 1e-30))
        return torch.stack(mu_all, dim=1)

    # ------------------------------------------------------------------
    # Two-block communication clipping and gradient normalization
    # ------------------------------------------------------------------
    def _pmax_like_torch(self, P_total_max: TensorLike, ref: torch.Tensor) -> torch.Tensor:
        """Convert scalar/vector/matrix P_total_max to shape [B,K]."""
        Pmax = _scalar_like(P_total_max, ref)

        if Pmax.ndim == 0:
            return Pmax * torch.ones_like(ref)
        if Pmax.ndim == 1:
            return Pmax.reshape(1, -1).expand_as(ref)
        if Pmax.ndim == 2:
            return Pmax.to(dtype=ref.dtype, device=ref.device)

        raise ValueError(f"Unsupported P_total_max shape: {tuple(Pmax.shape)}")

    def clip_P_bar_comm_torch(
        self,
        P_bar_new: torch.Tensor,
        P_tilda: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Clip P_bar using fixed P_tilda: P_bar + P_tilda <= Pmax."""
        Pmax = self._pmax_like_torch(P_total_max, P_bar_new)
        upper = torch.clamp(Pmax - P_tilda - eps, min=eps)
        return torch.minimum(torch.clamp(P_bar_new, min=eps), upper)

    def clip_P_tilda_comm_torch(
        self,
        P_tilda_new: torch.Tensor,
        P_bar: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Clip P_tilda using fixed P_bar: P_bar + P_tilda <= Pmax."""
        Pmax = self._pmax_like_torch(P_total_max, P_tilda_new)
        upper = torch.clamp(Pmax - P_bar - eps, min=eps)
        return torch.minimum(torch.clamp(P_tilda_new, min=eps), upper)

    def sequential_clip_comm_torch(
        self,
        P_bar_candidate: torch.Tensor,
        P_tilda_candidate: torch.Tensor,
        P_bar_old: torch.Tensor,
        P_tilda_old: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Communication-only version of the previous sequential clipping style:
            1) clip P_bar first using old P_tilda
            2) clip P_tilda next using updated P_bar
        """
        P_bar_new = self.clip_P_bar_comm_torch(
            P_bar_new=P_bar_candidate,
            P_tilda=P_tilda_old,
            P_total_max=P_total_max,
            eps=eps,
        )
        P_tilda_new = self.clip_P_tilda_comm_torch(
            P_tilda_new=P_tilda_candidate,
            P_bar=P_bar_new,
            P_total_max=P_total_max,
            eps=eps,
        )
        return P_bar_new, P_tilda_new

    def normalize_single_power_grad_torch(
        self,
        grad: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Normalize a gradient block to make learned step sizes stable.

        The largest absolute gradient entry is scaled to roughly Pmax/K, so
        one PGD update cannot become excessively large due to raw gradient scale.
        """
        Pmax = self._pmax_like_torch(P_total_max, grad)
        rho = torch.clamp(torch.amax(torch.abs(grad), dim=1, keepdim=True), min=eps)
        return (Pmax / (rho * self.K)) * grad


    # ------------------------------------------------------------------
    # Forward unrolled network
    # ------------------------------------------------------------------
    def forward(
        self,
        P_bar_init: torch.Tensor,
        P_tilda_init: torch.Tensor,
        w: torch.Tensor,
        P_total_max: TensorLike,
        d: TensorLike,
        tau: TensorLike,
        kappa_S: torch.Tensor,
        kappa_V1: torch.Tensor,
        kappa_K1: torch.Tensor,
        kappa_V0: torch.Tensor,
        kappa_K0: torch.Tensor,
        kappa_DAC1: torch.Tensor,
        kappa_DAC0: torch.Tensor,
        kappa_Th1: torch.Tensor,
        kappa_ADC1: torch.Tensor,
        kappa_Th0: torch.Tensor,
        kappa_ADC0: torch.Tensor,
        kappa_M1: torch.Tensor,
        kappa_M0: torch.Tensor,
        # The following sensing/joint arguments are accepted only to keep this
        # communication-only forward API compatible with the joint DU main code.
        # They are deleted below and do not affect the communication update.
        P_prime_init: Optional[torch.Tensor] = None,
        a_sens: Optional[torch.Tensor] = None,
        b_sens: Optional[torch.Tensor] = None,
        n_sens: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
        w_c: TensorLike = 1.0,
        w_s: TensorLike = 0.0,
        return_history: bool = True,
        debug_trace: bool = False,
        debug_layer: int = 0,
    ) -> Dict[str, torch.Tensor]:
        del P_prime_init, a_sens, b_sens, n_sens, target_weights, w_c, w_s, debug_trace, debug_layer

        eps = 1e-12

        # Convert all inputs to batched format. This lets the same code handle
        # a single setup [K] or a batch of setups [B, K].
        P_bar = _batch_2d(P_bar_init).clone()
        P_tilda = _batch_2d(P_tilda_init).clone()
        w = _batch_2d(w)

        kappa_S = _batch_2d(kappa_S)
        kappa_V1 = _batch_2d(kappa_V1)
        kappa_K1 = _batch_2d(kappa_K1)
        kappa_V0 = _batch_2d(kappa_V0)
        kappa_K0 = _batch_2d(kappa_K0)
        kappa_DAC1 = _batch_2d(kappa_DAC1)
        kappa_DAC0 = _batch_2d(kappa_DAC0)
        kappa_Th1 = _batch_2d(kappa_Th1)
        kappa_ADC1 = _batch_2d(kappa_ADC1)
        kappa_Th0 = _batch_2d(kappa_Th0)
        kappa_ADC0 = _batch_2d(kappa_ADC0)
        kappa_M1 = _batch_3d(kappa_M1)
        kappa_M0 = _batch_3d(kappa_M0)

        # Effective communication weights in the transformed objective.
        # The common d/tau factor is included to match the joint-unfolding convention.
        d_t = _scalar_like(d, P_bar)
        tau_t = _scalar_like(tau, P_bar)
        w_eff = (d_t / tau_t) * w

        # Initial communication-only feasibility clipping.
        P_bar, P_tilda = self.sequential_clip_comm_torch(
            P_bar_candidate=P_bar,
            P_tilda_candidate=P_tilda,
            P_bar_old=P_bar,
            P_tilda_old=P_tilda,
            P_total_max=P_total_max,
            eps=eps,
        )

        wsr_history = []

        # Empty debug trace is kept for compatibility with older plotting/debug code
        # that expects this key in the output dictionary.
        trace_rows = []

        def compute_from_powers(P_bar_now: torch.Tensor, P_tilda_now: torch.Tensor):
            # Hardware distortion terms depend on pilot power P_bar.
            P_th = P_bar_now * kappa_Th1 + kappa_Th0
            P_adc = P_bar_now * kappa_ADC1 + kappa_ADC0
            gamma = self.compute_comm_sinr_torch(
                P_bar_now,
                P_tilda_now,
                P_th,
                P_adc,
                kappa_S,
                kappa_V1,
                kappa_K1,
                kappa_V0,
                kappa_K0,
                kappa_DAC1,
                kappa_DAC0,
                kappa_M1,
                kappa_M0,
            )
            # Evaluate communication weighted sum-rate for current powers.
            comm_wsr = self.compute_comm_wsr_torch(gamma, w, d, tau)
            return gamma, comm_wsr, P_th, P_adc

        gamma, comm_wsr, P_th, P_adc = compute_from_powers(P_bar, P_tilda)
        if return_history:
            wsr_history.append(comm_wsr)

        idx = 0
        for _layer_idx in range(self.num_layers):
            # Recompute SINR/WSR and hardware terms at the start of each layer.
            gamma, comm_wsr, P_th, P_adc = compute_from_powers(P_bar, P_tilda)

            # QT auxiliary variable is updated once per unfolding layer and then
            # kept fixed during the inner PGD steps of that layer.
            mu = self.update_auxiliary_torch(
                gamma,
                P_bar,
                P_tilda,
                P_th,
                P_adc,
                w_eff,
                kappa_S,
                kappa_V1,
                kappa_K1,
                kappa_V0,
                kappa_K0,
                kappa_DAC1,
                kappa_DAC0,
                kappa_M1,
                kappa_M0,
            )

            for _pgd_step in range(self.num_pgd_steps):
                # Common numerator coefficient in the transformed QT objective.
                a = w_eff * (1.0 + gamma) * kappa_S

                # ----------------------------------------------------------
                # 1) P_bar update using current P_tilda, then clip P_bar.
                # ----------------------------------------------------------
                # Denominator slope for the P_bar block.
                # It includes self terms, MUI, DAC, thermal-noise, and ADC terms.
                H_bar = (
                    P_tilda * (kappa_S + kappa_V1 + kappa_K1)
                    + self.mui_row_sum(kappa_M1, P_tilda)
                    + kappa_DAC1
                    + kappa_Th1
                    + kappa_ADC1
                )
                # Normalized gradient-ascent direction for pilot power P_bar.
                grad_P_bar = (
                    mu
                    * torch.sqrt(torch.clamp(a * P_tilda, min=eps))
                    / torch.sqrt(torch.clamp(P_bar, min=eps))
                    - (mu ** 2) * H_bar
                )
                grad_P_bar = self.normalize_single_power_grad_torch(
                    grad_P_bar,
                    P_total_max,
                    eps=eps,
                )

                P_bar_old = P_bar
                P_tilda_old = P_tilda

                # Apply the learned step size and clip to maintain
                # P_bar + P_tilda <= P_total_max.
                P_bar_cand = P_bar_old + self.step_Pbar[idx] * grad_P_bar
                P_bar_new = self.clip_P_bar_comm_torch(
                    P_bar_new=P_bar_cand,
                    P_tilda=P_tilda_old,
                    P_total_max=P_total_max,
                    eps=eps,
                )

                # ----------------------------------------------------------
                # 2) Recompute P_tilda gradient using updated P_bar, then clip.
                # ----------------------------------------------------------
                # Self-denominator contribution for the P_tilda block,
                # evaluated using the updated P_bar.
                self_tilda = (
                    P_bar_new * (kappa_S + kappa_V1 + kappa_K1)
                    + kappa_V0
                    + kappa_K0
                )
                # Cross-user effect: changing P_tilda[k] also changes the
                # interference seen in other users' SINR denominators.
                interference_to_others = (
                    self.mui_col_sum(kappa_M1, (mu ** 2) * P_bar_new)
                    + self.mui_col_sum(kappa_M0, mu ** 2)
                )
                H_tilda = (mu ** 2) * self_tilda + interference_to_others

                # Normalized gradient-ascent direction for data power P_tilda.
                grad_P_tilda = (
                    mu
                    * torch.sqrt(torch.clamp(a * P_bar_new, min=eps))
                    / torch.sqrt(torch.clamp(P_tilda_old, min=eps))
                    - H_tilda
                )
                grad_P_tilda = self.normalize_single_power_grad_torch(
                    grad_P_tilda,
                    P_total_max,
                    eps=eps,
                )

                # Apply the learned step size and clip using the updated P_bar.
                P_tilda_cand = P_tilda_old + self.step_Ptilda[idx] * grad_P_tilda
                P_tilda_new = self.clip_P_tilda_comm_torch(
                    P_tilda_new=P_tilda_cand,
                    P_bar=P_bar_new,
                    P_total_max=P_total_max,
                    eps=eps,
                )

                P_bar = P_bar_new
                P_tilda = P_tilda_new
                idx += 1

            gamma, comm_wsr, P_th, P_adc = compute_from_powers(P_bar, P_tilda)
            if return_history:
                wsr_history.append(comm_wsr)

        final_gamma, final_comm_wsr, _, _ = compute_from_powers(P_bar, P_tilda)

        # Exact zero sensing power in communication-only mode.
        P_prime_zero = torch.zeros_like(P_bar)
        sensing_sinr_zero = torch.zeros((P_bar.shape[0], 1), dtype=P_bar.dtype, device=P_bar.device)
        sensing_wsr_zero = torch.zeros(P_bar.shape[0], dtype=P_bar.dtype, device=P_bar.device)

        # Keep both communication-only names and older generic names
        # (SINR, WSR, joint_obj) so existing main scripts continue to work.
        output: Dict[str, torch.Tensor] = {
            "P_bar": P_bar,
            "P_tilda": P_tilda,
            "P_prime": P_prime_zero,
            "comm_sinr": final_gamma,
            "SINR": final_gamma,
            "comm_wsr": final_comm_wsr,
            "WSR": final_comm_wsr,
            "sensing_sinr": sensing_sinr_zero,
            "sensing_wsr": sensing_wsr_zero,
            "joint_obj": final_comm_wsr,
            "debug_trace": trace_rows,
        }

        if return_history:
            wsr_stack = torch.stack(wsr_history, dim=0)
            sensing_stack = torch.zeros_like(wsr_stack)
            output["comm_history"] = wsr_stack
            output["WSR_history"] = wsr_stack
            output["joint_history"] = wsr_stack
            output["sensing_history"] = sensing_stack

        return output


# Backward-compatible aliases used by the current main script.
# The main code imports JointCommSensingUnfoldingPGD even in communication mode,
# so these aliases should be kept unless the main imports are changed.
JointCommSensingUnfoldingPGD = CommunicationUnfoldingPGD
DeepUnfoldingPGD = CommunicationUnfoldingPGD
