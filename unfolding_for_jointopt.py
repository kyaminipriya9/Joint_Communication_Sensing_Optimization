"""
joint_comm_sensing_unfolding.py

Standalone deep-unfolding PGD module for joint communication + sensing power allocation.

Variables:
    P_bar[k]   : communication pilot power
    P_tilda[k] : communication data power
    P_prime[k] : sensing power

Constraint per user:
    P_bar[k] + P_tilda[k] + P_prime[k] <= P_total_max

This file intentionally contains only unfolding-related code:
    - tensor conversion helpers
    - JointCommSensingUnfoldingPGD model
    - communication SINR/WSR functions
    - sensing SINR/WSR and exact sensing-gradient functions
    - joint three-block projection

It does NOT include:
    - .mat loading
    - kappa construction
    - classical FP/QT solver
    - plotting or train/test loops
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
# import torch.nn.functional as F

TensorLike = Union[torch.Tensor, float, int]


def to_torch_1d(x, device=None, dtype=torch.float64) -> torch.Tensor:
    """Convert a vector to shape [1, K] or [1, T]."""
    return torch.as_tensor(x, dtype=dtype, device=device).reshape(1, -1)


def to_torch_2d(x, device=None, dtype=torch.float64) -> torch.Tensor:
    """Convert a 2-D array to shape [1, dim1, dim2]."""
    return torch.as_tensor(x, dtype=dtype, device=device).unsqueeze(0)


def scalar_to_torch(x, device=None, dtype=torch.float64) -> torch.Tensor:
    """Convert scalar to torch scalar."""
    return torch.as_tensor(x, dtype=dtype, device=device)


def _batch_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1-D or 2-D tensor, got shape {tuple(x.shape)}")


def _batch_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2-D or 3-D tensor, got shape {tuple(x.shape)}")


def _scalar_like(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


# def _positive_raw_init(value: float, dtype=torch.float64) -> torch.Tensor:
#     """Initialize raw parameter r such that softplus(r) ~= value."""
#     v = torch.as_tensor(value, dtype=dtype)
#     return torch.log(torch.expm1(v))


class JointCommSensingUnfoldingPGD(nn.Module):
    """
    Deep-unfolded PGD model for joint communication-sensing power optimization.

    Communication part:
        Same FP/QT transformed gradients as the communication-only unfolding code.

    Sensing part:
        Exact gradient of sum_q omega_q log2(1 + S_q / D_q), where
        S_q = sum_k a_sens[q,k] P_prime[k]
        D_q = sum_k b_sens[q,k] P_prime[k] + n_sens[q]

    Joint objective:
        joint = w_c * communication_WSR + w_s * sensing_WSR
    """

    def __init__(
        self,
        K: int,
        num_layers: int = 20,
        num_pgd_steps: int = 20,
        init_step_bar: float = 0.1,
        init_step_tilda: float = 0.1,
        init_step_prime: float = 0.1,
        enforce_full_power: bool = False,
        # use_softplus_steps: bool = True,
    ):
        super().__init__()
        self.K = int(K)
        self.num_layers = int(num_layers)
        self.num_pgd_steps = int(num_pgd_steps)
        self.enforce_full_power = bool(enforce_full_power)
        # self.use_softplus_steps = bool(use_softplus_steps)

        total_steps = self.num_layers * self.num_pgd_steps

        self.step_Pbar = nn.Parameter(
            init_step_bar * torch.ones(total_steps, dtype=torch.float64)
        )

        self.step_Ptilda = nn.Parameter(
            init_step_tilda * torch.ones(total_steps, dtype=torch.float64)
        )

        self.step_Pprime = nn.Parameter(
            init_step_prime * torch.ones(total_steps, dtype=torch.float64)
        )
        # if self.use_softplus_steps:
        #     self.raw_step_Pbar = nn.Parameter(
        #         _positive_raw_init(init_step_bar) * torch.ones(total_steps, dtype=torch.float64)
        #     )
        #     self.raw_step_Ptilda = nn.Parameter(
        #         _positive_raw_init(init_step_tilda) * torch.ones(total_steps, dtype=torch.float64)
        #     )
        #     self.raw_step_Pprime = nn.Parameter(
        #         _positive_raw_init(init_step_prime) * torch.ones(total_steps, dtype=torch.float64)
        #     )
        # else:
        #     self.raw_step_Pbar = nn.Parameter(init_step_bar * torch.ones(total_steps, dtype=torch.float64))
        #     self.raw_step_Ptilda = nn.Parameter(init_step_tilda * torch.ones(total_steps, dtype=torch.float64))
        #     self.raw_step_Pprime = nn.Parameter(init_step_prime * torch.ones(total_steps, dtype=torch.float64))

    # @property
    # def step_Pbar(self) -> torch.Tensor:
    #     return F.softplus(self.raw_step_Pbar) if self.use_softplus_steps else self.raw_step_Pbar

    # @property
    # def step_Ptilda(self) -> torch.Tensor:
    #     return F.softplus(self.raw_step_Ptilda) if self.use_softplus_steps else self.raw_step_Ptilda

    # @property
    # def step_Pprime(self) -> torch.Tensor:
    #     return F.softplus(self.raw_step_Pprime) if self.use_softplus_steps else self.raw_step_Pprime

    def mui_row_sum(self, kappa_M: torch.Tensor, P_tilda: torch.Tensor) -> torch.Tensor:
        """sum_{i != k} P_tilda[i] * kappa_M[k,i]."""
        K = kappa_M.shape[-1]
        mask = (1.0 - torch.eye(K, dtype=kappa_M.dtype, device=kappa_M.device)).unsqueeze(0)
        return torch.sum(P_tilda.unsqueeze(1) * kappa_M * mask, dim=2)

    def mui_col_sum(self, kappa_M: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        """sum_{i != k} coeff[i] * kappa_M[i,k]."""
        K = kappa_M.shape[-1]
        mask = (1.0 - torch.eye(K, dtype=kappa_M.dtype, device=kappa_M.device)).unsqueeze(0)
        return torch.sum(coeff.unsqueeze(2) * kappa_M * mask, dim=1)

    def compute_Ak(self, k: int, kappa_S: torch.Tensor, P_tilda: torch.Tensor) -> torch.Tensor:
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
        gamma_all = []
        for k in range(self.K):
            Ak = self.compute_Ak(k, kappa_S, P_tilda)
            Bk = self.compute_Bk(k, kappa_V1, kappa_K1, kappa_DAC1, kappa_M1, P_tilda)
            Ck = self.compute_Ck(k, kappa_V0, kappa_K0, kappa_DAC0, kappa_M0, P_tilda)
            den = Bk * P_bar[:, k] + Ck + P_th[:, k] + P_adc[:, k]
            gamma_all.append(Ak * P_bar[:, k] / (den + 1e-30))
        return torch.stack(gamma_all, dim=1)

    def compute_comm_wsr_torch(self, gamma: torch.Tensor, w: torch.Tensor, d: TensorLike, tau: TensorLike) -> torch.Tensor:
        d_t = _scalar_like(d, gamma)
        tau_t = _scalar_like(tau, gamma)
        comm_wsr = torch.sum(w * torch.log2(1.0 + torch.clamp(gamma, min=0.0)), dim=1)
        return comm_wsr * (d_t / tau_t)

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
        """Communication QT auxiliary mu. Use w_eff = w_c * w."""
        mu_all = []
        for k in range(self.K):
            Ak = self.compute_Ak(k, kappa_S, P_tilda)
            Bk = self.compute_Bk(k, kappa_V1, kappa_K1, kappa_DAC1, kappa_M1, P_tilda)
            Ck = self.compute_Ck(k, kappa_V0, kappa_K0, kappa_DAC0, kappa_M0, P_tilda)
            numer = w_eff[:, k] * (1.0 + gamma[:, k]) * Ak * P_bar[:, k]
            denom = (Ak + Bk) * P_bar[:, k] + Ck + P_th[:, k] + P_adc[:, k]
            mu_all.append(torch.sqrt(torch.clamp(numer, min=0.0)) / (denom + 1e-30))
        return torch.stack(mu_all, dim=1)

    def compute_sensing_torch(
        self,
        P_prime: torch.Tensor,
        a_sens: torch.Tensor,
        b_sens: torch.Tensor,
        n_sens: torch.Tensor,
        target_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sensing_sinr [B,Q] and raw sensing_wsr [B]."""
        S = torch.sum(a_sens * P_prime.unsqueeze(1), dim=2)
        D = torch.sum(b_sens * P_prime.unsqueeze(1), dim=2) + n_sens
        sensing_sinr = S / (D + 1e-30)
        if target_weights is None:
            target_weights = torch.ones_like(sensing_sinr)
        sensing_wsr = torch.sum(target_weights * torch.log2(1.0 + torch.clamp(sensing_sinr, min=0.0)), dim=1)
        return sensing_sinr, sensing_wsr

    def update_sensing_auxiliary_ldt_qt_torch(
        self,
        P_prime: torch.Tensor,
        a_sens: torch.Tensor,
        b_sens: torch.Tensor,
        n_sens: torch.Tensor,
        target_weights: Optional[torch.Tensor],
        w_s: TensorLike,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LDT/QT sensing auxiliary updates.

        S_q = sum_k a_qk P'_k
        D_q = sum_k b_qk P'_k + n_q

        LDT:
            mu_q = S_q / D_q

        QT:
            y_q = sqrt(w_s * omega_q * (1 + mu_q) * S_q) / (S_q + D_q)

        Returns:
            S      : [B, Q]
            D      : [B, Q]
            mu_s   : [B, Q]
            y_s    : [B, Q]
        """
        S = torch.sum(a_sens * P_prime.unsqueeze(1), dim=2)
        D = torch.sum(b_sens * P_prime.unsqueeze(1), dim=2) + n_sens

        S_safe = torch.clamp(S, min=eps)
        D_safe = torch.clamp(D, min=eps)

        mu_s = S_safe / D_safe

        if target_weights is None:
            target_weights = torch.ones_like(S_safe)

        w_s_t = _scalar_like(w_s, P_prime)
        rho = w_s_t * target_weights

        y_num = torch.sqrt(torch.clamp(rho * (1.0 + mu_s) * S_safe, min=0.0))
        y_den = S_safe + D_safe

        y_s = y_num / (y_den + eps)

        return S_safe, D_safe, mu_s, y_s


    def sensing_grad_ldt_qt_torch(
        self,
        P_prime: torch.Tensor,
        a_sens: torch.Tensor,
        b_sens: torch.Tensor,
        n_sens: torch.Tensor,
        mu_s: torch.Tensor,
        y_s: torch.Tensor,
        target_weights: Optional[torch.Tensor],
        w_s: TensorLike,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        LDT/QT transformed sensing gradient wrt P_prime.

        For fixed mu_q and y_q:

            grad_k =
                sum_q [
                    y_q * sqrt(w_s * omega_q * (1 + mu_q)) * a_qk / sqrt(S_q)
                    - y_q^2 * (a_qk + b_qk)
                ]

        Shape:
            output grad_P_prime : [B, K]
        """

        # Recompute S_q using current P_prime inside the PGD inner step.
        # mu_s and y_s are fixed within the current unfolding layer.
        S = torch.sum(a_sens * P_prime.unsqueeze(1), dim=2)
        S_safe = torch.clamp(S, min=eps)

        if target_weights is None:
            target_weights = torch.ones_like(S_safe)

        w_s_t = _scalar_like(w_s, P_prime)
        rho = w_s_t * target_weights

        # First term:
        # y_q * sqrt(rho_q * (1 + mu_q)) / sqrt(S_q)
        coeff1 = (
            y_s
            * torch.sqrt(torch.clamp(rho * (1.0 + mu_s), min=0.0))
            / torch.sqrt(S_safe)
        )  # [B, Q]

        grad_first = torch.sum(
            a_sens * coeff1.unsqueeze(2),
            dim=1,
        )  # [B, K]

        # Second term:
        # y_q^2 * (a_qk + b_qk)
        coeff2 = y_s ** 2  # [B, Q]

        grad_second = torch.sum(
            (a_sens + b_sens) * coeff2.unsqueeze(2),
            dim=1,
        )  # [B, K]

        grad_P_prime = grad_first - grad_second

        return grad_P_prime

    def normalize_joint_power_grads_torch(
        self,
        grad_P_bar: torch.Tensor,
        grad_P_tilda: torch.Tensor,
        grad_P_prime: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize all three gradient blocks separately for stable learned steps."""
        Pmax = _scalar_like(P_total_max, grad_P_bar)
        while Pmax.ndim < grad_P_bar.ndim:
            Pmax = Pmax.unsqueeze(-1)
        K = self.K
        rho_bar = torch.clamp(torch.amax(torch.abs(grad_P_bar), dim=1, keepdim=True), min=eps)
        rho_tilda = torch.clamp(torch.amax(torch.abs(grad_P_tilda), dim=1, keepdim=True), min=eps)
        rho_prime = torch.clamp(torch.amax(torch.abs(grad_P_prime), dim=1, keepdim=True), min=eps)
        return (
            (Pmax / (rho_bar * K)) * grad_P_bar,
            (Pmax / (rho_tilda * K)) * grad_P_tilda,
            (Pmax / (rho_prime * K)) * grad_P_prime,
        )


    def _pmax_like_torch(
        self,
        P_total_max: TensorLike,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert scalar or vector P_total_max to shape [B, K].
        """
        Pmax = _scalar_like(P_total_max, ref)

        if Pmax.ndim == 0:
            return Pmax * torch.ones_like(ref)

        if Pmax.ndim == 1:
            return Pmax.reshape(1, -1).expand_as(ref)

        if Pmax.ndim == 2:
            return Pmax.to(dtype=ref.dtype, device=ref.device)

        raise ValueError(f"Unsupported P_total_max shape: {tuple(Pmax.shape)}")


    def clip_P_bar_joint_torch(
        self,
        P_bar_new: torch.Tensor,
        P_tilda: torch.Tensor,
        P_prime: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        Pmax = self._pmax_like_torch(P_total_max, P_bar_new)
        upper = torch.clamp(Pmax - P_tilda - P_prime - eps, min=eps)
        return torch.minimum(torch.clamp(P_bar_new, min=eps), upper)


    def clip_P_tilda_joint_torch(
        self,
        P_tilda_new: torch.Tensor,
        P_bar: torch.Tensor,
        P_prime: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        Pmax = self._pmax_like_torch(P_total_max, P_tilda_new)
        upper = torch.clamp(Pmax - P_bar - P_prime - eps, min=eps)
        return torch.minimum(torch.clamp(P_tilda_new, min=eps), upper)


    def clip_P_prime_joint_torch(
        self,
        P_prime_new: torch.Tensor,
        P_bar: torch.Tensor,
        P_tilda: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        Pmax = self._pmax_like_torch(P_total_max, P_prime_new)
        upper = torch.clamp(Pmax - P_bar - P_tilda - eps, min=eps)
        return torch.minimum(torch.clamp(P_prime_new, min=eps), upper)


    def sequential_clip_joint_torch(
        self,
        P_bar_candidate: torch.Tensor,
        P_tilda_candidate: torch.Tensor,
        P_prime_candidate: torch.Tensor,
        P_bar_old: torch.Tensor,
        P_tilda_old: torch.Tensor,
        P_prime_old: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classical-style sequential clipping order:

            1) clip P_bar first
            2) clip P_tilda next
            3) clip P_prime last

        This intentionally gives priority to P_bar, then P_tilda, then P_prime,
        matching the classical sequential clipping behavior.
        """

        P_bar_new = self.clip_P_bar_joint_torch(
            P_bar_new=P_bar_candidate,
            P_tilda=P_tilda_old,
            P_prime=P_prime_old,
            P_total_max=P_total_max,
            eps=eps,
        )

        P_tilda_new = self.clip_P_tilda_joint_torch(
            P_tilda_new=P_tilda_candidate,
            P_bar=P_bar_new,
            P_prime=P_prime_old,
            P_total_max=P_total_max,
            eps=eps,
        )

        P_prime_new = self.clip_P_prime_joint_torch(
            P_prime_new=P_prime_candidate,
            P_bar=P_bar_new,
            P_tilda=P_tilda_new,
            P_total_max=P_total_max,
            eps=eps,
        )

        return P_bar_new, P_tilda_new, P_prime_new
    def project_joint_three_power_blocks_torch(
        self,
        P_bar: torch.Tensor,
        P_tilda: torch.Tensor,
        P_prime: torch.Tensor,
        P_total_max: TensorLike,
        eps: float = 1e-12,
        enforce_full_power: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Feasible projection for P_bar + P_tilda + P_prime <= P_total_max.
        If enforce_full_power=True, all users are projected to equality.
        """
        if enforce_full_power is None:
            enforce_full_power = self.enforce_full_power

        Pmax = _scalar_like(P_total_max, P_bar)
        while Pmax.ndim < P_bar.ndim:
            Pmax = Pmax.unsqueeze(-1)

        X = torch.stack([P_bar, P_tilda, P_prime], dim=-1)
        X = torch.clamp(X, min=eps)

        total = torch.sum(X, dim=-1)
        violation = total > Pmax
        if enforce_full_power:
            violation = torch.ones_like(violation, dtype=torch.bool)

        lower = torch.as_tensor(eps, dtype=X.dtype, device=X.device)
        budget = torch.clamp(Pmax - 3.0 * eps, min=eps)
        excess = torch.clamp(X - lower, min=0.0)
        excess_sum = torch.sum(excess, dim=-1, keepdim=True)
        X_scaled = lower + budget.unsqueeze(-1) * excess / (excess_sum + eps)
        X = torch.where(violation.unsqueeze(-1), X_scaled, X)
        return X[..., 0], X[..., 1], X[..., 2]

    def forward(
        self,
        P_bar_init: torch.Tensor,
        P_tilda_init: torch.Tensor,
        P_prime_init: torch.Tensor,
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
        a_sens: torch.Tensor,
        b_sens: torch.Tensor,
        n_sens: torch.Tensor,
        target_weights: Optional[torch.Tensor] = None,
        w_c: TensorLike = 1.0,
        w_s: TensorLike = 1.0,
        return_history: bool = True,
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-12

        P_bar = _batch_2d(P_bar_init).clone()
        P_tilda = _batch_2d(P_tilda_init).clone()
        P_prime = _batch_2d(P_prime_init).clone()
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
        a_sens = _batch_3d(a_sens)
        b_sens = _batch_3d(b_sens)
        n_sens = _batch_2d(n_sens)
        if target_weights is not None:
            target_weights = _batch_2d(target_weights)

        w_c_t = _scalar_like(w_c, P_bar)
        w_s_t = _scalar_like(w_s, P_bar)
        w_comm_eff = w_c_t * w

        P_bar, P_tilda, P_prime = self.project_joint_three_power_blocks_torch(
            P_bar, P_tilda, P_prime, P_total_max, eps=eps
        )

        joint_history, comm_history, sensing_history = [], [], []

        def compute_from_powers(P_bar_now, P_tilda_now, P_prime_now):
            P_th = P_bar_now * kappa_Th1 + kappa_Th0
            P_adc = P_bar_now * kappa_ADC1 + kappa_ADC0
            gamma = self.compute_comm_sinr_torch(
                P_bar_now, P_tilda_now, P_th, P_adc,
                kappa_S, kappa_V1, kappa_K1, kappa_V0, kappa_K0,
                kappa_DAC1, kappa_DAC0, kappa_M1, kappa_M0,
            )
            comm_wsr = self.compute_comm_wsr_torch(gamma, w, d, tau)
            sensing_sinr, sensing_wsr = self.compute_sensing_torch(P_prime_now, a_sens, b_sens, n_sens, target_weights)
            joint_obj = w_c_t * comm_wsr + w_s_t * sensing_wsr
            return gamma, comm_wsr, sensing_sinr, sensing_wsr, joint_obj, P_th, P_adc

        gamma, comm_wsr, sensing_sinr, sensing_wsr, joint_obj, P_th, P_adc = compute_from_powers(P_bar, P_tilda, P_prime)
        if return_history:
            joint_history.append(joint_obj)
            comm_history.append(comm_wsr)
            sensing_history.append(sensing_wsr)

        idx = 0
        for _layer in range(self.num_layers):
            gamma, comm_wsr, sensing_sinr, sensing_wsr, joint_obj, P_th, P_adc = compute_from_powers(P_bar, P_tilda, P_prime)
            mu = self.update_auxiliary_torch(
                gamma, P_bar, P_tilda, P_th, P_adc, w_comm_eff,
                kappa_S, kappa_V1, kappa_K1, kappa_V0, kappa_K0,
                kappa_DAC1, kappa_DAC0, kappa_M1, kappa_M0,
            )
            _, _, mu_s, y_s = self.update_sensing_auxiliary_ldt_qt_torch(
                P_prime=P_prime,
                a_sens=a_sens,
                b_sens=b_sens,
                n_sens=n_sens,
                target_weights=target_weights,
                w_s=w_s_t,
                eps=eps,
            )
            for _step in range(self.num_pgd_steps):
                a = w_comm_eff * (1.0 + gamma) * kappa_S

                H_bar = (
                    P_tilda * (kappa_S + kappa_V1 + kappa_K1)
                    + self.mui_row_sum(kappa_M1, P_tilda)
                    + kappa_DAC1 + kappa_Th1 + kappa_ADC1
                )
                grad_P_bar = (
                    mu * torch.sqrt(torch.clamp(a * P_tilda, min=eps)) / torch.sqrt(torch.clamp(P_bar, min=eps))
                    - (mu ** 2) * H_bar
                )

                self_tilda = P_bar * (kappa_S + kappa_V1 + kappa_K1) + kappa_V0 + kappa_K0
                interference_to_others = self.mui_col_sum(kappa_M1, (mu ** 2) * P_bar) + self.mui_col_sum(kappa_M0, mu ** 2)
                H_tilda = (mu ** 2) * self_tilda + interference_to_others
                grad_P_tilda = (
                    mu * torch.sqrt(torch.clamp(a * P_bar, min=eps)) / torch.sqrt(torch.clamp(P_tilda, min=eps))
                    - H_tilda
                )

                grad_P_prime = self.sensing_grad_ldt_qt_torch(
                    P_prime=P_prime,
                    a_sens=a_sens,
                    b_sens=b_sens,
                    n_sens=n_sens,
                    mu_s=mu_s,
                    y_s=y_s,
                    target_weights=target_weights,
                    w_s=w_s_t,
                    eps=eps,
                )

                grad_P_bar, grad_P_tilda, grad_P_prime = self.normalize_joint_power_grads_torch(
                    grad_P_bar, grad_P_tilda, grad_P_prime, P_total_max, eps=eps
                )
                step_bar = self.step_Pbar[idx]
                step_tilda = self.step_Ptilda[idx]
                step_prime = self.step_Pprime[idx]

                P_bar_cand = P_bar + step_bar * grad_P_bar
                P_tilda_cand = P_tilda + step_tilda * grad_P_tilda
                P_prime_cand = P_prime + step_prime * grad_P_prime
                # P_bar_cand = P_bar + self.step_Pbar[idx] * grad_P_bar
                # P_tilda_cand = P_tilda + self.step_Ptilda[idx] * grad_P_tilda
                # P_prime_cand = P_prime + self.step_Pprime[idx] * grad_P_prime
                
                P_bar, P_tilda, P_prime = self.sequential_clip_joint_torch(
                    P_bar_candidate=P_bar_cand,
                    P_tilda_candidate=P_tilda_cand,
                    P_prime_candidate=P_prime_cand,
                    P_bar_old=P_bar,
                    P_tilda_old=P_tilda,
                    P_prime_old=P_prime,
                    P_total_max=P_total_max,
                    eps=eps,
                )
                # P_bar, P_tilda, P_prime = self.project_joint_three_power_blocks_torch(
                #     P_bar_cand, P_tilda_cand, P_prime_cand, P_total_max, eps=eps
                # )
                idx += 1

            gamma, comm_wsr, sensing_sinr, sensing_wsr, joint_obj, P_th, P_adc = compute_from_powers(P_bar, P_tilda, P_prime)
            if return_history:
                joint_history.append(joint_obj)
                comm_history.append(comm_wsr)
                sensing_history.append(sensing_wsr)

        final_gamma, final_comm_wsr, final_sensing_sinr, final_sensing_wsr, final_joint_obj, _, _ = compute_from_powers(P_bar, P_tilda, P_prime)

        output = {
            "P_bar": P_bar,
            "P_tilda": P_tilda,
            "P_prime": P_prime,
            "comm_sinr": final_gamma,
            "sensing_sinr": final_sensing_sinr,
            "comm_wsr": final_comm_wsr,
            "sensing_wsr": final_sensing_wsr,
            "joint_obj": final_joint_obj,
        }
        if return_history:
            output["joint_history"] = torch.stack(joint_history, dim=0)
            output["comm_history"] = torch.stack(comm_history, dim=0)
            output["sensing_history"] = torch.stack(sensing_history, dim=0)
        return output
