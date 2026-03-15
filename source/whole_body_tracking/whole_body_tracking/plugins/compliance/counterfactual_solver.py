"""Local counterfactual optimization for compliance latent targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SolverResult:
    z_star_by_expert: torch.Tensor
    best_expert: torch.Tensor
    objective_values: torch.Tensor


class CounterfactualSolver:
    """Optimizes latent z with gradient/random/CEM backends."""

    def __init__(self, method: str = "gradient", steps: int = 16, lr: float = 0.05, samples: int = 64, elite_frac: float = 0.2):
        self.method = method
        self.steps = steps
        self.lr = lr
        self.samples = samples
        self.elite_frac = elite_frac

    def _objective(self, y_hat: torch.Tensor, rho: torch.Tensor, z: torch.Tensor, z_prev: torch.Tensor, z_nom: torch.Tensor, lambda_delta: float, lambda_nom: float) -> torch.Tensor:
        cost = (rho * y_hat).sum(dim=-1)
        return cost + lambda_delta * ((z - z_prev) ** 2).sum(dim=-1) + lambda_nom * ((z - z_nom) ** 2).sum(dim=-1)

    def solve(self, predictor, obs_hist: torch.Tensor, priv_hist: torch.Tensor, z_init: torch.Tensor, z_prev: torch.Tensor, z_nom: torch.Tensor, rho_m: torch.Tensor, lambda_delta: float = 1e-2, lambda_nom: float = 1e-2) -> SolverResult:
        bsz, n_exp = z_init.shape[:2]
        z_star = []
        obj_all = []
        for m in range(n_exp):
            rho = rho_m[m].view(1, -1).expand(bsz, -1)
            z0 = z_init[:, m]
            if self.method == "gradient":
                z = z0.detach().clone().requires_grad_(True)
                for _ in range(self.steps):
                    y = predictor(obs_hist, priv_hist, z)["mean"]
                    obj = self._objective(y, rho, z, z_prev, z_nom, lambda_delta, lambda_nom).mean()
                    (grad,) = torch.autograd.grad(obj, z)
                    z = (z - self.lr * grad).detach().requires_grad_(True)
                z_best = z.detach()
                obj_m = self._objective(predictor(obs_hist, priv_hist, z_best)["mean"], rho, z_best, z_prev, z_nom, lambda_delta, lambda_nom)
            elif self.method == "random":
                cand = z0.unsqueeze(1) + 0.1 * torch.randn(bsz, self.samples, z0.shape[-1], device=z0.device)
                flat = cand.reshape(-1, z0.shape[-1])
                y = predictor(obs_hist.repeat_interleave(self.samples, 0), priv_hist.repeat_interleave(self.samples, 0), flat)["mean"]
                obj = self._objective(y, rho.repeat_interleave(self.samples, 0), flat, z_prev.repeat_interleave(self.samples, 0), z_nom.repeat_interleave(self.samples, 0), lambda_delta, lambda_nom).reshape(bsz, self.samples)
                idx = obj.argmin(dim=-1)
                z_best = cand[torch.arange(bsz), idx]
                obj_m = obj[torch.arange(bsz), idx]
            else:  # CEM
                mean = z0
                std = torch.full_like(mean, 0.2)
                for _ in range(self.steps):
                    cand = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(bsz, self.samples, mean.shape[-1], device=mean.device)
                    flat = cand.reshape(-1, mean.shape[-1])
                    y = predictor(obs_hist.repeat_interleave(self.samples, 0), priv_hist.repeat_interleave(self.samples, 0), flat)["mean"]
                    obj = self._objective(y, rho.repeat_interleave(self.samples, 0), flat, z_prev.repeat_interleave(self.samples, 0), z_nom.repeat_interleave(self.samples, 0), lambda_delta, lambda_nom).reshape(bsz, self.samples)
                    k = max(1, int(self.samples * self.elite_frac))
                    elite_idx = obj.topk(k, largest=False).indices
                    elite = cand.gather(1, elite_idx.unsqueeze(-1).expand(-1, -1, mean.shape[-1]))
                    mean = elite.mean(dim=1)
                    std = elite.std(dim=1).clamp_min(1e-3)
                z_best = mean
                obj_m = self._objective(predictor(obs_hist, priv_hist, z_best)["mean"], rho, z_best, z_prev, z_nom, lambda_delta, lambda_nom)
            z_star.append(z_best)
            obj_all.append(obj_m)
        z_star = torch.stack(z_star, dim=1)
        obj_all = torch.stack(obj_all, dim=1)
        best = obj_all.argmin(dim=1)
        return SolverResult(z_star_by_expert=z_star, best_expert=best, objective_values=obj_all)
