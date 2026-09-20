"""Adversarial motion-prior discriminator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl_amp.modules import EmpiricalNormalization, MLP


class AMPDiscriminator(nn.Module):
    """Classify short motion windows as policy-generated or expert motion."""

    def __init__(
        self,
        frame_dim: int,
        history_steps: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 256),
        activation: str = "elu",
        reward_scale: float = 1.0,
        task_reward_weight: float = 0.0,
        gradient_penalty: float = 10.0,
        normalization_until: int = 100_000_000,
        reward_mode: str = "lsq",
        classification_margin: float = 0.8,
        logit_regularization: float = 0.05,
    ) -> None:
        super().__init__()
        if frame_dim < 1 or history_steps < 1:
            raise ValueError("frame_dim and history_steps must both be positive.")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        if not 0.0 <= task_reward_weight <= 1.0:
            raise ValueError("task_reward_weight must be in [0, 1].")
        if reward_mode not in {"softplus", "lsq"}:
            raise ValueError("reward_mode must be either 'softplus' or 'lsq'.")
        if not 0.0 < classification_margin <= 1.0:
            raise ValueError("classification_margin must be in (0, 1].")
        if logit_regularization < 0.0:
            raise ValueError("logit_regularization must be non-negative.")

        self.frame_dim = int(frame_dim)
        self.history_steps = int(history_steps)
        self.window_dim = self.frame_dim * self.history_steps
        self.reward_scale = float(reward_scale)
        self.task_reward_weight = float(task_reward_weight)
        self.gradient_penalty_scale = float(gradient_penalty)
        self.reward_mode = reward_mode
        self.classification_margin = float(classification_margin)
        self.logit_regularization_scale = float(logit_regularization)
        self.normalizer = EmpiricalNormalization(self.frame_dim, until=normalization_until)
        self.network = MLP(self.window_dim, 1, list(hidden_dims), activation)

    def _as_sequence(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim == 3 and tuple(samples.shape[1:]) == (self.history_steps, self.frame_dim):
            return samples
        if samples.ndim == 2 and samples.shape[1] == self.window_dim:
            return samples.reshape(-1, self.history_steps, self.frame_dim)
        raise ValueError(
            "expected AMP samples shaped "
            f"[batch, {self.history_steps}, {self.frame_dim}] or [batch, {self.window_dim}], "
            f"got {tuple(samples.shape)}."
        )

    def normalize(self, samples: torch.Tensor) -> torch.Tensor:
        sequence = self._as_sequence(samples)
        frames = sequence.reshape(-1, self.frame_dim)
        return self.normalizer(frames).reshape(sequence.shape[0], self.window_dim)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.network(self.normalize(samples)).squeeze(-1)

    @torch.no_grad()
    def quality_from_score(self, score: torch.Tensor) -> torch.Tensor:
        """Map discriminator scores to bounded motion quality without changing its loss."""
        if self.reward_mode == "lsq":
            return torch.clamp(1.0 - 0.25 * torch.square(score - 1.0), min=0.0, max=1.0)
        reference = F.softplus(score.new_tensor(1.0))
        return torch.clamp(F.softplus(score) / reference, max=1.0)

    @torch.no_grad()
    def style_reward(self, samples: torch.Tensor, step_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounded AMP reward and raw discriminator score."""
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        score = self.forward(samples)
        # TODO: 根据判别器输出计算 AMP 风格奖励。
        # quality = TODO: 将 score 转换为有界的动作质量。
        # style_rewards = TODO: 根据动作质量、step_dt 和 reward_scale 计算风格奖励。
        # 数学形式：r_amp = dt * beta * q(D_psi(x))。
        # 其中 q(.) 为有界风格质量函数，beta 为奖励缩放系数。
        raise NotImplementedError("TODO: 计算 AMP 风格奖励。")

    def mix_rewards(self, style_rewards: torch.Tensor, task_rewards: torch.Tensor) -> torch.Tensor:
        # TODO: 补全奖励混合公式。
        # task_weight = TODO: 读取任务奖励权重。
        # mixed_rewards = TODO: 混合 style_rewards 和 task_rewards。
        # 数学形式：r_t = (1 - alpha) r_amp + alpha r_task。
        # alpha 越大，策略越偏向速度跟踪等任务目标；alpha 越小，策略越偏向专家动作风格。
        raise NotImplementedError("TODO: 混合 AMP 风格奖励和任务奖励。")

    def loss(self, policy_samples: torch.Tensor, expert_samples: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute least-squares classification and an expert R1 penalty."""
        policy_input = self.normalize(policy_samples)
        expert_input = self.normalize(expert_samples).detach().requires_grad_(True)
        policy_score = self.network(policy_input).squeeze(-1)
        expert_score = self.network(expert_input).squeeze(-1)

        policy_loss = torch.square(policy_score + self.classification_margin).mean()
        expert_loss = torch.square(expert_score - self.classification_margin).mean()
        classification = 0.5 * (policy_loss + expert_loss)
        logit_regularization = 0.5 * self.logit_regularization_scale * (
            torch.square(policy_score).mean() + torch.square(expert_score).mean()
        )
        expert_gradient = torch.autograd.grad(
            outputs=expert_score.sum(),
            inputs=expert_input,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_penalty = 0.5 * self.gradient_penalty_scale * torch.square(expert_gradient).sum(dim=-1).mean()
        total = classification + logit_regularization + gradient_penalty
        return {
            "total": total,
            "classification": classification,
            "logit_regularization": logit_regularization,
            "gradient_penalty": gradient_penalty,
            "policy_score": policy_score.mean().detach(),
            "expert_score": expert_score.mean().detach(),
            "policy_score_std": policy_score.std(unbiased=False).detach(),
            "expert_score_std": expert_score.std(unbiased=False).detach(),
        }

    @torch.no_grad()
    def update_normalization(self, *sample_sets: torch.Tensor) -> None:
        for samples in sample_sets:
            frames = self._as_sequence(samples).reshape(-1, self.frame_dim)
            self.normalizer.update(frames)
