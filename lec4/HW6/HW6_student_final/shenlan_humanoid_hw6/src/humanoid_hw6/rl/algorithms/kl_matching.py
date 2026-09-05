"""HW6 KL Matching 蒸馏算法。

本文件包含 TODO 7/7。
快速定位：grep -n "【作业 TODO" src/humanoid_hw6/rl/algorithms/
"""

from __future__ import annotations

from humanoid_hw6.rl.algorithms.distillation_ppo import DistillationOutput, DistillationPPO


class KlMatchingPPO(DistillationPPO):
  """PPO with an additional KL-matching loss on the student actor."""

  def __init__(
    self,
    actor,
    critic,
    storage,
    *args,
    kl_coef: float = 0.1,
    kl_coef_min: float = 0.0,
    kl_coef_anneal_iters: int = 10_000,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, *args, **kwargs)
    self.kl_coef_start = float(kl_coef)
    self.kl_coef_min = float(kl_coef_min)
    self.kl_coef_anneal_iters = int(kl_coef_anneal_iters)
    self.num_kl_updates = 0

  def _current_distillation_coef(self) -> float:
    # >>> HOMEWORK_TODO_7A_START
    # 【作业 TODO 7/7 · 系数退火】KL matching 的 distill_coef
    #
    # 目标：从 kl_coef_start 线性退火到 kl_coef_min。
    # 提示：调用 distillation_utils.linear_anneal(...)。
    # 提示：step 使用 self.num_kl_updates。
    # 提示：duration 使用 self.kl_coef_anneal_iters。
    raise NotImplementedError("Implement KL-matching coefficient schedule.")
    # <<< HOMEWORK_TODO_7A_END

  def _compute_distillation_output(self, batch) -> DistillationOutput:
    # >>> HOMEWORK_TODO_7B_START
    # 【作业 TODO 7/7 · 蒸馏集成】KL matching distillation
    #
    # 目标：组装 KL matching 的 loss 与诊断指标。
    # 提示：在 torch.no_grad() 下调用
    #       self.actor.teacher_distribution_params(batch.observations)。
    # 提示：读取 self.actor.output_distribution_params 作为 student 参数。
    # 提示：用 diagonal_gaussian_kl(...) 计算标量 KL(teacher || student)。
    # 提示：用 gaussian_matching_metrics(...) 计算 detached diagnostics。
    # 提示：返回 DistillationOutput(loss=..., metrics=...)。
    # 提示：KL 方向必须是 forward：KL(teacher || student)。
    # 提示：此 KL 与 PPO 的 desired_kl 无关。
    # 提示：不要修改 distillation_ppo.py。
    #
    # 验证：tests/test_instructor_solutions.py::test_kl_matching_integration_matches_reference
    raise NotImplementedError("Implement KL-matching distillation output.")
    # <<< HOMEWORK_TODO_7B_END

  def _format_distillation_metrics(self, mean_distill_loss: float) -> dict[str, float]:
    return {"kl": mean_distill_loss}

  def _on_distillation_update_end(self) -> None:
    self.num_kl_updates += 1
