"""HW6 Action Matching 蒸馏算法。

本文件包含 TODO 6/7。
快速定位：grep -n "【作业 TODO" src/humanoid_hw6/rl/algorithms/
"""

from __future__ import annotations

from humanoid_hw6.rl.algorithms.distillation_ppo import DistillationOutput, DistillationPPO


class ActionMatchingPPO(DistillationPPO):
  """PPO with an additional teacher action-matching loss on the student actor."""

  def __init__(
    self,
    actor,
    critic,
    storage,
    *args,
    bc_coef_start: float = 1.0,
    bc_coef_end: float = 0.0,
    bc_anneal_iters: int = 10_000,
    bc_loss_type: str = "mse",
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, *args, **kwargs)
    self.bc_coef_start = float(bc_coef_start)
    self.bc_coef_end = float(bc_coef_end)
    self.bc_anneal_iters = int(bc_anneal_iters)
    self.bc_loss_type = str(bc_loss_type)
    self.num_bc_updates = 0

  def _current_distillation_coef(self) -> float:
    # >>> HOMEWORK_TODO_6A_START
    # 【作业 TODO 6/7 · 系数退火】Action matching 的 distill_coef
    #
    # 目标：从 bc_coef_start 线性退火到 bc_coef_end。
    # 提示：调用 distillation_utils.linear_anneal(...)。
    # 提示：step 使用 self.num_bc_updates。
    # 提示：duration 使用 self.bc_anneal_iters。
    raise NotImplementedError("Implement action-matching coefficient schedule.")
    # <<< HOMEWORK_TODO_6A_END

  def _compute_distillation_output(self, batch) -> DistillationOutput:
    # >>> HOMEWORK_TODO_6B_START
    # 【作业 TODO 6/7 · 蒸馏集成】Action matching distillation
    #
    # 目标：组装 action matching 的 loss 与诊断指标。
    # 提示：在 torch.no_grad() 下调用 self.actor.teacher_forward(batch.observations)。
    # 提示：调用 self.actor(batch.observations) 得到 student actions。
    # 提示：用 action_regression_loss(...) 计算标量 loss。
    # 提示：用 action_matching_metrics(...) 计算 detached diagnostics。
    # 提示：返回 DistillationOutput(loss=..., metrics=...)。
    # 提示：teacher 不参与梯度。
    # 提示：不要修改 distillation_ppo.py。
    #
    # 验证：tests/test_instructor_solutions.py::test_action_matching_integration_matches_reference
    raise NotImplementedError("Implement action-matching distillation output.")
    # <<< HOMEWORK_TODO_6B_END

  def _format_distillation_metrics(self, mean_distill_loss: float) -> dict[str, float]:
    return {"bc": mean_distill_loss}

  def _on_distillation_update_end(self) -> None:
    self.num_bc_updates += 1
