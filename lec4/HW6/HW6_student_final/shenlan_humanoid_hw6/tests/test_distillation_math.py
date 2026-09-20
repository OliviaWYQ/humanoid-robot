"""Independent numerical and gradient checks for the submitted HW6 algorithms."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Independent, Normal, kl_divergence

from humanoid_hw6.rl.algorithms.action_matching import ActionMatchingPPO
from humanoid_hw6.rl.algorithms.distillation_utils import (
  action_matching_metrics,
  action_regression_loss,
  diagonal_gaussian_kl,
  gaussian_matching_metrics,
  linear_anneal,
)
from humanoid_hw6.rl.algorithms.kl_matching import KlMatchingPPO


@pytest.mark.parametrize(
  "step,duration,expected",
  [
    (-1, 10, 1.0),
    (0, 10, 1.0),
    (5, 10, 0.55),
    (20, 10, 0.1),
    (5, 0, 0.1),
    (5, -1, 0.1),
  ],
)
def test_schedule_boundaries(step, duration, expected):
  assert linear_anneal(1.0, 0.1, step, duration) == pytest.approx(expected)


def test_huber_and_invalid_loss():
  student = torch.tensor([0.5, 2.0], requires_grad=True)
  loss = action_regression_loss(student, torch.zeros(2), "huber")
  assert loss.item() == pytest.approx((0.125 + 1.5) / 2)
  loss.backward()
  torch.testing.assert_close(student.grad, torch.tensor([0.25, 0.5]))
  with pytest.raises(ValueError):
    action_regression_loss(student, torch.zeros(2), "invalid")


def test_kl_matches_torch_distribution_and_gradients():
  generator = torch.Generator().manual_seed(6)
  teacher_mean = torch.randn(5, 3, generator=generator, dtype=torch.float64)
  teacher_std = torch.rand(5, 3, generator=generator, dtype=torch.float64) + 0.1
  student_mean = torch.randn(
    5, 3, generator=generator, dtype=torch.float64, requires_grad=True
  )
  student_std = (
    torch.rand(5, 3, generator=generator, dtype=torch.float64) + 0.1
  ).requires_grad_()
  actual = diagonal_gaussian_kl(teacher_mean, teacher_std, student_mean, student_std)
  expected = kl_divergence(
    Independent(Normal(teacher_mean, teacher_std), 1),
    Independent(Normal(student_mean, student_std), 1),
  ).mean()
  torch.testing.assert_close(actual, expected)
  for actual_grad, expected_grad in zip(
    torch.autograd.grad(actual, (student_mean, student_std)),
    torch.autograd.grad(expected, (student_mean, student_std)),
    strict=True,
  ):
    torch.testing.assert_close(actual_grad, expected_grad)


def test_kl_zero_std_is_finite():
  zeros = torch.zeros(2, 3)
  assert diagonal_gaussian_kl(zeros, zeros, zeros, zeros).item() == 0.0
  assert torch.isfinite(diagonal_gaussian_kl(zeros, zeros, torch.ones(2, 3), zeros))


@pytest.mark.parametrize("algorithm", [ActionMatchingPPO, KlMatchingPPO])
def test_integration_updates_student_only(algorithm):
  class Actor:
    def __init__(self):
      self.teacher = torch.nn.Parameter(torch.zeros(2, 3))
      self.student = torch.nn.Parameter(torch.ones(2, 3))
      self.std = torch.nn.Parameter(torch.full((2, 3), 1.5))

    def teacher_forward(self, obs):
      assert not torch.is_grad_enabled()
      return self.teacher * 1.0

    def teacher_distribution_params(self, obs):
      return self.teacher_forward(obs), torch.ones_like(self.teacher)

    def __call__(self, obs):
      return self.student * 1.0

    @property
    def output_distribution_params(self):
      return self.student, self.std

  actor = Actor()
  alg = SimpleNamespace(actor=actor, bc_loss_type="mse")
  output = algorithm._compute_distillation_output(alg, SimpleNamespace(observations={}))
  output.loss.backward()
  assert actor.teacher.grad is None
  assert actor.student.grad is not None
  assert actor.student.grad.abs().sum() > 0
  if algorithm is KlMatchingPPO:
    assert actor.std.grad is not None
    assert actor.std.grad.abs().sum() > 0
  assert all(isinstance(value, float) for value in output.metrics.values())


def test_algorithm_schedules_and_update_counters():
  action = SimpleNamespace(
    bc_coef_start=1.0, bc_coef_end=0.1, bc_anneal_iters=10, num_bc_updates=5
  )
  kl = SimpleNamespace(
    kl_coef_start=0.2, kl_coef_min=0.05, kl_coef_anneal_iters=10, num_kl_updates=5
  )
  assert ActionMatchingPPO._current_distillation_coef(action) == pytest.approx(0.55)
  assert KlMatchingPPO._current_distillation_coef(kl) == pytest.approx(0.125)
  ActionMatchingPPO._on_distillation_update_end(action)
  KlMatchingPPO._on_distillation_update_end(kl)
  assert action.num_bc_updates == kl.num_kl_updates == 6


def test_diagnostics_numeric_values():
  teacher = torch.zeros(2, 2)
  student = torch.tensor([[1.0, -1.0], [3.0, -3.0]], requires_grad=True)
  action = action_matching_metrics(student, teacher)
  assert action == pytest.approx({"action_mae": 2.0, "action_rmse": 5.0**0.5})
  gaussian = gaussian_matching_metrics(
    teacher, torch.ones(2, 2), student, torch.full((2, 2), 3.0)
  )
  assert gaussian == pytest.approx({"mean_rmse": 5.0**0.5, "std_rmse": 2.0})
