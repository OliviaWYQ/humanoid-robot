"""PPO and AMP hyperparameters for the G1 flat-ground tasks."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "AMP"
    amp_cfg: dict = MISSING


@configclass
class RslRlAMPActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic configuration with trainable exploration covariance."""

    learn_std: bool = True


@configclass
class G1AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    experiment_name = "unitree_g1_29dof_amp"
    amp_motion_profile: str = "mixed"
    # TODO: 补全 RSL-RL 观测分组。
    # 需要指定 actor、critic 和 AMP 判别器分别读取哪些 observation group。
    obs_groups = {}

    policy = RslRlAMPActorCriticCfg(
        init_noise_std=0.8,
        noise_std_type="scalar",
        learn_std=True,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlAMPAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg={
            "replay_buffer_size": 200_000,
            "discriminator_updates": 4,
            "discriminator_batch_size": 8192,
            "normalization_batch_size": 8192,
            "discriminator_learning_rate": 1.0e-4,
            "discriminator_weight_decay": 1.0e-4,
            "discriminator_output_weight_decay": 1.0e-3,
            "discriminator_max_grad_norm": 1.0,
            "discriminator_balance": {
                "enabled": True,
                "saturation_threshold": 0.7,
                "saturated_updates": 0,
            },
            "discriminator": {
                "hidden_dims": [512, 256],
                "activation": "elu",
                "reward_mode": "lsq",
                "classification_margin": 0.8,
                "logit_regularization": 0.05,
                "reward_scale": 5.0,
                "task_reward_weight": 0.4,
                "gradient_penalty": 10.0,
            },
        },
    )


@configclass
class G1AMPWalkRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_walk"
    amp_motion_profile: str = "walk"

    def __post_init__(self):
        self.algorithm.amp_cfg["discriminator"]["task_reward_weight"] = 0.6


@configclass
class G1AMPRunRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_run"
    amp_motion_profile: str = "run"


@configclass
class G1AMPOmniRunRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_omni_run"
    amp_motion_profile: str = "omni_run"


@configclass
class G1AMPWalkToRunRunnerCfg(G1AMPRunnerCfg):
    # TODO: 补全走跑任务的 runner 配置。
    # 需要设置实验名称、专家运动 profile，并调整任务奖励和 AMP 风格奖励的混合权重。
    pass


@configclass
class G1AMPDanceRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_dance"
    amp_motion_profile: str = "dance"


@configclass
class G1AMPMixedRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_mixed"
    amp_motion_profile: str = "mixed"


@configclass
class G1AMPPlayRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_play"
    amp_motion_profile: str = "auto"


G1FlatRslRlOnPolicyRunnerAmpCfg = G1AMPRunnerCfg
