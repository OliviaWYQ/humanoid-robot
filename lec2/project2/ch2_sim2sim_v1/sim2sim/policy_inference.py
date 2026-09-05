import numpy as np
import torch


class ActorFromCheckpoint(torch.nn.Module):
    """Build the actor network from an RSL-RL checkpoint state_dict."""

    def __init__(self, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        actor_state = {
            name.removeprefix("actor."): value
            for name, value in state_dict.items()
            if name.startswith("actor.")
        }
        if not actor_state:
            raise ValueError("No actor.* weights found in checkpoint")

        layer_ids = sorted(
            {
                int(name.split(".")[0])
                for name in actor_state
                if name.endswith(".weight")
            }
        )
        layers = []
        for layer_id in layer_ids:
            out_features, in_features = actor_state[f"{layer_id}.weight"].shape
            layers.append(torch.nn.Linear(in_features, out_features))
            if layer_id != layer_ids[-1]:
                layers.append(torch.nn.ELU())

        self.actor = torch.nn.Sequential(*layers)
        self.actor.load_state_dict(actor_state)

        # RSL-RL stores the empirical observation normalizer as a separate
        # module (`actor_obs_normalizer`) applied BEFORE the actor MLP.
        # Rebuild it here so raw checkpoints behave like the exported policy.
        if "actor_obs_normalizer._mean" in state_dict:
            self.register_buffer(
                "norm_mean", state_dict["actor_obs_normalizer._mean"].float()
            )
            self.register_buffer(
                "norm_std", state_dict["actor_obs_normalizer._std"].float()
            )
            # rsl_rl EmpiricalNormalization default eps
            self.norm_eps = 1e-2
        else:
            self.register_buffer("norm_mean", None)
            self.register_buffer("norm_std", None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.norm_mean is not None:
            obs = (obs - self.norm_mean) / (self.norm_std + self.norm_eps)
        return self.actor(obs)


class PolicyInference:
    """Load a .pt policy and run inference: numpy observation in, numpy action out.

    The policy file can be an exported TorchScript policy.pt or an RSL-RL
    checkpoint such as model_10000.pt.
    """

    def __init__(self, policy_file: str, device: str = "cpu"):
        self.policy_file = policy_file
        self.device = torch.device(device)
        self.policy, self.policy_type = self._load_policy(policy_file)

    def __str__(self):
        return f"{self.policy_file} ({self.policy_type})"

    def _load_policy(self, policy_file: str):
        try:
            policy = torch.jit.load(policy_file, map_location=self.device)
            policy.eval()
            return policy.to(self.device), "torchscript"
        except Exception:
            checkpoint = torch.load(policy_file, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            policy = ActorFromCheckpoint(state_dict).to(self.device)
            policy.eval()
            return policy, "rsl_rl_checkpoint"

    def forward(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy(obs_tensor)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.forward(obs)
