"""CPU contract smoke, without Isaac Sim. Run with a Python environment containing torch.

Executes actual source functions/classes with small Isaac Lab interface doubles.
This checks tensor behavior, not Isaac Lab integration or PPO training.
"""
from __future__ import annotations

import ast
import copy
import io
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'source/unitree_rl_lab/unitree_rl_lab/tasks'
NAV = TASKS / 'navigation'


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert len(nodes) == len(names), names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class Term:
    def __init__(self, func, params=None, scale=1.0, noise=None, **kwargs):
        self.func, self.params, self.scale, self.noise = func, params or {}, scale, noise


def configclass(cls):
    def init(self):
        for base in reversed(cls.__mro__):
            for name, value in vars(base).items():
                if isinstance(value, Term) or value is None:
                    setattr(self, name, copy.deepcopy(value))
        if hasattr(self, '__post_init__'):
            self.__post_init__()
    cls.__init__ = init
    return cls


class MDP:
    def __getattr__(self, name):
        def observation(env, **params):
            if name == 'generated_commands':
                assert params['command_name'] == 'pose_command'
            dims = {'joint_pos_rel': 29, 'joint_vel_rel': 29, 'last_action': 29,
                    'low_level_last_action': 29, 'generated_commands': 4,
                    'base_pos_z': 1, 'command_distance': 1}
            return torch.zeros(env.num_envs, dims.get(name, 3))
        return observation


def main():
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for path in NAV.rglob('*.py'):
        assert '_raise_homework_todo_' not in path.read_text()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Raise):
                assert 'HOMEWORK_TODO' not in ast.unparse(node)
    ns = {'torch': torch, 'F': F, 'copy': copy, 'configclass': configclass,
          'ObsGroup': object, 'ObsTerm': Term, 'mdp': MDP(),
          'SceneEntityCfg': lambda name: NS(name=name),
          'Unoise': lambda **kw: NS(**kw)}
    # Compile postponed annotations so no simulator types are required.
    ns['__builtins__'] = __builtins__
    extract(NAV / 'mdp/observations.py',
            ['last_high_level_command', 'height_scan_pooled', '_height_scan_grid_shape'], ns)
    env = NS(num_envs=2, device='cpu')
    scan = torch.arange(2 * 26 * 42, dtype=torch.float32).reshape(2, -1)
    ns['_height_scan'] = lambda env, sensor_cfg, offset: scan - offset
    env.scene = NS(sensors={'height_scanner': NS(cfg=NS(pattern_cfg=NS(size=(5., 3.), resolution=.12)))})
    pooled = ns['height_scan_pooled'](env, NS(name='height_scanner'), offset=.5)
    expected = torch.stack([torch.stack([
        (scan[b].reshape(26, 42)[y:y+2, x:x+2] - .5).max()
        for y in range(0, 26, 2) for x in range(0, 42, 2)]) for b in range(2)])
    torch.testing.assert_close(pooled, expected)
    assert pooled.shape == (2, 273)

    extract(TASKS / 'locomotion/robots/g1/29dof/low_level_env_cfg.py', ['LowLevelObservationsCfg'], ns)
    training = ns['LowLevelObservationsCfg'].PolicyCfg()
    ns['LOW_LEVEL_ENV_CFG'] = NS(observations=NS(policy=training))
    cfg_path = NAV / 'robots/g1/29dof/navigation_env_cfg.py'
    extract(cfg_path, ['make_low_level_inference_observations', 'NavigationObservationsCfg',
                      'NavigationV5ObservationsCfg', 'NavigationV5CompactObservationsCfg'], ns)
    low = ns['make_low_level_inference_observations']()
    assert low.history_length == 5 and low.concatenate_terms and not low.enable_corruption
    for name in ('base_ang_vel', 'projected_gravity', 'joint_pos_rel', 'joint_vel_rel'):
        assert getattr(low, name).noise is None
        assert getattr(training, name).noise is not None
    assert training.enable_corruption
    high = ns['NavigationV5CompactObservationsCfg'].PolicyCfg()
    assert high.height_scan is None
    terms = {k: v for k, v in vars(high).items() if isinstance(v, Term)}
    assert set(terms) == {'base_lin_vel', 'projected_gravity', 'base_ang_vel', 'pose_command',
                          'last_cmd', 'joint_pos_rel', 'joint_vel_rel', 'low_level_last_action',
                          'height_scan_pooled'}
    assert sum(273 if k == 'height_scan_pooled' else v.func(env, **v.params).shape[1]
               for k, v in terms.items()) == 376
    assert high.base_ang_vel.scale == .2 and high.joint_vel_rel.scale == .05

    class ActionBase:
        def __init__(self, cfg, env):
            self.cfg, self.num_envs, self.device = cfg, env.num_envs, env.device

    class JointAction:
        action_dim = 29
        def __init__(self, cfg, env):
            self.processed, self.applied = 0, 0
        def process_actions(self, actions):
            assert actions.shape == (2, 29) and torch.isfinite(actions).all()
            self.processed += 1
        def apply_actions(self):
            self.applied += 1
        def reset(self, ids):
            pass

    class ObsManager:
        def __init__(self, cfg, env):
            self.group, self.env, self.calls = cfg['ll_policy'], env, 0
        def compute_group(self, name):
            assert name == 'll_policy'
            self.calls += 1
            cmd = self.group.velocity_commands.func(self.env, **self.group.velocity_commands.params)
            last = self.group.last_action.func(self.env, **self.group.last_action.params)
            assert cmd.shape == (2, 3) and last.shape == (2, 29)
            # Synthetic five-frame input; real history semantics require Isaac Lab.
            return torch.cat((torch.zeros(2, 6), cmd, torch.zeros(2, 58), last), dim=1).repeat(1, 5)
        def reset(self, ids):
            pass

    ns.update(ActionTerm=ActionBase, ObservationManager=ObsManager,
              check_file_path=lambda p: Path(p).is_file(),
              read_file=lambda p: io.BytesIO(Path(p).read_bytes()))
    extract(NAV / 'mdp/pre_trained_policy_action.py', ['PreTrainedPolicyAction'], ns)
    env.scene = {'robot': NS()}
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    cfg = NS(asset_name='robot', policy_path=str(ROOT / 'pretrained/g1_29dof_lowlevel/policy.pt'),
             low_level_actions=NS(class_type=JointAction), low_level_observations=low,
             velocity_clip=((-0.5, 1.), (-0.5, .5), (-.5, .5)), low_level_decimation=4)
    action = ns['PreTrainedPolicyAction'](cfg, env)
    assert not action._policy.training
    assert all(not p.requires_grad for p in action._policy.parameters())
    env.action_manager = NS(get_term=lambda name: action)
    inputs = torch.tensor([[2., -2., 1.], [-1., .25, -.1]])
    original = inputs.clone()
    action.process_actions(inputs)
    torch.testing.assert_close(inputs, original)
    torch.testing.assert_close(action.raw_actions, original)
    torch.testing.assert_close(action.processed_actions, torch.tensor([[1., -.5, .5], [-.5, .25, -.1]]))
    assert ns['last_high_level_command'](env) is action.processed_actions
    inputs.zero_()
    torch.testing.assert_close(action.raw_actions, original)
    # Confirm the actual policy call is under inference mode, using the real checkpoint.
    actual_policy = action._policy
    def checked_policy(obs):
        assert torch.is_inference_mode_enabled()
        assert obs.shape == (2, 480)
        return actual_policy(obs)
    action._policy = checked_policy
    for _ in range(40):
        action.apply_actions()
    assert action._low_level_obs_manager.calls == 10
    assert action._low_level_action_term.processed == 10
    assert action._low_level_action_term.applied == 40
    assert not action.low_level_actions.requires_grad
    before = action.low_level_actions[1].clone()
    action.reset([0])
    assert action.low_level_actions[0].count_nonzero() == 0
    torch.testing.assert_close(action.low_level_actions[1], before)
    action.apply_actions()  # Cached tensor must stay writable outside inference mode.
    action.reset()
    assert action.raw_actions.count_nonzero() == action.processed_actions.count_nonzero() == 0
    assert action.low_level_actions.count_nonzero() == 0
    print('PASS: Python syntax; 7 TODOs; 273 scan / 376 actor dimensions; clean 5-frame config')
    print('PASS: real TorchScript CPU 480 -> 29; frozen inference; clipping; 10 updates / 40 steps; reset')
    print('LIMIT: Isaac Lab managers, sensors, assets, CUDA and PPO update require Linux smoke.')


if __name__ == '__main__':
    main()
