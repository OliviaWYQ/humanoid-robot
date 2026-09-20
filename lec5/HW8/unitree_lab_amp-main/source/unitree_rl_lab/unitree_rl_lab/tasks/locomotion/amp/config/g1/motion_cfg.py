"""Readable G1 expert-motion selections shared by train and Play environments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ...motion_dataset import DEFAULT_FEATURES, MotionDatasetCfg


AMP_ROOT = Path(__file__).parents[2]
DATA_ROOT = Path(os.environ.get("UNITREE_AMP_MOTION_ROOT", AMP_ROOT / "data")).expanduser()

G1_REFERENCE_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

G1_AMP_KEY_LINK_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


@dataclass
class G1MotionSourceCfg:
    """A named directory and explicit sampling weights for one G1 motion style."""

    profile_name: str
    motion_dir: str
    clip_weights: dict[str, float] = field(default_factory=dict)
    file_pattern: str = "*.npz"
    features: tuple[str, ...] = DEFAULT_FEATURES

    def __post_init__(self) -> None:
        if not self.profile_name.strip():
            raise ValueError("G1 motion profile_name must be non-empty.")
        if not self.motion_dir:
            raise ValueError(f"G1 motion profile '{self.profile_name}' must define motion_dir.")
        if not self.clip_weights or sum(self.clip_weights.values()) <= 0.0:
            raise ValueError(f"G1 motion profile '{self.profile_name}' needs positive clip weights.")
        if any(weight < 0.0 for weight in self.clip_weights.values()):
            raise ValueError(f"G1 motion profile '{self.profile_name}' contains a negative clip weight.")

    def apply_to_dataset_cfg(self, dataset_cfg: MotionDatasetCfg) -> MotionDatasetCfg:
        dataset_cfg.motion_dir = self.motion_dir
        dataset_cfg.profile_name = self.profile_name
        dataset_cfg.clip_weights = dict(self.clip_weights)
        dataset_cfg.file_pattern = self.file_pattern
        return dataset_cfg


@dataclass
class G1WalkMotionCfg(G1MotionSourceCfg):
    profile_name: str = "walk"
    motion_dir: str = str(DATA_ROOT / "walk")
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "B3_-_walk1_stageii": 4.0,
        "B5_-_walk_backwards_stageii": 3.0,
        "B9_-_walk_turn_left_(90)_stageii": 1.0,
        "B10_-_walk_turn_left_(45)_stageii": 1.0,
        "B12_-_walk_turn_right_(90)_stageii": 1.0,
        "B13_-_walk_turn_right_(45)_stageii": 1.0,
    })


@dataclass
class G1RunMotionCfg(G1MotionSourceCfg):
    profile_name: str = "run"
    motion_dir: str = str(DATA_ROOT / "run")
    clip_weights: dict[str, float] = field(default_factory=lambda: {"C3_-_Run_stageii": 1.0})


@dataclass
class G1OmniRunMotionCfg(G1MotionSourceCfg):
    profile_name: str = "omni_run"
    motion_dir: str = str(DATA_ROOT / "run")
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "C3_-_Run_stageii": 6.0,
        "C11_-__run_turn_left_(90)_stageii": 1.5,
        "C14_-__run_turn_right__(90)_stageii": 1.0,
        "C15_-__run_turn_right__(45)_stageii": 0.5,
    })


@dataclass
class G1WalkToRunMotionCfg(G1MotionSourceCfg):
    """Task-agnostic walking, running, transition, reverse, and turning clips."""

    profile_name: str = "walk_to_run"
    motion_dir: str = str(DATA_ROOT / "mixed")
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "B3_-_walk1_stageii": 4.0,
        "B5_-_walk_backwards_stageii": 2.0,
        "B9_-_walk_turn_left_(90)_stageii": 1.0,
        "B12_-_walk_turn_right_(90)_stageii": 1.0,
        "C3_-_Run_stageii": 5.0,
        "C5_-_walk_to_run_stageii": 3.0,
        "C2_-_Run_to_stand_stageii": 1.0,
        "C6_-_stand_to_run_backwards_stageii": 1.0,
        "C11_-__run_turn_left_(90)_stageii": 1.0,
        "C14_-__run_turn_right__(90)_stageii": 1.0,
    })


@dataclass
class G1DanceMotionCfg(G1MotionSourceCfg):
    profile_name: str = "dance"
    motion_dir: str = str(DATA_ROOT / "dance")
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "irish_dance_stageii": 0.5,
        "salsa_1_stageii": 1.0,
        "salsa_stageii": 1.0,
    })


@dataclass
class G1MixedMotionCfg(G1MotionSourceCfg):
    profile_name: str = "mixed"
    motion_dir: str = os.environ.get("UNITREE_AMP_MOTION_DIR", str(DATA_ROOT / "mixed"))
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "B3_-_walk1_stageii": 2.0,
        "B5_-_walk_backwards_stageii": 1.0,
        "B9_-_walk_turn_left_(90)_stageii": 1.0,
        "B12_-_walk_turn_right_(90)_stageii": 1.0,
        "C11_-__run_turn_left_(90)_stageii": 1.0,
        "C14_-__run_turn_right__(90)_stageii": 1.0,
        "C2_-_Run_to_stand_stageii": 1.0,
        "C5_-_walk_to_run_stageii": 1.0,
    })


MOTION_CONFIGS = {
    "walk": G1WalkMotionCfg,
    "run": G1RunMotionCfg,
    "omni_run": G1OmniRunMotionCfg,
    "walk_to_run": G1WalkToRunMotionCfg,
    "dance": G1DanceMotionCfg,
    "mixed": G1MixedMotionCfg,
}


def make_motion_source(profile_name: str) -> G1MotionSourceCfg:
    try:
        return MOTION_CONFIGS[profile_name.strip().lower()]()
    except KeyError as error:
        choices = ", ".join(sorted(MOTION_CONFIGS))
        raise ValueError(f"Unknown G1 AMP motion profile '{profile_name}'. Available: {choices}.") from error


def make_motion_dataset_cfg(source: G1MotionSourceCfg, history_steps: int = 3) -> MotionDatasetCfg:
    return MotionDatasetCfg(
        motion_dir=source.motion_dir,
        joint_names=(),
        profile_name=source.profile_name,
        source_joint_names=G1_REFERENCE_JOINT_NAMES,
        key_link_names=G1_AMP_KEY_LINK_NAMES,
        history_steps=history_steps,
        step_dt=0.02,
        quaternion_order="xyzw",
        clip_weights=dict(source.clip_weights),
        file_pattern=source.file_pattern,
        features=source.features,
    )


# Compatibility names retained for existing student configs.
MotionSourceCfg = G1MotionSourceCfg
G1WalkMotionSourceCfg = G1WalkMotionCfg
G1RunMotionSourceCfg = G1RunMotionCfg
G1OmniRunMotionSourceCfg = G1OmniRunMotionCfg
G1WalkToRunMotionSourceCfg = G1WalkToRunMotionCfg
G1DanceMotionSourceCfg = G1DanceMotionCfg
G1MixedMotionSourceCfg = G1MixedMotionCfg
G1_walk_Cfg = G1WalkMotionCfg
G1_Run_Cfg = G1RunMotionCfg
G1_dance_Cfg = G1DanceMotionCfg
