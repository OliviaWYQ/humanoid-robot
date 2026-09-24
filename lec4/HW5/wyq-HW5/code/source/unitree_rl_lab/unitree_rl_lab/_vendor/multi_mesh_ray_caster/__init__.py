# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vendored MultiMeshRayCaster sensor back-ported from IsaacLab ``main``.

This sub-package mirrors ``isaaclab.sensors.ray_caster`` from the IsaacLab main branch,
adapted to run against a locally installed IsaacLab 2.3.1. Only the pieces missing from
IsaacLab 2.3.1 are vendored; everything else (math utilities, sensor base classes, pattern
configs, etc.) is imported from the installed ``isaaclab`` package.
"""

from .multi_mesh_ray_caster import MultiMeshRayCaster
from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg

__all__ = ["MultiMeshRayCaster", "MultiMeshRayCasterCfg"]
