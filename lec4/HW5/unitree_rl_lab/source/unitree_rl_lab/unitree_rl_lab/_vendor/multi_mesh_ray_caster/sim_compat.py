# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility shim for ``isaaclab.sim`` utilities missing from IsaacLab 2.3.1.

The functions below are back-ported verbatim (modulo docstring trimming) from the IsaacLab
``main`` branch (``source/isaaclab/isaaclab/sim/utils/{stage,transforms}.py``), because the
installed IsaacLab 2.3.1 does not provide them. They are used by the vendored
MultiMeshRayCaster sensor modules.
"""

from __future__ import annotations

import logging

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

logger = logging.getLogger(__name__)

_INVALID_XFORM_OPS = [
    "xformOp:rotateX",
    "xformOp:rotateXZY",
    "xformOp:rotateY",
    "xformOp:rotateYXZ",
    "xformOp:rotateYZX",
    "xformOp:rotateZ",
    "xformOp:rotateZYX",
    "xformOp:rotateZXY",
    "xformOp:rotateXYZ",
    "xformOp:transform",
]
"""List of invalid xform ops that should be removed."""


def get_current_stage(fabric: bool = False) -> Usd.Stage:
    """Get the current open USD or Fabric stage.

    Back-ported from IsaacLab main's ``isaaclab.sim.utils.stage.get_current_stage``,
    without the per-thread stage context override (which 2.3.1 does not have).
    """
    stage = omni.usd.get_context().get_stage()

    if fabric:
        import usdrt

        from isaaclab.sim import get_current_stage_id

        stage_id = get_current_stage_id()
        return usdrt.Usd.Stage.Attach(stage_id)

    return stage


def update_stage() -> None:
    """Updates the current stage by triggering an application update cycle."""
    omni.kit.app.get_app_interface().update()


def standardize_xform_ops(
    prim: Usd.Prim,
    translation: tuple[float, ...] | None = None,
    orientation: tuple[float, ...] | None = None,
    scale: tuple[float, ...] | None = None,
) -> bool:
    """Standardize the transform operation stack on a USD prim to [translate, orient, scale].

    Back-ported verbatim from IsaacLab main's ``isaaclab.sim.utils.transforms.standardize_xform_ops``.
    """
    # Validate prim
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath()}' is not valid.")

    # Check if prim is an Xformable
    if not prim.IsA(UsdGeom.Xformable):
        logger.error(
            f"Prim at path '{prim.GetPath().pathString}' is of type '{prim.GetTypeName()}', "
            "which is not an Xformable. Transform operations will not be standardized. "
            "This is expected for material, shader, and scope prims."
        )
        return False

    # Create xformable interface
    xformable = UsdGeom.Xformable(prim)
    # Get current property names
    prop_names = prim.GetPropertyNames()

    # Obtain current local transformations
    tf = Gf.Transform(xformable.GetLocalTransformation())
    xform_pos = Gf.Vec3d(tf.GetTranslation())
    xform_quat = Gf.Quatd(tf.GetRotation().GetQuat())
    xform_scale = Gf.Vec3d(tf.GetScale())

    if translation is not None:
        xform_pos = Gf.Vec3d(*translation)
    if orientation is not None:
        xform_quat = Gf.Quatd(*orientation)

    # Handle scale resolution
    if scale is not None:
        xform_scale = Gf.Vec3d(scale)
    elif "xformOp:scale" in prop_names:
        if "xformOp:scale:unitsResolve" in prop_names:
            units_resolve = prim.GetAttribute("xformOp:scale:unitsResolve").Get()
            for i in range(3):
                xform_scale[i] = xform_scale[i] * units_resolve[i]
    else:
        xform_scale = Gf.Vec3d(1.0, 1.0, 1.0)

    # Verify if xform stack is reset
    has_reset = xformable.GetResetXformStack()
    # Batch the operations
    with Sdf.ChangeBlock():
        # Clear the existing transform operation order
        for prop_name in prop_names:
            if prop_name in _INVALID_XFORM_OPS:
                prim.RemoveProperty(prop_name)

        # Remove unitsResolve attribute if present (already handled in scale resolution above)
        if "xformOp:scale:unitsResolve" in prop_names:
            prim.RemoveProperty("xformOp:scale:unitsResolve")

        # Set up or retrieve scale operation
        xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))
        if not xform_op_scale:
            xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")

        # Set up or retrieve translate operation
        xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
        if not xform_op_translate:
            xform_op_translate = xformable.AddXformOp(
                UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
            )

        # Set up or retrieve orient (quaternion rotation) operation
        xform_op_orient = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
        if not xform_op_orient:
            xform_op_orient = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")

        # Handle different floating point precisions
        xform_ops = [xform_op_translate, xform_op_orient, xform_op_scale]
        xform_values = [xform_pos, xform_quat, xform_scale]
        for xform_op, value in zip(xform_ops, xform_values):
            current_value = xform_op.Get()
            xform_op.Set(type(current_value)(value) if current_value is not None else value)

        # Set the transform operation order: translate -> orient -> scale
        xformable.SetXformOpOrder([xform_op_translate, xform_op_orient, xform_op_scale], has_reset)

    return True


def validate_standard_xform_ops(prim: Usd.Prim) -> bool:
    """Validate if the transform operations on a prim are standardized to [translate, orient, scale].

    Back-ported verbatim from IsaacLab main's ``isaaclab.sim.utils.transforms.validate_standard_xform_ops``.
    """
    # check if prim is valid
    if not prim.IsValid():
        logger.error(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
        return False
    # check if prim is an xformable
    if not prim.IsA(UsdGeom.Xformable):
        logger.error(f"Prim at path '{prim.GetPath().pathString}' is not an xformable.")
        return False
    # get the xformable interface
    xformable = UsdGeom.Xformable(prim)
    # get the xform operation order
    xform_op_order = xformable.GetOrderedXformOps()
    xform_op_order = [op.GetOpName() for op in xform_op_order]
    # check if the xform operation order is the canonical form
    if xform_op_order != ["xformOp:translate", "xformOp:orient", "xformOp:scale"]:
        msg = f"Xform operation order for prim at path '{prim.GetPath().pathString}' is not the canonical form."
        msg += f" Received order: {xform_op_order}"
        msg += " Expected order: ['xformOp:translate', 'xformOp:orient', 'xformOp:scale']"
        logger.error(msg)
        return False
    return True


def resolve_prim_pose(
    prim: Usd.Prim, ref_prim: Usd.Prim | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Resolve the pose of a prim with respect to another prim (or the world frame).

    Back-ported verbatim from IsaacLab main's ``isaaclab.sim.utils.transforms.resolve_prim_pose``.
    Returns the position and the quaternion orientation in (w, x, y, z) format.
    """
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    # get prim xform
    xform = UsdGeom.Xformable(prim)
    prim_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # sanitize quaternion
    prim_tf.Orthonormalize()

    if ref_prim is not None:
        # if reference prim is the root, we can skip the computation
        if ref_prim.GetPath() != Sdf.Path.absoluteRootPath:
            ref_xform = UsdGeom.Xformable(ref_prim)
            ref_tf = ref_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ref_tf.Orthonormalize()
            prim_tf = prim_tf * ref_tf.GetInverse()

    # extract position and orientation
    prim_pos = [*prim_tf.ExtractTranslation()]
    prim_quat = [prim_tf.ExtractRotationQuat().real, *prim_tf.ExtractRotationQuat().imaginary]
    return tuple(prim_pos), tuple(prim_quat)


def resolve_prim_scale(prim: Usd.Prim) -> tuple[float, float, float]:
    """Resolve the scale of a prim in the world frame.

    Back-ported verbatim from IsaacLab main's ``isaaclab.sim.utils.transforms.resolve_prim_scale``.
    """
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    # compute local to world transform
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # extract scale
    return tuple([*(v.GetLength() for v in world_transform.ExtractRotationMatrix())])
