# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapping around warp kernels for compatibility with torch tensors."""

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import logging
import os

import numpy as np
import torch
import warp as wp

# disable warp module initialization messages
wp.config.quiet = True
# initialize the warp module
wp.init()

from isaaclab.utils.math import convert_quat

from . import kernels

logger = logging.getLogger(__name__)

# Set MMC_RAYCAST_NO_SYNC_CHECK=1 to skip the per-call finite check (a device sync) and rely on
# the purely GPU-side nan_to_num-style replacement below.
_SYNC_CHECK = os.environ.get("MMC_RAYCAST_NO_SYNC_CHECK", "0") != "1"
_sanitize_warned: set[str] = set()


def _sanitize_finite(name: str, tensor: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Replace non-finite values in ``tensor`` with ``fallback`` (broadcastable).

    Defense against device-side hangs: ``wp.mesh_query_ray`` can loop forever inside the BVH
    traversal when a ray start/direction (or a mesh transform used to compute the local ray)
    contains NaN/Inf. Replacing non-finite inputs keeps the kernel finite; the returned hits
    for those rays are garbage either way, but training then crashes on a readable NaN error
    downstream instead of hanging the GPU.

    The finite check is done on the host (a device sync) on purpose: besides validating the
    inputs, the periodic host-side pacing works around a timing-sensitive livelock of PhysX's
    ``convexCoreTrimeshNphase_Kernel32`` observed with this sensor stack (Isaac Sim 5.1
    PhysX 107.3 + warp 1.16, ~165k kinematic obstacle bodies at 1024 envs), where training
    otherwise deadlocks inside PhysX ``simulate/fetch_results`` with the GPU spinning at 100%
    (deterministically at the first rollout after the second PPO update, with bit-identical
    trajectories). Throughput is unaffected because the loop is GPU-bound.
    """
    if _SYNC_CHECK:
        num_bad = int((~torch.isfinite(tensor)).sum())
        if num_bad > 0 and name not in _sanitize_warned:
            _sanitize_warned.add(name)
            logger.warning(
                f"raycast input '{name}' contains {num_bad} non-finite values (shape {tuple(tensor.shape)}). "
                "Replacing them with a finite fallback to avoid a device-side hang in wp.mesh_query_ray. "
                "The NaN/Inf originates upstream (policy / physics state)."
            )
    return torch.where(torch.isfinite(tensor), tensor, fallback)



def _current_torch_stream(torch_device) -> "wp.Stream | None":
    """Return a warp stream aliasing torch's current stream for the device (None on CPU).

    Launching the raycast kernels on torch's current stream guarantees they are ordered after
    the torch kernels that produced the ray/mesh buffers. Without this, warp launches on its own
    default stream and can read the buffers before torch has finished writing them; the torn
    values can send ``wp.mesh_query_ray``'s BVH traversal into an endless loop, hanging the GPU.
    """
    if torch.device(torch_device).type == "cuda":
        return wp.stream_from_torch(torch.cuda.current_stream(torch_device))
    return None


def raycast_mesh(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    mesh: wp.Mesh,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against a mesh.

    Note that the `ray_starts` and `ray_directions`, and `ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    Args:
        ray_starts: The starting position of the rays. Shape (N, 3).
        ray_directions: The ray directions for each ray. Shape (N, 3).
        mesh: The warp mesh to ray-cast against.
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.

    Returns:
        The ray hit position. Shape (N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
    """
    # extract device and shape information
    shape = ray_starts.shape
    device = ray_starts.device
    # device of the mesh
    torch_device = wp.device_to_torch(mesh.device)
    # reshape the tensors
    ray_starts = ray_starts.to(torch_device).view(-1, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).view(-1, 3).contiguous()
    # sanitize non-finite inputs to avoid a device-side hang in wp.mesh_query_ray
    zero = torch.zeros(3, dtype=ray_starts.dtype, device=torch_device)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=ray_starts.dtype, device=torch_device)
    ray_starts = _sanitize_finite("raycast_mesh.ray_starts", ray_starts, zero)
    ray_directions = _sanitize_finite("raycast_mesh.ray_directions", ray_directions, down)
    num_rays = ray_starts.shape[0]

    # create output tensor for the ray hits
    ray_hits = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()

    # map the memory to warp arrays
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

    if return_distance:
        ray_distance = torch.full((num_rays,), float("inf"), device=torch_device).contiguous()
        ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)
    else:
        ray_distance = None
        ray_distance_wp = wp.empty((1,), dtype=wp.float32, device=torch_device)

    if return_normal:
        ray_normal = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1,), dtype=wp.vec3, device=torch_device)

    if return_face_id:
        ray_face_id = torch.ones((num_rays,), dtype=torch.int32, device=torch_device).contiguous() * (-1)
        ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)
    else:
        ray_face_id = None
        ray_face_id_wp = wp.empty((1,), dtype=wp.int32, device=torch_device)

    # launch the warp kernel
    wp.launch(
        kernel=kernels.raycast_mesh_kernel,
        dim=num_rays,
        inputs=[
            mesh.id,
            ray_starts_wp,
            ray_directions_wp,
            ray_hits_wp,
            ray_distance_wp,
            ray_normal_wp,
            ray_face_id_wp,
            float(max_dist),
            int(return_distance),
            int(return_normal),
            int(return_face_id),
        ],
        device=mesh.device,
        stream=_current_torch_stream(torch_device),
    )
    # NOTE: Synchronize is not needed anymore, but we keep it for now. Check with @dhoeller.
    wp.synchronize()

    if return_distance:
        ray_distance = ray_distance.to(device).view(shape[0], shape[1])
    if return_normal:
        ray_normal = ray_normal.to(device).view(shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).view(shape[0], shape[1])

    return ray_hits.to(device).view(shape), ray_distance, ray_normal, ray_face_id


def raycast_single_mesh(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    mesh_id: int,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against a mesh.

    Note that the :attr:`ray_starts` and :attr:`ray_directions`, and :attr:`ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    Args:
        ray_starts: The starting position of the rays. Shape (B, N, 3).
        ray_directions: The ray directions for each ray. Shape (B, N, 3).
        mesh_id: The warp mesh id to ray-cast against.
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.

    Returns:
        The ray hit position. Shape (B, N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (B, N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (B, N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (B, N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
    """
    # cast mesh id into array
    mesh_ids = wp.array2d(
        [[mesh_id] for _ in range(ray_starts.shape[0])], dtype=wp.uint64, device=wp.device_from_torch(ray_starts.device)
    )
    ray_hits, ray_distance, ray_normal, ray_face_id, ray_mesh_id = raycast_dynamic_meshes(
        ray_starts=ray_starts,
        ray_directions=ray_directions,
        mesh_ids_wp=mesh_ids,
        max_dist=max_dist,
        return_distance=return_distance,
        return_normal=return_normal,
        return_face_id=return_face_id,
    )

    return ray_hits, ray_distance, ray_normal, ray_face_id


def raycast_dynamic_meshes(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    mesh_ids_wp: wp.Array,
    mesh_positions_w: torch.Tensor | None = None,
    mesh_orientations_w: torch.Tensor | None = None,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
    return_mesh_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against multiple, dynamic meshes.

    Note that the :attr:`ray_starts` and :attr:`ray_directions`, and :attr:`ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    If mesh positions and rotations are provided, they need to have to have the same shape as the
    number of meshes.

    Args:
        ray_starts: The starting position of the rays. Shape (B, N, 3).
        ray_directions: The ray directions for each ray. Shape (B, N, 3).
        mesh_ids_wp: The warp mesh ids to ray-cast against. Length (B, M).
        mesh_positions_w: The world positions of the meshes. Shape (B, M, 3).
        mesh_orientations_w: The world orientation as quaternion (wxyz) format. Shape (B, M, 4).
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.
        return_mesh_id: Whether to return the mesh id of the mesh face the ray hits. Defaults to False.
                        NOTE: the type of the returned tensor is torch.int16, so you can't have more than 32767 meshes.

    Returns:
        The ray hit position. Shape (B, N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (B, N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (B, N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (B, N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
        The ray hit mesh id. Shape (B, N,).
            Will only return if :attr:`return_mesh_id` is True else returns None.
            The returned tensor contains :obj:`-1` for missed hits.
    """
    # extract device and shape information
    shape = ray_starts.shape
    device = ray_starts.device

    # device of the mesh
    torch_device = wp.device_to_torch(mesh_ids_wp.device)
    n_meshes = mesh_ids_wp.shape[1]

    n_envs = ray_starts.shape[0]
    n_rays_per_env = ray_starts.shape[1]

    # reshape the tensors
    ray_starts = ray_starts.to(torch_device).view(n_envs, n_rays_per_env, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).view(n_envs, n_rays_per_env, 3).contiguous()
    # sanitize non-finite inputs to avoid a device-side hang in wp.mesh_query_ray
    zero = torch.zeros(3, dtype=ray_starts.dtype, device=torch_device)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=ray_starts.dtype, device=torch_device)
    ray_starts = _sanitize_finite("raycast_dynamic_meshes.ray_starts", ray_starts, zero)
    ray_directions = _sanitize_finite("raycast_dynamic_meshes.ray_directions", ray_directions, down)


    # create output tensor for the ray hits
    ray_hits = torch.full((n_envs, n_rays_per_env, 3), float("inf"), device=torch_device).contiguous()

    # map the memory to warp arrays
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)
    # required to check if a closer hit is reported, returned only if return_distance is true
    ray_distance = torch.full(
        (
            n_envs,
            n_rays_per_env,
        ),
        float("inf"),
        device=torch_device,
    ).contiguous()
    ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)

    if return_normal:
        ray_normal = torch.full((n_envs, n_rays_per_env, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1, 1), dtype=wp.vec3, device=torch_device)

    if return_face_id:
        ray_face_id = torch.ones(
            (
                n_envs,
                n_rays_per_env,
            ),
            dtype=torch.int32,
            device=torch_device,
        ).contiguous() * (-1)
        ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)
    else:
        ray_face_id = None
        ray_face_id_wp = wp.empty((1, 1), dtype=wp.int32, device=torch_device)

    if return_mesh_id:
        ray_mesh_id = -torch.ones((n_envs, n_rays_per_env), dtype=torch.int16, device=torch_device).contiguous()
        ray_mesh_id_wp = wp.from_torch(ray_mesh_id, dtype=wp.int16)
    else:
        ray_mesh_id = None
        ray_mesh_id_wp = wp.empty((1, 1), dtype=wp.int16, device=torch_device)

    ##
    # Call the warp kernels
    ###
    if mesh_positions_w is None and mesh_orientations_w is None:
        # Static mesh case, no need to pass in positions and rotations.
        # launch the warp kernel
        wp.launch(
            kernel=kernels.raycast_static_meshes_kernel,
            dim=[n_meshes, n_envs, n_rays_per_env],
            inputs=[
                mesh_ids_wp,
                ray_starts_wp,
                ray_directions_wp,
                ray_hits_wp,
                ray_distance_wp,
                ray_normal_wp,
                ray_face_id_wp,
                ray_mesh_id_wp,
                float(max_dist),
                int(return_normal),
                int(return_face_id),
                int(return_mesh_id),
            ],
            device=torch_device,
            stream=_current_torch_stream(torch_device),
        )
    else:
        # dynamic mesh case
        if mesh_positions_w is None:
            mesh_positions_wp_w = wp.zeros((n_envs, n_meshes), dtype=wp.vec3, device=torch_device)
        else:
            mesh_positions_w = mesh_positions_w.to(torch_device).view(n_envs, n_meshes, 3).contiguous()
            mesh_positions_w = _sanitize_finite("raycast_dynamic_meshes.mesh_positions_w", mesh_positions_w, zero)
            mesh_positions_wp_w = wp.from_torch(mesh_positions_w, dtype=wp.vec3)

        if mesh_orientations_w is None:
            # Note (zrene): This is a little bit ugly, since it requires to initialize torch memory first
            # But I couldn't find a better way to initialize a quaternion identity in warp
            # wp.zeros(1, dtype=wp.quat, device=torch_device) gives all zero quaternion
            quat_identity = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=torch_device).repeat(
                n_envs, n_meshes, 1
            )
            mesh_quat_wp_w = wp.from_torch(quat_identity, dtype=wp.quat)
        else:
            mesh_orientations_w = mesh_orientations_w.to(dtype=torch.float32, device=torch_device)
            identity_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=mesh_orientations_w.dtype, device=torch_device)
            mesh_orientations_w = _sanitize_finite(
                "raycast_dynamic_meshes.mesh_orientations_w", mesh_orientations_w, identity_wxyz
            )
            mesh_orientations_w = convert_quat(mesh_orientations_w, "xyzw").contiguous()
            mesh_quat_wp_w = wp.from_torch(mesh_orientations_w, dtype=wp.quat)

        # launch the warp kernel
        wp.launch(
            kernel=kernels.raycast_dynamic_meshes_kernel,
            dim=[n_meshes, n_envs, n_rays_per_env],
            inputs=[
                mesh_ids_wp,
                ray_starts_wp,
                ray_directions_wp,
                ray_hits_wp,
                ray_distance_wp,
                ray_normal_wp,
                ray_face_id_wp,
                ray_mesh_id_wp,
                mesh_positions_wp_w,
                mesh_quat_wp_w,
                float(max_dist),
                int(return_normal),
                int(return_face_id),
                int(return_mesh_id),
            ],
            device=torch_device,
            stream=_current_torch_stream(torch_device),
        )
    ##
    # Cleanup and convert back to torch tensors
    ##

    # NOTE: Synchronize is not needed anymore, but we keep it for now. Check with @dhoeller.
    wp.synchronize()

    if return_distance:
        ray_distance = ray_distance.to(device).view(shape[:2])
    if return_normal:
        ray_normal = ray_normal.to(device).view(shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).view(shape[:2])
    if return_mesh_id:
        ray_mesh_id = ray_mesh_id.to(device).view(shape[:2])

    return ray_hits.to(device).view(shape), ray_distance, ray_normal, ray_face_id, ray_mesh_id


def convert_to_warp_mesh(points: np.ndarray, indices: np.ndarray, device: str) -> wp.Mesh:
    """Create a warp mesh object with a mesh defined from vertices and triangles.

    Args:
        points: The vertices of the mesh. Shape is (N, 3), where N is the number of vertices.
        indices: The triangles of the mesh as references to vertices for each triangle.
            Shape is (M, 3), where M is the number of triangles / faces.
        device: The device to use for the mesh.

    Returns:
        The warp mesh object.
    """
    return wp.Mesh(
        points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(indices.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )
