"""Fuse one ideal front/rear fisheye range pair into an observed height field.

Depth is radial range, not camera-axis Z. The cameras use the same nominal poses
and equidistant rays as render_fixed_cameras.py. This is simulated depth, not
stereo or monocular depth estimation from the physical cameras.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from render_fixed_cameras import demonstration_scene, equidistant_rays


def simulate_depth(model, data, camera_name, size=640, fov_degrees=200,
                   min_range=0.02, max_range=8.0):
  """First visible surface on each exact pixel ray, including robot occlusion.

  Invalid/out-of-range pixels have NaN range and geom ID -1. Collision proxies
  in group 3 are excluded, matching the RGB example's visible geometry. Meshes
  are ray-tested as triangles, not their collision convex hulls.
  """
  if not 0 <= min_range < max_range:
    raise ValueError("Expected 0 <= min_range < max_range")
  rays, valid = equidistant_rays(size, fov_degrees)
  camera = data.camera(camera_name)
  world_rays = np.ascontiguousarray(rays[valid] @ camera.xmat.reshape(3, 3).T)
  ranges = np.full(len(world_rays), -1.0)
  ids = np.full(len(world_rays), -1, dtype=np.int32)
  groups = np.ones(6, dtype=np.uint8)
  groups[3] = 0
  mujoco.mj_multiRay(model, data, camera.xpos, world_rays.ravel(), groups,
    True, -1, ids, ranges, None, len(world_rays), max_range)
  # The API cutoff culls geometry bounds; explicitly enforce the sensor range.
  hits = (ids >= 0) & (ranges >= min_range) & (ranges <= max_range)
  depth = np.full((size, size), np.nan)
  geom_ids = np.full((size, size), -1, dtype=np.int32)
  depth[valid] = np.where(hits, ranges, np.nan)
  geom_ids[valid] = np.where(hits, ids, -1)
  return depth, geom_ids


def unproject_depth(depth, origin, rotation, fov_degrees=200):
  """Backproject radial meters to world XYZ, preserving NaNs for missing rays."""
  if depth.ndim != 2 or depth.shape[0] != depth.shape[1]:
    raise ValueError("Expected a square circular-fisheye range image")
  rays, valid = equidistant_rays(depth.shape[0], fov_degrees)
  points = origin + (rays * depth[:, :, None]) @ rotation.T
  points[~valid] = np.nan
  return points


def robot_geometries(model, body_name="pelvis"):
  """Identify robot returns AFTER casting so the map cannot see through it."""
  root = model.body(body_name).id
  bodies = np.zeros(model.nbody, dtype=bool)
  for body in range(1, model.nbody):
    bodies[body] = body == root or bodies[model.body_parentid[body]]
  return bodies[model.geom_bodyid]


def height_field(points, half_extent=2.5, resolution=0.05, z_limits=(-0.05, 1.5)):
  """Highest observed Z per XY cell, not terrain truth or a traversability map.

  Axes are [forward X row, left Y column], meters, half-open XY bounds.
  Unobserved cells stay NaN. No interpolation, ray carving or ground-truth fill.
  Multiple layers/overhangs collapse to their highest *observed* surface.
  """
  cells = 2 * half_extent / resolution if resolution > 0 else 0
  if half_extent <= 0 or cells < 1 or not np.isclose(cells, round(cells)):
    raise ValueError("Grid extent must contain an integer number of positive-sized cells")
  if z_limits[0] >= z_limits[1]:
    raise ValueError("Expected increasing height limits")
  size = round(cells)
  points = np.asarray(points).reshape(-1, 3)
  keep = np.all(np.isfinite(points), axis=1)
  keep &= np.all((points[:, :2] >= -half_extent) & (points[:, :2] < half_extent), axis=1)
  keep &= (points[:, 2] >= z_limits[0]) & (points[:, 2] <= z_limits[1])
  points = points[keep]
  ij = np.floor((points[:, :2] + half_extent) / resolution).astype(int)
  # Guard floating-point roundoff at the upper half-open edge, not out-of-bounds points.
  ij = np.minimum(ij, size - 1)
  heights = np.full((size, size), -np.inf)
  counts = np.zeros((size, size), dtype=np.int32)
  np.maximum.at(heights, (ij[:, 0], ij[:, 1]), points[:, 2])
  np.add.at(counts, (ij[:, 0], ij[:, 1]), 1)
  heights[counts == 0] = np.nan
  return heights, counts


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--size", type=int, default=640)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  started = time.perf_counter()
  robot = "dollhouse_g1_27dof_with_camera_mount"
  model, data = demonstration_scene(robot)
  robot_geoms = robot_geometries(model)
  # A gravity-aligned map centered under the torso. The demo's floor is world Z=0.
  origin = data.body("torso_link").xpos.copy()
  origin[2] = 0
  arrays = {"map_origin_world_m": origin}
  grid = {"half_extent": 2.5, "resolution": 0.05, "z_limits": (-0.05, 1.5)}
  sensor = {"size": args.size, "fov_degrees": 200, "min_range": 0.02, "max_range": 8}
  report = {
    "experiment": "g1-dual-fisheye-depth-heightfield",
    "robot": robot, "mujoco": mujoco.__version__, "cameras": {},
    "sensor": {"projection": "ideal equidistant", "circular_fov_degrees": sensor["fov_degrees"],
      "image_size": [args.size, args.size], "depth": "radial range in meters",
      "range_limits_m": [sensor["min_range"], sensor["max_range"]],
      "noise": "none", "frames_per_camera": 1},
    "grid": {"resolution_m": grid["resolution"], "half_extent_m": grid["half_extent"],
      "height_limits_m": grid["z_limits"], "origin_world_m": origin.tolist(),
      "axes": "gravity-aligned +X forward, +Y left, Z above the known demo floor",
      "reduction": "maximum observed height", "unobserved": "NaN; never filled"},
    "limits": ["Nominal camera placement, not physically calibrated.",
      "Ideal simulated depth, not an assertion that the physical cameras measure depth.",
      "Robot occludes rays; perfect simulator geometry IDs remove its first returns afterward.",
      "One static pair, known pose/floor, no temporal fusion, noise, stereo or learned depth.",
      "Visible obstacle surfaces, not walkable ground; overhangs cannot be represented faithfully.",
      "Only returns between -0.05 and 1.5 m above the floor enter the map; ceiling excluded."]}
  all_points = []
  maps = []
  for name in ("fisheye_front", "fisheye_back"):
    tick = time.perf_counter()
    depth, ids = simulate_depth(model, data, name, **sensor)
    points = unproject_depth(depth, data.camera(name).xpos,
      data.camera(name).xmat.reshape(3, 3), sensor["fov_degrees"])
    hit = ids >= 0
    is_robot = hit & robot_geoms[np.maximum(ids, 0)]
    environment = hit & ~is_robot
    env_points = points[environment] - origin
    heights, counts = height_field(env_points, **grid)
    all_points.append(env_points)
    maps.append(heights)
    arrays.update({f"{name}_range_m": depth, f"{name}_geom_id": ids,
      f"{name}_environment_points_m": env_points, f"{name}_height_m": heights,
      f"{name}_count": counts, f"{name}_robot_mask": is_robot,
      f"{name}_origin_world_m": data.camera(name).xpos.copy(),
      f"{name}_rotation_world": data.camera(name).xmat.reshape(3, 3).copy()})
    report["cameras"][name] = {"valid_range_pixels": int(hit.sum()),
      "robot_return_pixels": int(is_robot.sum()),
      "environment_return_pixels": int(environment.sum()),
      "mapped_points": int(counts.sum()), "observed_cells": int(np.isfinite(heights).sum()),
      "coverage_fraction": float(np.isfinite(heights).mean()),
      "simulation_seconds": time.perf_counter() - tick}
    print(name, json.dumps(report["cameras"][name]), flush=True)
  combined, counts = height_field(np.concatenate(all_points), **grid)
  report["grid"]["shape"] = list(combined.shape)
  arrays.update(height_m=combined, count=counts)
  front_seen, rear_seen = [np.isfinite(h) for h in maps]
  arrays["source_mask"] = front_seen.astype(np.uint8) + 2 * rear_seen.astype(np.uint8)
  report["fusion"] = {"observed_cells": int(np.isfinite(combined).sum()),
    "unknown_cells": int(np.isnan(combined).sum()),
    "coverage_fraction": float(np.isfinite(combined).mean()),
    "front_only_cells": int((front_seen & ~rear_seen).sum()),
    "rear_only_cells": int((rear_seen & ~front_seen).sum()),
    "overlap_cells": int((rear_seen & front_seen).sum())}
  centers = -grid["half_extent"] + (np.arange(len(combined)) + 0.5) * grid["resolution"]
  distance = np.hypot(centers[:, None], centers[None, :])
  report["near_robot_coverage"] = []
  for radius in (0.3, 0.5, 1.0):
    inside = distance < radius
    seen = np.isfinite(combined[inside])
    report["near_robot_coverage"].append({"radius_m": radius,
      "cell_center_denominator": int(inside.sum()), "observed_cells": int(seen.sum()),
      "coverage_fraction": float(seen.mean())})
  report["duration_seconds"] = time.perf_counter() - started
  report["observed_at"] = datetime.now(timezone.utc).isoformat()
  np.savez_compressed(args.output / "depth-heightfield.npz", **arrays)
  (args.output / "experiment.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
