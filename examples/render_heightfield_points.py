"""Show saved camera-derived heights as a dot grid around the actual G1 model.

This only changes the presentation of an existing fisheye_heightfield.py run.
It does not generate new observations or replace missing heights with the floor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from render_fixed_cameras import demonstration_scene, save_png


def render_grid(model, data, heights, origin, resolution, half_extent,
                view_half_extent, distance, output):
  centers = -half_extent + (np.arange(heights.shape[0]) + .5) * resolution
  x, y = np.meshgrid(centers, centers, indexing="ij")
  visible = np.isfinite(heights) & (np.abs(x) < view_half_extent) & (np.abs(y) < view_half_extent)
  points = np.column_stack((x[visible], y[visible], heights[visible])) + origin
  options = mujoco.MjvOption()
  options.geomgroup[0] = 0  # Hide the room; its observed heights remain as dots.
  options.geomgroup[3] = 0
  options.sitegroup[:] = 0
  camera = mujoco.MjvCamera()
  camera.lookat[:] = origin + [0, 0, .35]
  camera.distance = distance
  camera.azimuth = 225
  camera.elevation = -43
  with mujoco.Renderer(model, width=1400, height=1100, max_geom=len(points) + 500) as renderer:
    renderer.update_scene(data, camera=camera, scene_option=options)
    scene = renderer.scene
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    def add(kind, size, pos, color):
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(geom, kind, np.array(size), np.array(pos),
        np.eye(3).ravel(), np.array(color, dtype=np.float32))
      scene.ngeom += 1
      return geom

    # Neutral display backdrop only, 15 mm below the known demo floor to avoid
    # clipping the floor-level dots. Dot centers retain their recorded heights.
    add(mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, .1], origin + [0, 0, -.015], [.84, .87, .9, 1])
    for point in points:
      elevated = point[2] - origin[2] > .05
      color = [.95, .44, .08, 1] if elevated else [.015, .48, .62, 1]
      geom = add(mujoco.mjtGeom.mjGEOM_SPHERE, [.007, .007, .007], point, color)
      geom.emission = .45
      geom.specular = 0
    save_png(output, renderer.render())
  return {"selected_cells": int(visible.sum()), "view_width_m": 2 * view_half_extent,
    "all_centers_match_recorded_heights": bool(np.array_equal(points[:, 2] - origin[2], heights[visible]))}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, type=Path, help="Directory containing saved NPZ and experiment.json")
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  record = json.loads((args.input / "experiment.json").read_text())
  source = args.input / "depth-heightfield.npz"
  digest = hashlib.sha256(source.read_bytes()).hexdigest()
  with np.load(source) as arrays:
    heights = arrays["height_m"]
    origin = arrays["map_origin_world_m"]
  model, data = demonstration_scene(record["robot"])
  renders = {}
  for label, extent, distance in (("near-robot", 1.0, 3.8), ("whole-grid", 2.5, 9.5)):
    renders[label] = render_grid(model, data, heights, origin,
      record["grid"]["resolution_m"], record["grid"]["half_extent_m"],
      extent, distance, args.output / f"height-dots-{label}.png")
  assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
  result = {"source_sha256": digest, "source_observed_at": record["observed_at"],
    "new_depth_observations": False, "spacing_m": record["grid"]["resolution_m"],
    "missing_cells": "omitted, not placed at zero", "renders": renders}
  (args.output / "height-dots-render.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
