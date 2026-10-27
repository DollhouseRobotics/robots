"""Render a nominal fixed-camera G1 and true 200-degree equidistant sensor views.

Run with the repository dev environment. Set MUJOCO_GL=egl on headless Linux.
No measured lens calibration is implied. The example room is not Leaning Tree's room.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from xml.etree import ElementTree as ET
import zlib

import mujoco
import numpy as np

from dollhouse_robots import get_robot


def save_png(path: Path, pixels: np.ndarray) -> None:
  """Write RGB PNGs without adding an image-library dependency to the package."""
  height, width, _ = pixels.shape
  def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
  rows = b"".join(b"\0" + row.tobytes() for row in pixels.astype(np.uint8))
  path.write_bytes(b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def equidistant_rays(size: int, fov_degrees: float) -> tuple[np.ndarray, np.ndarray]:
  """Unit rays in the MuJoCo camera frame: right, up, backward.

  Pixel centers sample a circle inscribed in the image bounds. Its continuous
  radius corresponds to fov/2, so the 200-degree rim looks 10 degrees backward.
  """
  if size < 2 or not 0 < fov_degrees < 360:
    raise ValueError("Expected size >= 2 and 0 < circular FOV < 360 degrees")
  y, x = np.mgrid[:size, :size]
  x = (x + 0.5 - size / 2) / (size / 2)
  y = (size / 2 - y - 0.5) / (size / 2)
  radius = np.hypot(x, y)
  theta = radius * np.deg2rad(fov_degrees / 2)
  scale = np.divide(np.sin(theta), radius, out=np.zeros_like(radius), where=radius != 0)
  return np.stack((x * scale, y * scale, -np.cos(theta)), axis=-1), radius <= 1


def render_fisheye(renderer, data, camera_name, size=1000, fov_degrees=200):
  """Six coincident 90-degree renders, then bilinear cubemap reprojection.

  The robot, including both camera covers, participates in visibility. Do not
  widen a pinhole FOV to 200 or warp one perspective image: those miss rear rays.
  """
  if renderer.width != renderer.height:
    raise ValueError("Cubemap renderer must be square")
  rays, valid = equidistant_rays(size, fov_degrees)
  # (forward, up) in camera-local coordinates, one pair for each cube face.
  faces = np.array([
    [[0, 0, -1], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]],
    [[-1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]],
    [[0, -1, 0], [0, 0, -1]], [[0, 0, 1], [0, 1, 0]],
  ], dtype=float)
  selected = np.argmax(rays @ faces[:, 0].T, axis=-1)
  rotation = data.camera(camera_name).xmat.reshape(3, 3)
  origin = data.camera(camera_name).xpos
  options = mujoco.MjvOption()
  options.geomgroup[3] = 0
  options.sitegroup[:] = 0
  output = np.zeros((size, size, 3), dtype=np.uint8)
  for index, (forward, up) in enumerate(faces):
    renderer.update_scene(data, camera=camera_name, scene_option=options)
    for camera in renderer.scene.camera:
      camera.pos[:] = origin
      camera.forward[:] = rotation @ forward
      camera.up[:] = rotation @ up
      # Avoid the default near plane clipping nearby covers and the robot.
      camera.frustum_near = 0.0001
      camera.frustum_far = 50
      camera.frustum_top = 0.0001
      camera.frustum_bottom = -0.0001
      camera.frustum_center = 0
      # Zero derives horizontal half-width from the square viewport. A nonzero
      # value is a HALF-width in MuJoCo, despite the field's name.
      camera.frustum_width = 0
      camera.orthographic = 0
    image = renderer.render()
    mask = valid & (selected == index)
    direction = rays[mask]
    depth = direction @ forward
    right = np.cross(forward, up)
    u = (direction @ right / depth + 1) * renderer.width / 2 - 0.5
    v = (1 - direction @ up / depth) * renderer.height / 2 - 0.5
    u = np.clip(u, 0, renderer.width - 1)
    v = np.clip(v, 0, renderer.height - 1)
    x0, y0 = u.astype(int), v.astype(int)
    x1 = np.minimum(x0 + 1, renderer.width - 1)
    y1 = np.minimum(y0 + 1, renderer.height - 1)
    dx, dy = (u - x0)[:, None], (v - y0)[:, None]
    output[mask] = np.round(
      image[y0, x0] * (1 - dx) * (1 - dy) + image[y0, x1] * dx * (1 - dy)
      + image[y1, x0] * (1 - dx) * dy + image[y1, x1] * dx * dy
    ).astype(np.uint8)
  return output


def demonstration_scene(robot_id: str):
  robot = get_robot(robot_id)
  root = ET.parse(robot.mjcf_path).getroot()
  root.find("compiler").set("meshdir", str(robot.assets_dir))
  visual = ET.SubElement(root, "visual")
  ET.SubElement(visual, "global", offwidth="1600", offheight="1200")
  ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.8 0.8 0.8")
  ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
  ET.SubElement(visual, "map", znear="0.0001")
  assets = root.find("asset")
  ET.SubElement(assets, "texture", type="skybox", builtin="gradient",
    rgb1="0.87 0.9 0.93", rgb2="0.95 0.96 0.97", width="512", height="3072")
  ET.SubElement(assets, "texture", name="room_checker", type="2d", builtin="checker",
    rgb1="0.47 0.51 0.56", rgb2="0.63 0.66 0.69", width="512", height="512")
  ET.SubElement(assets, "material", name="room_floor", texture="room_checker", texrepeat="12 12")
  world = root.find("worldbody")
  ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.65 0.65 0.65")
  ET.SubElement(world, "geom", name="demo_floor", type="plane", size="4 4 .1", material="room_floor")
  def box(name, pos, size, rgba):
    ET.SubElement(world, "geom", name=name, type="box", pos=pos, size=size, rgba=rgba)
  box("front_wall", "3 0 1.5", ".05 3 1.5", ".78 .8 .83 1")
  box("rear_wall", "-3 0 1.5", ".05 3 1.5", ".72 .77 .81 1")
  box("left_wall", "0 3 1.5", "3 .05 1.5", ".82 .83 .85 1")
  box("right_wall", "0 -3 1.5", "3 .05 1.5", ".82 .83 .85 1")
  box("ceiling", "0 0 3", "3 3 .025", ".88 .89 .9 1")
  box("front_green_panel", "2.92 .7 1.4", ".025 .65 .7", ".08 .48 .30 1")
  box("front_orange_panel", "2.92 -1 1.4", ".025 .35 .5", ".82 .32 .08 1")
  box("rear_blue_panel", "-2.92 0 1.4", ".025 .85 .65", ".1 .31 .66 1")
  box("left_yellow_panel", ".7 2.92 1.3", ".5 .025 .65", ".82 .64 .14 1")
  box("right_red_panel", "-.5 -2.92 1.3", ".55 .025 .65", ".68 .18 .18 1")
  box("table_top", "1.5 0 .72", ".35 .8 .035", ".48 .29 .14 1")
  for x in (1.22, 1.78):
    for y in (-.7, .7):
      box(f"table_leg_{x}_{y}", f"{x} {y} .35", ".03 .03 .35", ".2 .21 .22 1")
  for z in (.35, .85, 1.35):
    box(f"rear_shelf_{z}", "-1.8 .6 " + str(z), ".3 .8 .025", ".48 .36 .23 1")
  for y in (-.15, 1.35):
    box(f"shelf_support_{y}", f"-1.95 {y} .75", ".04 .04 .75", ".22 .23 .25 1")
  model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--robot", default="dollhouse_g1_27dof_with_camera_mount")
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  model, data = demonstration_scene(args.robot)
  opt = mujoco.MjvOption()
  # Hide the example room only for the close-ups, not for the sensor images.
  opt.geomgroup[0] = 0
  opt.geomgroup[3] = 0
  opt.sitegroup[:] = 0
  with mujoco.Renderer(model, height=1000, width=1400) as renderer:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.body("torso_link").xpos + [0, 0, .335]
    camera.distance = .42
    camera.elevation = -12
    for label, azimuth in (("torso-front", 218), ("torso-rear", 328)):
      camera.azimuth = azimuth
      renderer.update_scene(data, camera=camera, scene_option=opt)
      save_png(args.output / f"{label}.png", renderer.render())
  poses = {}
  with mujoco.Renderer(model, height=1024, width=1024) as renderer:
    for name in ("fisheye_front", "fisheye_back"):
      save_png(args.output / f"{name}-200.png", render_fisheye(renderer, data, name))
      poses[name] = {
        "world_position_m": data.camera(name).xpos.tolist(),
        "world_rotation": data.camera(name).xmat.reshape(3, 3).tolist(),
      }
  report = {"robot": args.robot, "mujoco": mujoco.__version__, "cameras": poses,
    "projection": "ideal equidistant, 200-degree circular FOV; not calibrated",
    "scene": "synthetic example room, static zero-joint pose; not the photographed room"}
  (args.output / "render.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
