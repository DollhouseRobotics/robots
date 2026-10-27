"""Regenerate the fixed-camera G1 from the body25 definition and pinned CAD meshes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/dollhouse_robots/unitree_g1"
ROBOT_ID = "dollhouse_g1_27dof_with_camera_mount"


def vector(values) -> str:
  return " ".join(f"{value:.12g}" for value in values)


def build(hardware: Path) -> None:
  metadata = json.loads((PACKAGE / "dual-camera-mount.json").read_text())
  # Verify every input before replacing any generated output.
  for source in metadata["mesh_sources"].values():
    actual = hashlib.sha256((hardware / source["source"]).read_bytes()).hexdigest()
    if actual != source["sha256"]:
      raise ValueError(f"Unexpected hardware source: {source['source']}: {actual}")
  for name, source in metadata["mesh_sources"].items():
    shutil.copyfile(hardware / source["source"], PACKAGE / "assets" / name)

  parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
  root = ET.parse(PACKAGE / "g1_body25_head2.xml", parser=parser).getroot()
  root.set("model", ROBOT_ID)
  asset = root.find("asset")
  for name in ("head_pan_secure", "head_tilt_holder"):
    asset.remove(asset.find(f"mesh[@name='{name}']"))
  for name, file in (("dual_camera_mount", "dual-camera-neck-mount.stl"),
                     ("camera_cover", "camera-cover.stl")):
    ET.SubElement(asset, "mesh", name=name, file=file, scale="0.001 0.001 0.001")

  torso = root.find(".//body[@name='torso_link']")
  for selector in ("geom[@name='head_collision']", "body[@name='head_pan_link']",
                   "body[@name='fisheye_front']", "body[@name='fisheye_back']"):
    torso.remove(torso.find(selector))

  # Torso = Rz(90 degrees) * CAD_mm / 1000 + translation_m.
  # Nominal seat: X=13.5 mm, Y=0, Z=303.899 mm. This is NOT a calibrated transform.
  mount = ET.SubElement(torso, "body", name="fixed_camera_mount",
    pos=vector(metadata["registration"]["translation_m"]),
    quat=vector(metadata["registration"]["rotation_wxyz"]))
  mount.append(ET.Comment(" Nominal socket fit and lens offsets; see dual-camera-mount.json. "))
  # A nominal 180 g bounding-box inertia, not a measured assembly mass distribution.
  sizes = (0.05, 0.1262, 0.0885)
  mass = metadata["dynamics"]["assembly_mass_kg"]
  inertia = [mass * sum(sizes[j] ** 2 for j in range(3) if j != i) / 12
             for i in range(3)]
  ET.SubElement(mount, "inertial", pos="0 0.062 -0.104888134", mass=str(mass),
    diaginertia=vector(inertia))
  ET.SubElement(mount, "geom", name="fixed_camera_mount_visual", attrib={"class": "visual"},
    mesh="dual_camera_mount")
  ET.SubElement(mount, "geom", name="front_camera_cover_visual", attrib={"class": "visual"},
    mesh="camera_cover", pos="0 0.073100000358 -0.165582515587", quat="0 0 0 1")
  ET.SubElement(mount, "geom", name="back_camera_cover_visual", attrib={"class": "visual"},
    mesh="camera_cover", pos="0 0.050900000358 -0.165582515587")
  for name, pos, size in (
    ("fixed_neck_collision", "0 0.062 -0.135542515587", "0.009075 0.0065 0.0297"),
    ("camera_bridge_collision", "0 0.062 -0.097342515587", "0.01815 0.0385 0.011"),
    ("front_camera_collision", "0 0.012200000358 -0.101742515587", "0.025 0.0113 0.025"),
    ("back_camera_collision", "0 0.111800000358 -0.101742515587", "0.025 0.0113 0.025"),
  ):
    ET.SubElement(mount, "geom", name=name, attrib={"class": "collision"},
      type="box", pos=pos, size=size, density="0")

  optics = metadata["optical_model"]
  front_y = optics["front_holder_face_y_mm"] - optics["nominal_pupil_outward_from_holder_face_mm"]
  rear_y = 2 * optics["rear_rotation_axis_y_mm"] - front_y
  # Explicit quaternions also work with consumers whose MJCF parsers ignore xyaxes.
  for name, y, quat in (
    ("fisheye_front", front_y, "0 0 0.707106781187 0.707106781187"),
    ("fisheye_back", rear_y, "0.707106781187 0.707106781187 0 0"),
  ):
    body = ET.SubElement(mount, "body", name=name,
      pos=vector((0, y / 1000, optics["camera_center_z_mm"] / 1000)), quat=quat)
    ET.SubElement(body, "site", name=name, type="sphere", size="0.002")
    body.append(ET.Comment(" Pose anchor only: standard MuJoCo camera rendering is pinhole. "))
    ET.SubElement(body, "camera", name=name, fovy=str(optics["mjcf_camera_fovy_degrees"]), ipd="0")
    # Nominal module visuals. The entrance pupil is at the camera/site origin;
    # all opaque lens geometry is behind it, never covering its central ray.
    ET.SubElement(body, "geom", name=f"{name}_board_visual", attrib={"class": "visual"},
      type="box", size="0.018 0.018 0.001", pos="0 0 0.013", rgba="0.035 0.08 0.07 1")
    ET.SubElement(body, "geom", name=f"{name}_lens_barrel", attrib={"class": "visual"},
      type="cylinder", size="0.009 0.004", pos="0 0 0.006", rgba="0.035 0.04 0.045 1")
    ET.SubElement(body, "geom", name=f"{name}_lens_glass", attrib={"class": "visual"},
      type="ellipsoid", size="0.006 0.006 0.002", pos="0 0 0.0021", rgba="0.07 0.17 0.23 1")

  actuators = root.find("actuator")
  for actuator in list(actuators):
    if actuator.get("joint") in ("head_pan_joint", "head_tilt_joint"):
      actuators.remove(actuator)
  ET.indent(root, space="  ")
  ET.ElementTree(root).write(PACKAGE / f"{ROBOT_ID}.xml", encoding="unicode")

  config = json.loads((PACKAGE / "retarget/g1_body25_head2/soma_to_g1_retargeter_config.json").read_text())
  config["target_name"] = ROBOT_ID
  config["mjcf_path"] = f"../../{ROBOT_ID}.xml"
  config["target_joint_names"] = [a.get("joint") for a in actuators]
  for name in ("head_pan_link", "head_tilt_link"):
    del config["smooth_joint_filter_objective_body_masks"][name]
  # Do not make the fixed camera mount an IK target for human head movement.
  del config["ik_map"]["Neck1"]
  directory = PACKAGE / "retarget" / ROBOT_ID
  directory.mkdir(exist_ok=True)
  (directory / "soma_to_g1_retargeter_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("hardware", type=Path, help="Checkout containing the hash-pinned v4 meshes")
  build(parser.parse_args().hardware)
