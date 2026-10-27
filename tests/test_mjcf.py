import json
from xml.etree import ElementTree

import numpy as np
import pytest

from dollhouse_robots import get_robot


@pytest.mark.parametrize(
  ("robot_id", "expected_actuators"),
  [("g1_body23_head2", 25), ("g1_body25_head2", 27), ("dollhouse_g1_27dof_with_camera_mount", 25)],
)
def test_definition_contract(robot_id: str, expected_actuators: int) -> None:
  robot = get_robot(robot_id)
  root = ElementTree.parse(robot.mjcf_path).getroot()

  assert root.attrib["model"] == robot_id
  assert root.find("option") is None
  assert root.find(".//geom[@name='checkpoint_floor']") is None

  joint_default = root.find("./default/default[@class='g1']/joint")
  assert joint_default is not None
  assert joint_default.attrib == {
    "armature": "0.01",
    "damping": "0.05",
    "frictionloss": "0.2",
  }

  actuator_section = root.find("./actuator")
  assert actuator_section is not None
  actuators = list(actuator_section)
  assert len(actuators) == expected_actuators
  assert robot.total_dof == expected_actuators

  mesh_files = {mesh.attrib["file"] for mesh in root.findall("./asset/mesh")}
  assert mesh_files
  assert not [
    filename for filename in mesh_files if not (robot.assets_dir / filename).is_file()
  ]

  with robot.retarget_config_path.open(encoding="utf-8") as handle:
    retarget = json.load(handle)
  assert retarget["target_name"] == robot_id
  assert len(retarget["target_joint_names"]) == expected_actuators
  assert set(retarget["target_joint_names"]) == {
    actuator.attrib["joint"] for actuator in actuators
  }


@pytest.mark.parametrize("robot_id", ["g1_body23_head2", "g1_body25_head2"])
def test_scalebfm_collision_primitives(robot_id: str) -> None:
  mujoco = pytest.importorskip("mujoco")
  model = mujoco.MjModel.from_xml_path(str(get_robot(robot_id).mjcf_path))

  def geom_id(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

  def radius(name: str) -> float:
    return float(model.geom_size[geom_id(name), 0])

  assert geom_id("torso_collision") < 0
  assert geom_id("head_pan_collision") < 0
  assert geom_id("left_wrist_hand_collision") < 0
  assert radius("torso_collision1") == pytest.approx(0.073)
  assert radius("torso_collision2") == pytest.approx(0.07)
  assert radius("torso_collision3") == pytest.approx(0.065)
  assert radius("head_collision") == pytest.approx(0.068)
  assert radius("left_wrist_collision") == pytest.approx(0.035)
  assert radius("right_wrist_collision") == pytest.approx(0.035)
  assert radius("left_hand_collision") == pytest.approx(0.04)
  assert radius("right_hand_collision") == pytest.approx(0.04)
  assert int(model.geom_type[geom_id("left_hand_collision")]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_pitch_link") < 0


@pytest.mark.parametrize("robot_id", ["g1_body23_head2", "g1_body25_head2", "dollhouse_g1_27dof_with_camera_mount"])
def test_mjcf_compiles_with_mujoco(robot_id: str) -> None:
  mujoco = pytest.importorskip("mujoco")
  model = mujoco.MjModel.from_xml_path(str(get_robot(robot_id).mjcf_path))
  assert model.nu == get_robot(robot_id).total_dof


def test_25dof_fisheye_mounts_have_expected_world_poses() -> None:
  mujoco = pytest.importorskip("mujoco")
  model = mujoco.MjModel.from_xml_path(str(get_robot("g1_body25_head2").mjcf_path))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  expected = {
    "fisheye_front": (
      np.array([0.048714, 0.008499, 1.160682]),
      np.array([1.0, 0.0, 0.0]),
    ),
    "fisheye_back": (
      np.array([-0.062882, 0.013141, 1.158913]),
      np.array([-1.0, 0.0, 0.0]),
    ),
  }
  for name, (position, forward) in expected.items():
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    assert body_id >= 0
    assert site_id >= 0
    assert camera_id >= 0
    np.testing.assert_allclose(data.site_xpos[site_id], position, atol=1e-9)
    np.testing.assert_allclose(data.cam_xpos[camera_id], position, atol=1e-9)
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    np.testing.assert_allclose(-camera_rotation[:, 2], forward, atol=1e-9)


@pytest.mark.parametrize("waist_pose", [(0, 0, 0), (0.47, -0.23, 0.31)])
def test_fixed_cameras_follow_torso_without_head_actuators(waist_pose) -> None:
  mujoco = pytest.importorskip("mujoco")
  robot = get_robot("dollhouse_g1_27dof_with_camera_mount")
  model = mujoco.MjModel.from_xml_path(str(robot.mjcf_path))
  old = mujoco.MjModel.from_xml_path(str(get_robot("g1_body25_head2").mjcf_path))
  assert robot.head_dof == 0
  assert model.nu == 25
  assert model.nq == 32
  assert [model.actuator(i).name for i in range(model.nu)] == [
    old.actuator(i).name for i in range(25)
  ]
  for kind, names in (
    (mujoco.mjtObj.mjOBJ_JOINT, ("head_pan_joint", "head_tilt_joint")),
    (mujoco.mjtObj.mjOBJ_BODY, ("head_pan_link", "head_tilt_link")),
    (mujoco.mjtObj.mjOBJ_GEOM, ("head_collision",)),
  ):
    for name in names:
      assert mujoco.mj_name2id(model, kind, name) == -1

  data = mujoco.MjData(model)
  data.qpos[:3] = [0.8, -0.3, 1.1]
  data.qpos[3:7] = [np.cos(.19), 0, 0, np.sin(.19)]
  for axis, value in zip(("yaw", "roll", "pitch"), waist_pose):
    data.joint(f"waist_{axis}_joint").qpos[0] = value
  mujoco.mj_forward(model, data)
  torso = data.body("torso_link")
  rotation = torso.xmat.reshape(3, 3)
  # Independently specified nominal torso-relative pupil centers, in meters.
  for name, x, forward in (
    ("fisheye_front", .0766, [1, 0, 0]),
    ("fisheye_back", -.0496, [-1, 0, 0]),
  ):
    site = data.site(name)
    camera = data.camera(name)
    position = torso.xpos + rotation @ [x, 0, .367399]
    np.testing.assert_allclose(site.xpos, position, atol=1e-9)
    np.testing.assert_allclose(camera.xpos, position, atol=1e-9)
    np.testing.assert_allclose(camera.xmat, site.xmat, atol=1e-9)
    camera_rotation = camera.xmat.reshape(3, 3)
    np.testing.assert_allclose(-camera_rotation[:, 2], rotation @ forward, atol=1e-9)
    np.testing.assert_allclose(camera_rotation[:, 1], rotation @ [0, 0, 1], atol=1e-9)

  retarget = json.loads(robot.retarget_config_path.read_text())
  assert retarget["target_joint_names"] == [
    model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)
  ]
  assert "Neck1" not in retarget["ik_map"]
  for mapping in retarget["ik_map"].values():
    for key in ("t_body", "r_body"):
      assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mapping[key]) >= 0
