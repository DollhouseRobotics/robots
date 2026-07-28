import json
from xml.etree import ElementTree

import numpy as np
import pytest

from dollhouse_robots import get_robot


@pytest.mark.parametrize(
  ("robot_id", "expected_actuators"),
  [("g1_body23_head2", 25), ("g1_body25_head2", 27)],
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
