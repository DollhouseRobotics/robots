import json
from xml.etree import ElementTree

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
