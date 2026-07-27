from pathlib import Path

import pytest

from dollhouse_robots import available_robots, get_robot


def test_only_two_canonical_variants() -> None:
  assert available_robots() == ("g1_body23_head2", "g1_body25_head2")


@pytest.mark.parametrize(
  ("alias", "canonical"),
  [
    ("23dof", "g1_body23_head2"),
    ("g1-23dof-head", "g1_body23_head2"),
    ("25dof", "g1_body25_head2"),
    ("29dof", "g1_body25_head2"),
  ],
)
def test_legacy_aliases_collapse_to_canonical_variant(alias: str, canonical: str) -> None:
  robot = get_robot(alias)
  assert robot.id == canonical
  assert robot.mjcf_path.is_file()
  assert robot.assets_dir.is_dir()
  assert robot.retarget_config_path.is_file()
  assert isinstance(robot.mjcf_path, Path)


def test_unknown_variant_is_rejected() -> None:
  with pytest.raises(ValueError, match="Unknown robot"):
    get_robot("g1_body24_head2")
