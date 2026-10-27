from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final


_ROOT: Final = Path(str(files("dollhouse_robots").joinpath("unitree_g1")))

_ALIASES: Final = {
  "g1_body23_head2": "g1_body23_head2",
  "23dof": "g1_body23_head2",
  "g1_23dof": "g1_body23_head2",
  "g1_23dof_head": "g1_body23_head2",
  "g1_body25_head2": "g1_body25_head2",
  "25dof": "g1_body25_head2",
  "29dof": "g1_body25_head2",
  "g1_25dof": "g1_body25_head2",
  "g1_25dof_head": "g1_body25_head2",
  "g1_29dof": "g1_body25_head2",
  "dollhouse_g1_27dof_with_camera_mount": "dollhouse_g1_27dof_with_camera_mount",
  "g1_body25_head0_dual_camera": "dollhouse_g1_27dof_with_camera_mount",
}


@dataclass(frozen=True, slots=True)
class RobotDefinition:
  """Paths and compatibility metadata for one canonical robot definition."""

  id: str
  body_dof: int
  head_dof: int
  mjcf_path: Path
  assets_dir: Path
  retarget_config_path: Path

  @property
  def total_dof(self) -> int:
    return self.body_dof + self.head_dof

  @property
  def manifest(self) -> dict:
    with (_ROOT / "manifest.json").open(encoding="utf-8") as handle:
      return json.load(handle)["robots"][self.id]


def available_robots() -> tuple[str, ...]:
  """Return canonical IDs. Aliases are intentionally omitted."""

  with (_ROOT / "manifest.json").open(encoding="utf-8") as handle:
    return tuple(json.load(handle)["robots"])


def get_robot(name: str) -> RobotDefinition:
  """Resolve a canonical robot ID or a supported historical alias."""

  normalized = name.strip().lower().replace("-", "_")
  try:
    robot_id = _ALIASES[normalized]
  except KeyError as exc:
    choices = ", ".join(available_robots())
    raise ValueError(f"Unknown robot {name!r}; expected one of: {choices}") from exc

  with (_ROOT / "manifest.json").open(encoding="utf-8") as handle:
    definition = json.load(handle)["robots"][robot_id]
  return RobotDefinition(
    id=robot_id,
    body_dof=definition["body_dof"],
    head_dof=definition["head_dof"],
    mjcf_path=_ROOT / definition["mjcf"],
    assets_dir=_ROOT / "assets",
    retarget_config_path=_ROOT / definition["retarget_config"],
  )
