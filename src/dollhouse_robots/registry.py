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

  return ("g1_body23_head2", "g1_body25_head2")


def get_robot(name: str) -> RobotDefinition:
  """Resolve a canonical robot ID or a supported historical alias."""

  normalized = name.strip().lower().replace("-", "_")
  try:
    robot_id = _ALIASES[normalized]
  except KeyError as exc:
    choices = ", ".join(available_robots())
    raise ValueError(f"Unknown robot {name!r}; expected one of: {choices}") from exc

  body_dof = 23 if robot_id == "g1_body23_head2" else 25
  return RobotDefinition(
    id=robot_id,
    body_dof=body_dof,
    head_dof=2,
    mjcf_path=_ROOT / f"{robot_id}.xml",
    assets_dir=_ROOT / "assets",
    retarget_config_path=(
      _ROOT / "retarget" / robot_id / "soma_to_g1_retargeter_config.json"
    ),
  )
