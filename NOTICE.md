# Third-party notices

The Unitree G1 model geometry and portions of the MJCF definitions are derived
from assets published by Unitree Robotics and the MuJoCo Menagerie under the
BSD 3-Clause License:

- Copyright 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
- https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1
- https://github.com/unitreerobotics/unitree_ros

The canonical Dollhouse variants add custom head joints, custom head and hand
geometry, collision definitions, actuator metadata, and retargeting metadata.
Those additions are covered by this repository's LICENSE.

The fixed dual-camera variant includes unchanged STL exports from
https://github.com/DollhouseRobotics/hardware:

- `parts/head-module/dual-camera-neck-mount.stl`, v4 derived from native
  `neck-adapter` object 422 in the September 25, 2026 head-module STEP export.
- `parts/head-module/camera-cover.stl`, exported from the user-supplied
  `neck-v2-camera-cover.step` without geometry edits.

The original head-module CAD is from
https://cad.onshape.com/documents/bef14ae1bcdcb17e3ad8d1d6/w/94ac311f8769171e80bc9526/e/7203dcc260522014059f75d3.
The source paths and SHA-256 hashes are recorded in
`src/dollhouse_robots/unitree_g1/dual-camera-mount.json`. Mesh coordinates remain
in the native millimeter frame; MJCF applies the scale and assembly transforms.
