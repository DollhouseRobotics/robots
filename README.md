# dollhouse-robots

Canonical, public Unitree G1 robot definitions used by Dollhouse control,
simulation, visualization, and motion-retargeting tools.

This repository intentionally exposes exactly two Dollhouse G1 variants:

| ID | Body DoF | Head DoF | Total actuators |
| --- | ---: | ---: | ---: |
| `g1_body23_head2` | 23 | 2 | 25 |
| `g1_body25_head2` | 25 | 2 | 27 |

Both variants use the same passive joint defaults: armature `0.01`, damping
`0.05`, and friction loss `0.2`. The MJCF files contain no floor, gravity,
timestep, or integrator settings; those belong to the consuming simulation.

## Python usage

Pin a commit for deterministic installs:

```toml
dependencies = [
  "dollhouse-robots @ git+https://github.com/carlosdp/dollhouse-robots.git@<commit>",
]
```

Then resolve a definition and its colocated resources:

```python
from dollhouse_robots import get_robot

robot = get_robot("g1_body23_head2")
print(robot.mjcf_path)
print(robot.assets_dir)
print(robot.retarget_config_path)
```

Historical names such as `23dof`, `25dof`, and `29dof` are accepted as API
aliases, but they do not create duplicate definitions.

## Compatibility contract

Consumers must pin a commit or release tag. Each release is validated for:

- exact joint and actuator order;
- referenced mesh completeness;
- uniform passive dynamics;
- MJCF compilation with MuJoCo;
- retarget joint order matching the MJCF actuator order.

Simulator scene configuration and robot-specific calibration remain outside
this package.
