# dollhouse-robots

Canonical, public Unitree G1 robot definitions used by Dollhouse control,
simulation, visualization, and motion-retargeting tools.

This repository exposes these Dollhouse G1 variants:

| ID | Body DoF | Head DoF | Total actuators |
| --- | ---: | ---: | ---: |
| `g1_body23_head2` | 23 | 2 | 25 |
| `g1_body25_head2` | 25 | 2 | 27 |
| `dollhouse_g1_27dof_with_camera_mount` | 25 | 0 | 25 |

All variants use the same passive joint defaults: armature `0.01`, damping
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

## Dollhouse G1 27 DOF with camera mount

`dollhouse_g1_27dof_with_camera_mount` keeps the body25 torso, limbs and actuator order,
removes the pan/tilt mechanism and head collision capsule, and adds the hardware
repository's v4 printed mount and two covers. Existing variants and aliases do
not change. Its retarget config has no head joints or human neck IK target.

The name identifies its 27-DOF source model, `g1_body25_head2`. Replacing the two
head joints with a fixed mount leaves **25 actuated joints**, not 27. The MJCF,
manifest and retarget config all retain that actual count. The earlier working
name `g1_body25_head0_dual_camera` remains an API alias so saved experiment records
can still be rendered.

```python
robot = get_robot("dollhouse_g1_27dof_with_camera_mount")
```

The front and rear camera/site pairs are `fisheye_front` and `fisheye_back`.
They are rigid descendants of `torso_link`. Both use MuJoCo's camera frame,
+X image right, +Y image up, and -Z optical forward. In the torso frame they
look along +X and -X, respectively, with +Z image-up.

**Placement is nominal, not calibrated to Leaning Tree.** The photos establish
orientation but not seating depth or optical centers. The CAD foot is placed on
the central torso mesh ledge at Z=303.899 mm, with neck axis X=13.5 mm, Y=0.
The assumed entrance pupil is centered in each cover opening, 7 mm outward
from the holder face. This gives torso-relative positions in meters:

- Front: `[0.0766, 0, 0.367399]`.
- Rear: `[-0.0496, 0, 0.367399]`.

The 180 g assembly mass, bounding-box inertia and camera board/lens primitives
are estimates. Contact boxes are coarse approximations. The unchanged torso
mesh includes shell geometry absent from the installation photos. Do not use
this model as evidence of physical clearance, mass calibration or pixel accuracy.

Mesh hashes, native coordinates, the editable registration transform and optical
assumptions are recorded in
[`dual-camera-mount.json`](src/dollhouse_robots/unitree_g1/dual-camera-mount.json).
The generator derives the new MJCF and retarget config without editing either
articulated-head definition. Regenerate with a hardware checkout containing the
hash-pinned v4 source meshes:

```sh
python tools/build_fixed_camera_variant.py /path/to/hardware
uv run pytest -q
```

### Render actual 200-degree ray coverage

The MJCF cameras are pose anchors with ordinary 90-degree pinhole previews.
MuJoCo's standard camera does not become a fisheye by setting `fovy=200`.
The example below renders six coincident 90-degree views at each optical site
and reprojects them with the ideal equidistant law `r = f * theta`.
Here, **200 degrees means the diameter of a circular field**, extending to
100 degrees from the optical axis. It is not a measured horizontal or diagonal
specification for the physical cameras, and does not reproduce their sensor crop.

```sh
MUJOCO_GL=egl uv run python examples/render_fixed_cameras.py --output /tmp/g1-cameras
MUJOCO_GL=egl uv run pytest -q
```

This writes front/rear torso close-ups, two circular fisheye PNGs and a JSON pose
record. The synthetic room contains colored panels, a table and shelves; it is
not a reconstruction of the photographed room. The sensor images include robot
self-occlusion. The example uses the existing MuJoCo dev environment and adds no
package dependency. Other platforms can select their supported MuJoCo GL backend.
The render test checks known angular landmarks, including visible targets at 95
and 99 degrees off-axis and an excluded target at 101 degrees. It skips only when
neither a GL backend nor a display has been selected.

To obtain calibrated images, measure the mount-to-torso transform and each lens's
entrance pupil, then supply measured pixel-to-ray calibration and the sensor's
actual image footprint. Neither photos nor a nominal 200-degree label establish
those quantities.

### Simulate depth and an observed height field

```sh
uv run python examples/fisheye_heightfield.py --output /tmp/g1-depth
```

This CPU-only example raycasts the visible scene at each of the two cameras'
640 x 640 equidistant pixel directions. It measures radial distance, including
rays beyond the forward hemisphere, with a 0.02–8 m range. It does not infer
depth from RGB or assume the real cameras provide depth.

The output contains one synchronized, noiseless pair, backprojected environment
points, front-only/rear-only/fused height grids, point counts and coverage metrics
in `depth-heightfield.npz` and `experiment.json`. Coordinates use the demo's
known world pose and floor plane. The 5 x 5 m grid is centered beneath the torso,
with 5 cm cells, rows along forward X and columns along left Y. Heights are
meters above the floor. The camera transform remains nominal.

Each cell stores its highest observed return between -0.05 and 1.5 m above the
floor. Unknown cells stay NaN, including self-occluded and hidden regions.
Robot geometry blocks the rays before its returns are removed with exact
simulator geometry IDs; this masking is an idealization, not a real segmentation
system. Collision proxies are not used for sensing. There is no interpolation,
ground-truth fill, temporal fusion or noise model.

This is an observed-surface map, not walkable terrain or an omniscient height
field. Tables, shelves and overhangs can hide the floor or higher surfaces even
in cells that have returns. A single height per cell loses that layered geometry.

Render those saved heights as a 3D dot grid around the robot:

```sh
MUJOCO_GL=egl uv run python examples/render_heightfield_points.py --input /tmp/g1-depth --output /tmp/g1-depth
```

This writes 2 x 2 m and 5 x 5 m views with 5 cm dot spacing. Cyan marks floor-level
returns; orange marks returns more than 5 cm above the floor. Each dot retains its
saved height and unknown cells have no dot. This changes only the presentation,
not the recorded depth observations.
