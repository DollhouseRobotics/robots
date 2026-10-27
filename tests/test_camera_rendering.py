import os
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.mark.skipif(
  not (os.environ.get("MUJOCO_GL") or os.environ.get("DISPLAY")),
  reason="Rendering needs a GL backend; use MUJOCO_GL=egl on headless Linux",
)
def test_fisheye_projects_landmarks_through_100_degrees() -> None:
  mujoco = pytest.importorskip("mujoco")
  example = runpy.run_path(str(Path(__file__).parents[1] / "examples/render_fixed_cameras.py"))
  model = mujoco.MjModel.from_xml_string('''
    <mujoco>
      <visual>
        <global offwidth="512" offheight="512"/>
        <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
      </visual>
      <worldbody>
        <camera name="probe" pos=".23 -.17 .61" quat="-.5 -.5 .5 .5" fovy="90" ipd="0"/>
        <geom type="sphere" size=".006" pos="2 0 0" rgba="1 0 0 1"/>
      </worldbody>
    </mujoco>
  ''')
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  origin = data.camera("probe").xpos.copy()
  rotation = data.camera("probe").xmat.reshape(3, 3).copy()
  # The seam, behind-forward-hemisphere rays, both sides of the circular rim,
  # and asymmetric directions distinguish projection/axis mistakes.
  cases = [(0, 0), (37, 0), (45, 0), (75, 43), (89, 0), (95, 0),
           (99, 0), (101, 0), (95, 180), (67, 90), (52, -90)]
  with mujoco.Renderer(model, height=512, width=512) as renderer:
    for theta, phi in cases:
      t, p = np.deg2rad([theta, phi])
      ray = [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), -np.cos(t)]
      model.geom_pos[0] = origin + rotation @ ray
      mujoco.mj_forward(model, data)
      image = example["render_fisheye"](renderer, data, "probe", size=1000)
      red = (image[:, :, 0] > 100) & (image[:, :, 1] < 50) & (image[:, :, 2] < 50)
      ys, xs = np.nonzero(red)
      if theta > 100:
        assert len(xs) == 0, (theta, phi)
        continue
      assert len(xs) > 0, (theta, phi)
      # Independent equidistant expectation: 500 pixels spans 100 degrees.
      expected = [499.5 + 5 * theta * np.cos(p), 499.5 - 5 * theta * np.sin(p)]
      assert np.linalg.norm(np.array([xs.mean(), ys.mean()]) - expected) < 2, (theta, phi)
      # A duplicated landmark could have the right centroid; reject ghosts too.
      assert np.max(np.linalg.norm(np.column_stack((xs, ys)) - expected, axis=1)) < 5


@pytest.fixture
def depth_example(monkeypatch):
  directory = Path(__file__).parents[1] / "examples"
  monkeypatch.syspath_prepend(str(directory))
  return runpy.run_path(str(directory / "fisheye_heightfield.py"))


def test_fisheye_depth_is_radial_and_preserves_robot_occlusion(depth_example) -> None:
  mujoco = pytest.importorskip("mujoco")
  model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody>
      <camera name="probe" pos=".23 -.17 .61" quat="-.5 -.5 .5 .5"/>
      <geom name="front" type="box" pos="2.25 -.17 .61" size=".02 5 5"/>
      <geom name="side" type="box" pos=".23 -1.19 .61" size="5 .02 5"/>
      <geom name="floor" type="plane" size="10 10 .1"/>
      <body name="pelvis" pos=".8 -.17 .61">
        <geom name="self" type="sphere" size=".1"/>
        <body name="child" pos="0 0 .3"><geom type="sphere" size=".1"/></body>
      </body>
    </worldbody></mujoco>
  ''')
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  depth, ids = depth_example["simulate_depth"](model, data, "probe", size=101)
  assert depth[50, 50] == pytest.approx(.47)
  assert ids[50, 50] == model.geom("self").id
  robot = depth_example["robot_geometries"](model)
  assert robot.tolist() == [False, False, False, True, True]
  assert np.isnan(depth[0, 0]) and ids[0, 0] == -1
  theta = np.deg2rad(100 * 10 / 50.5)
  assert depth[50, 60] == pytest.approx(2 / np.cos(theta))
  assert ids[50, 60] == model.geom("front").id
  # A ray 95.05 degrees off-axis points behind the camera's forward hemisphere.
  theta_back = np.deg2rad(100 * 48 / 50.5)
  assert depth[50, 98] == pytest.approx(1 / np.sin(theta_back))
  points = depth_example["unproject_depth"](depth, data.camera("probe").xpos,
    data.camera("probe").xmat.reshape(3, 3))
  np.testing.assert_allclose(points[50, 60], [2.23, -.17 - 2 * np.tan(theta), .61], atol=1e-10)
  np.testing.assert_allclose(points[50, 98], [.23 + 1 / np.tan(theta_back), -1.17, .61], atol=1e-10)
  theta_down = np.deg2rad(100 * 25 / 50.5)
  assert depth[75, 50] == pytest.approx(.61 / np.sin(theta_down))
  assert ids[75, 50] == model.geom("floor").id
  np.testing.assert_allclose(points[75, 50], [.23 + .61 / np.tan(theta_down), -.17, 0], atol=1e-10)
  # A clipped first return must not reveal the surface behind it.
  limited, _ = depth_example["simulate_depth"](model, data, "probe", size=101,
    min_range=.5, max_range=1.5)
  assert np.isnan(limited[50, 50])
  assert np.isnan(limited[50, 60])
  assert np.isfinite(limited[50, 98])


def test_height_field_uses_observed_max_and_leaves_unknowns(depth_example) -> None:
  # Asymmetric XY, two layers, exact lower/upper edges and invalid samples.
  front = np.array([[-1, .1, .2], [-.8, .2, .7], [.6, -.7, .4]])
  rear = np.array([[-.9, .3, .5], [.8, .8, 0], [1, 0, .9],
    [-1.01, .1, .8], [0, 0, 2], [np.nan, 0, .4]])
  height_field = depth_example["height_field"]
  fused, counts = height_field(np.concatenate([front, rear]), half_extent=1, resolution=.5)
  expected = np.full((4, 4), np.nan)
  expected[0, 2], expected[3, 0], expected[3, 3] = .7, .4, 0
  np.testing.assert_allclose(fused, expected, equal_nan=True)
  assert counts[0, 2] == 3 and counts.sum() == 5
  a, _ = height_field(front, half_extent=1, resolution=.5)
  b, _ = height_field(rear, half_extent=1, resolution=.5)
  np.testing.assert_allclose(fused, np.fmax(a, b), equal_nan=True)
