# Copyright 2026 The visu3d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for transformation."""

"""Regression tests for finite look-at frames parallel to world up."""

import numpy as np
import pytest
import visu3d as v3d


def check_frame(transform, pos, target):
  transform = transform.as_np()
  np.testing.assert_allclose(transform.t, pos, atol=1e-6)
  rotation = np.asarray(transform.R)
  assert np.isfinite(rotation).all()
  expected_direction = target - pos
  expected_direction = expected_direction / np.linalg.norm(
      expected_direction, axis=-1, keepdims=True
  )
  np.testing.assert_allclose(
      rotation[..., :, 2], expected_direction, rtol=1e-6, atol=1e-6
  )
  np.testing.assert_allclose(
      np.swapaxes(rotation, -1, -2) @ rotation,
      np.broadcast_to(np.eye(3), rotation.shape),
      rtol=1e-6,
      atol=1e-6,
  )
  np.testing.assert_allclose(np.linalg.det(rotation), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('sign', [-1.0, 1.0])
@pytest.mark.parametrize('pos', [np.zeros(3), np.array([2.0, -3.0, 5.0])])
def test_vertical_target_has_finite_right_handed_frame(sign, pos):
  target = pos + np.array([0.0, 0.0, sign * 2.0])
  transform = v3d.Transform.from_look_at(pos=pos, target=target)
  check_frame(transform, pos, target)
  np.testing.assert_allclose(
      (transform.inv @ transform).matrix4x4, np.eye(4), rtol=1e-6, atol=1e-6
  )


def test_mixed_batch_preserves_ordinary_orientations():
  pos = np.array(
      [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
  )
  target = pos + np.array(
      [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 2.0, 3.0], [2.0, -1.0, 0.5]]
  )
  transform = v3d.Transform.from_look_at(pos=pos, target=target)
  check_frame(transform, pos, target)
  forward = target[2:] - pos[2:]
  forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
  width = np.cross(forward, [0.0, 0.0, 1.0])
  width /= np.linalg.norm(width, axis=-1, keepdims=True)
  expected = np.stack([width, np.cross(forward, width), forward], axis=-1)
  np.testing.assert_allclose(
      transform.as_np().R[2:], expected, rtol=1e-6, atol=1e-6
  )


@pytest.mark.parametrize('sign', [-1.0, 1.0])
def test_jax_jit_and_vertical_frame_gradients_are_finite(sign):
  import jax
  import jax.numpy as jnp

  pos = jnp.array([1.0, 2.0, 3.0])
  target = pos + jnp.array([0.0, 0.0, sign * 2.0])
  fn = lambda target: v3d.Transform.from_look_at(pos=pos, target=target)
  check_frame(jax.jit(fn)(target), np.asarray(pos), np.asarray(target))
  gradient = jax.jit(jax.jacrev(lambda target: fn(target).R))(target)
  assert np.isfinite(gradient).all()


def test_torch_vertical_frame_remains_differentiable():
  import torch

  pos = torch.tensor([1.0, 2.0, 3.0])
  target = torch.tensor([1.0, 2.0, 5.0], requires_grad=True)
  transform = v3d.Transform.from_look_at(pos=pos, target=target)
  assert torch.isfinite(transform.R).all()
  (transform.R * torch.arange(9.0).reshape(3, 3)).sum().backward()
  assert torch.isfinite(target.grad).all()


def test_regular_targets_keep_existing_world_up_convention():
  pos = np.zeros(3)
  target = np.array([1.0, 2.0, 3.0])
  tr = v3d.Transform.from_look_at(pos=pos, target=target)
  check_frame(tr, pos, target)
  assert abs(float(tr.as_np().R[2, 0])) < 1e-7