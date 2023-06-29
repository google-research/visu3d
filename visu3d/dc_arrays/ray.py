# Copyright 2023 The visu3d Authors.
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

"""Ray utils."""

from __future__ import annotations

import dataclasses

import dataclass_array as dca
from dataclass_array.typing import Axes
from etils import enp
from etils.array_types import Array, FloatArray  # pylint: disable=g-multiple-import
from visu3d import array_dataclass
from visu3d import plotly
from visu3d.dc_arrays import transformation
from visu3d.plotly import fig_config_utils
from visu3d.utils import np_utils
from visu3d.utils.lazy_imports import plotly_base

# TODO(epot):
# * Make the dir optional to allow:
#   ray = Ray(pos=[0, 0, 0]).look_at(target)
# * Check broadcasting for `+` and `*`


class Ray(array_dataclass.DataclassArray):
  """6d vector with position and direction.

  Note: The direction is not normalized by default.

  Attributes:
    pos: Position
    dir: Direction
  """

  pos: FloatArray['*shape 3']
  dir: FloatArray['*shape 3']

  # Overwrite `v3d.DataclassArray.fig_config`.
  fig_config: plotly.TraceConfig = dataclasses.field(
      default=plotly.TraceConfig(
          num_samples=fig_config_utils.LazyValue(
              lambda fig_config: fig_config.num_samples_ray
          )
      ),
      repr=False,
      init=False,
  )

  @property
  def end(self) -> FloatArray['*shape 3']:
    """The extremity of the ray (`ray.pos + ray.dir`)."""
    return self.pos + self.dir

  @classmethod
  @enp.check_and_normalize_arrays(strict=False)
  def from_look_at(
      cls,
      *,
      pos: Array['*d 3'],
      target: Array['*d 3'],
  ) -> Ray:
    """Factory to create a look at Ray.

    Alias of `Ray(pos=pos, dir=target-from)`.

    Args:
      pos: Origin position
      target: Target position

    Returns:
      The Ray.
    """
    return cls(
        pos=pos,
        dir=target - pos,
    )

  def __add__(self, translation: FloatArray['... 3']) -> Ray:
    """Translate the position."""
    if isinstance(translation, Ray):
      raise TypeError(
          'Cannot add Ray with Ray. '
          'In `ray + x`: x should be a FloatArray[..., 3].'
      )
    translation = self.xnp.asarray(translation)
    return self.replace(pos=self.pos + translation)

  __sub__ = np_utils.__sub__

  def scale_dir(self, scale: FloatArray['...']) -> Ray:
    """Scale the dir."""
    if isinstance(scale, Ray):
      raise TypeError(
          'Cannot multiply Ray with Ray. '
          'In `ray * x`: x should be a scalar factor.'
      )
    scale = self.xnp.asarray(scale)
    return self.replace(dir=self.dir * scale)

  def norm(self, keepdims: bool = False) -> FloatArray['*shape']:
    """Returns the norm of the dir."""
    return enp.compat.norm(self.dir, axis=-1, keepdims=keepdims)

  def normalize(self) -> Ray:
    """Normalize the directions."""
    return self.replace(dir=enp.linalg.normalize(self.dir))

  def mean(self, *, axis: Axes = None) -> Ray:
    """Returns the average ray."""
    # Mean reduce across all axis but the last one
    axis = self._to_absolute_axis(axis)
    if axis == ():  # Pytorch do not support `axis=()`  # pylint: disable=g-explicit-bool-comparison
      return self
    return self.map_field(lambda t: t.mean(axis=axis))

  def look_at(self, target: Array['*shape 3']) -> Ray:
    """Change the direction to point to the target point."""
    # Could add a `keep_norm=True` ?
    target = dca.utils.np_utils.asarray(target, xnp=self.xnp)
    return self.replace(dir=target - self.pos)

  # Protocols (inherited)

  def apply_transform(self, tr: transformation.Transform) -> Ray:
    return self.replace(
        pos=tr.apply_to_pos(self.pos),
        dir=tr.apply_to_dir(self.dir),
    )

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    start = self.pos
    end = self.end
    return plotly.make_lines_traces(
        start=start,
        end=end,
        end_marker='diamond',
        axis=-1,
    )
