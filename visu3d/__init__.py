# Copyright 2022 The visu3d Authors.
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

"""Visu3d API."""

from __future__ import annotations

import sys

# pylint: disable=g-import-not-at-top,g-bad-import-order

pytest = sys.modules.get('pytest')
if pytest:
  # Inside tests, rewrite `assert` statement in `v3d.testing` for better
  # debug messages
  pytest.register_assert_rewrite('visu3d.testing')

# Core API
from visu3d import typing
from visu3d.array_dataclass import array_field
from visu3d.array_dataclass import DataclassArray
from visu3d.ops import stack
from visu3d.utils import lazy_imports
from visu3d.vectorization import vectorize_method

# Visualization
from visu3d import plotly
from visu3d.plotly import make_fig
from visu3d.plotly import make_traces
from visu3d.plotly import Visualizable
from visu3d.plotly.auto_plot import auto_plot_figs
from visu3d.plotly.fig_config_utils import fig_config
from visu3d.utils.rotation_utils import DEG2RAD
from visu3d.utils.rotation_utils import RAD2DEG

# Arrays implementation
from visu3d.dc_arrays.camera import Camera
from visu3d.dc_arrays.camera_spec import CameraSpec
from visu3d.dc_arrays.camera_spec import PinholeCamera
from visu3d.dc_arrays.point import Point3d
from visu3d.dc_arrays.point import Point2d
from visu3d.dc_arrays.ray import Ray
from visu3d.dc_arrays.transformation import custom_transform
from visu3d.dc_arrays.transformation import Transform

# Inside tests, can use `v3d.testing`
if pytest:  # < Ensure open source does not trigger import
  try:
    from visu3d import testing
  except ImportError:
    pass

__version__ = '1.0.0'

del sys, pytest
