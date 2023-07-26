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

"""Visu3d API."""

from __future__ import annotations

# pylint: disable=g-import-not-at-top,g-bad-import-order,g-importing-member

# Core API
from visu3d.array_dataclass import DataclassArray

from visu3d import math
from visu3d.math import DEG2RAD
from visu3d.math import RAD2DEG

# Visualization
from visu3d import plotly
from visu3d.plotly import make_fig
from visu3d.plotly import make_traces
from visu3d.plotly import Visualizable
from visu3d.plotly.auto_plot import auto_plot_figs
from visu3d.plotly.fig_config_utils import fig_config

# Arrays implementation
from visu3d.dc_arrays.camera import Camera
from visu3d.dc_arrays.camera_spec import CameraSpec
from visu3d.dc_arrays.camera_spec import PinholeCamera
from visu3d.dc_arrays.camera_spec import Spec360
from visu3d.dc_arrays.point import Point3d
from visu3d.dc_arrays.point import Point2d
from visu3d.dc_arrays.ray import Ray
from visu3d.dc_arrays.transformation import custom_transform
from visu3d.dc_arrays.transformation import Transform

# Updating this will auto-trigger a release on PyPI and GitHub
# Note:
# * Make sure to also update the `CHANGELOG.md` before this.
# * Make sure to also trigger an `etils` release
__version__ = '1.5.3'
