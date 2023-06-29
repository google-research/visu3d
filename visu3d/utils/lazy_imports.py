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

"""Lazy imports."""

from __future__ import annotations

import typing

from visu3d.utils import py_utils

if typing.TYPE_CHECKING:
  # Plotly is an optional (but strongly recommended on colab) dependency
  import IPython  # pytype: disable=import-error
  from plotly import graph_objects as plotly_go
  from plotly import basedatatypes as plotly_base
  import scipy
else:
  # Hack: Use `globals` as a hack to avoid VS Code to infer go as
  # `go | LazyModule`
  globals().update(
      IPython=py_utils.LazyModule('IPython'),
      plotly_go=py_utils.LazyModule('plotly.graph_objects'),
      plotly_base=py_utils.LazyModule('plotly.basedatatypes'),
      scipy=py_utils.LazyModule('scipy'),
  )
