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

"""Math utils."""

from typing import Optional

import dataclass_array as dca
from dataclass_array.typing import DcT
import numpy as np


def subsample(
    array: DcT,
    *,
    num_samples: Optional[int],
    seed: Optional[int] = None,
) -> DcT:
  """Subsample the given DataclassArray."""
  if not isinstance(array, dca.DataclassArray):
    raise TypeError(f'Subsample only accept DataclassArray. Got: {type(array)}')

  if num_samples in (None, -1) or array.size <= num_samples:
    return array  # No-op

  rng = np.random.default_rng(seed)
  idx = rng.choice(array.size, size=num_samples, replace=False)

  array = array.flatten()
  array = array[idx]
  return array
