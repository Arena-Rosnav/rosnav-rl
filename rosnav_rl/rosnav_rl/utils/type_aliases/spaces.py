from typing import Dict

import numpy as np
import torch as th

TensorDict = Dict[str, th.Tensor]

ObservationSpaceName = str
ObservationEncoding = np.ndarray

EncodedObservationDict = Dict[ObservationSpaceName, ObservationEncoding]
