import numpy as np
import torch as th

TensorDict = dict[str, th.Tensor]

ObservationSpaceName = str
ObservationEncoding = np.ndarray

EncodedObservationDict = dict[ObservationSpaceName, ObservationEncoding]
