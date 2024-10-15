from typing import List

import numpy as np
from gymnasium import spaces


def stack_spaces(
    *spaces_list: List[spaces.Space], frame_stacking_enabled: bool = False
) -> spaces.Space:
    low = [np.array(space.low.tolist()).flatten() for space in spaces_list]
    lower_bounds = np.concatenate(low)
    high = [np.array(space.high.tolist()).flatten() for space in spaces_list]
    upper_bounds = np.concatenate(high)

    if frame_stacking_enabled:
        lower_bounds = np.expand_dims(lower_bounds, axis=0)
        upper_bounds = np.expand_dims(upper_bounds, axis=0)

    return spaces.Box(lower_bounds, upper_bounds)
