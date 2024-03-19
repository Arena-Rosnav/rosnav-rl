import numpy as np


def max_abs_scaling(observation_arr, min_value, max_value):
    denominator = max_value - min_value
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return (2 * (observation_arr - min_value)) / denominator - 1
