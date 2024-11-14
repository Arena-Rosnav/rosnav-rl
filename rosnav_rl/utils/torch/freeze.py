import re
from typing import List
from warnings import warn

import torch


def freeze_weights(
    model: torch.nn.Module,
    include: List[str] = None,
    exclude: List[str] = None,
) -> torch.nn.Module:
    """
    Freeze weights of a torch.nn.Module model based on include/exclude patterns.

    Args:
        model: PPO model to freeze weights for
        include: List of regex patterns to match parameter names to freeze
        exclude: List of patterns to exclude from freezing

    Returns:
        PPO model with frozen weights
    """
    if include is None:
        warn("No include list provided. Skipping weight freezing.")
        return model

    exclude = exclude or ["---"]

    state_dict_model = model.state_dict()

    weights_dict = {
        key: value
        for key, value in state_dict_model.items()
        if any(re.match(_key, key) for _key in include)
        and not any(item in key for item in exclude)
    }

    print(f"Freezing weights for {len(weights_dict.keys())} keys!")

    for name, param in model.named_parameters():
        if name in weights_dict.keys():
            param.requires_grad = False

    return model
