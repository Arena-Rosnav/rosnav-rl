import re
from typing import List, Dict
from warnings import warn

import torch


def transfer_weights(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    include: List[str] = None,
    exclude: List[str] = None,
) -> torch.nn.Module:
    """
    Transfers weights from model2 to model1 based on specified include and exclude lists.

    Args:
        model1 (torch.nn.Module): The target model to which weights will be transferred.
        model2 (torch.nn.Module): The source model from which weights will be transferred.
        include (List[str], optional): List of regex patterns for parameter names to include in the transfer.
        exclude (List[str], optional): List of substrings for parameter names to exclude from the transfer.

    Returns:
        torch.nn.Module: The target model (model1) with updated weights.

    Raises:
        Warning: If no include list is provided, a warning is issued and weight transfer is skipped.
    """
    if include is None:
        warn("No include list provided. Skipping weight transfer.")
        return model1

    exclude = exclude or ["---"]

    state_dict_model1: Dict[str, torch.Tensor] = model1.state_dict()
    state_dict_model2: Dict[str, torch.Tensor] = model2.state_dict()

    weights_dict = {
        key: value
        for key, value in state_dict_model2.items()
        if any(re.match(_key, key) for _key in include)
        and not any(item in key for item in exclude)
        and key in state_dict_model1
        and state_dict_model1[key].shape == value.shape
    }

    print(f"Transferring weights for {len(weights_dict.keys())} keys!")

    state_dict_model1.update(weights_dict)
    model1.load_state_dict(state_dict_model1, strict=True)

    return model1
