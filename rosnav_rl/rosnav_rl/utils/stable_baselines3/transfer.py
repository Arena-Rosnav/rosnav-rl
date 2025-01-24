import re
from typing import List, Dict
from warnings import warn

import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from pprint import pprint


def transfer_weights(
    target_model: ActorCriticPolicy,
    source_model: ActorCriticPolicy,
    include: List[str] = None,
    exclude: List[str] = None,
) -> ActorCriticPolicy:
    """
    Transfers weights from source_model to target_model based on specified inclusion and exclusion criteria.

    Args:
        target_model (ActorCriticPolicy): The target model to which weights will be transferred.
        source_model (ActorCriticPolicy): The source model from which weights will be transferred.
        include (List[str], optional): List of regex patterns for keys to include in the transfer. If None, no weights will be transferred.
        exclude (List[str], optional): List of substrings for keys to exclude from the transfer. Defaults to ["---"].

    Returns:
        ActorCriticPolicy: The target model (target_model) with updated weights.

    Raises:
        UserWarning: If no include list is provided, a warning is issued and no weights are transferred.
    """
    if include is None:
        warn("No include list provided. Skipping weight transfer.")
        return target_model

    exclude = exclude or ["---"]

    state_dict_target_model: Dict[str, torch.Tensor] = target_model.state_dict()
    state_dict_source_model: Dict[str, torch.Tensor] = source_model.state_dict()

    layers_dict = {
        key: value
        for key, value in state_dict_source_model.items()
        if any(re.match(_key, key) for _key in include)
        and not any(item in key for item in exclude)
        and key in state_dict_target_model
        and state_dict_target_model[key].shape == value.shape
    }

    print(f"Transferring weights for {len(layers_dict.keys())} keys!")
    pprint(list(layers_dict.keys()))

    state_dict_target_model.update(layers_dict)
    target_model.load_state_dict(state_dict_target_model, strict=True)

    return target_model
