from typing import Any, Dict, List
import inspect

from rosnav_rl.spaces import BaseObservationSpace


def extract_init_arguments(
    classes: List[BaseObservationSpace],
) -> Dict[str, Dict[str, str]]:
    """
    Extracts the arguments and their types from the __init__ method of each class in the provided list.

    Args:
        classes (list): A list of classes to inspect.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where keys are class names and values are dictionaries
                                    of argument names and their types.
    """
    init_args = {}

    for cls in classes:
        # Get the signature of the __init__ method
        init_method = cls.__init__
        signature = inspect.signature(init_method)

        # Extract parameter names and their types
        params = {}
        for param in signature.parameters.values():
            if param.name != "self":  # Skip 'self'
                params[param.name] = (
                    str(param.annotation)
                    if param.annotation is not inspect.Parameter.empty
                    else "Any"
                )

        init_args[cls.__name__] = params

    return init_args


def find_missing_keys(lead_dict: Dict, target_dict: Dict) -> List[str]:
    """
    Compares the keys of two dictionaries and identifies which keys from the lead
    dictionary are missing in the target dictionary.

    Args:
        lead_dict (Dict): The dictionary containing the required keys.
        target_dict (Dict): The dictionary to compare against.

    Returns:
        List[str]: A list of keys that are present in lead_dict but missing in target_dict.
    """
    # Get the set of keys from both dictionaries
    lead_keys = set(lead_dict.keys())
    target_keys = set(target_dict.keys())

    # Find missing keys
    missing_keys = lead_keys - target_keys

    return list(missing_keys)
