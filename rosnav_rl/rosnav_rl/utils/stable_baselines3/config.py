def check_batch_size(n_envs: int, batch_size: int, mn_batch_size: int) -> None:
    """
    Validates the batch size against the number of environments and mini batch size.

    Parameters:
    n_envs (int): Number of environments.
    batch_size (int): The total batch size.
    mn_batch_size (int): The mini batch size.

    Raises:
    ValueError: If any of the following conditions are met:
        - The mini batch size is greater than the batch size.
        - The batch size is not divisible by the mini batch size.
        - The batch size is not divisible by the number of environments.
    """
    errors = []

    if batch_size < mn_batch_size:
        errors.append(
            f"Mini batch size {mn_batch_size} is bigger than batch size {batch_size}"
        )

    if batch_size % mn_batch_size != 0:
        errors.append(
            f"Batch size {batch_size} isn't divisible by mini batch size {mn_batch_size}"
        )

    if batch_size % n_envs != 0:
        errors.append(f"Batch size {batch_size} isn't divisible by n_envs {n_envs}")

    if errors:
        raise ValueError(" | ".join(errors))
