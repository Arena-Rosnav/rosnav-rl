def load_lr_schedule(config: dict) -> Callable:
    lr_schedule_cfg = config["rl_agent"]["lr_schedule"]
    if lr_schedule_cfg["type"] == "linear":
        return linear_decay(**lr_schedule_cfg["settings"])
    elif lr_schedule_cfg["type"] == "square_root":
        return square_root_decay(**lr_schedule_cfg["settings"])
    else:
        raise NotImplementedError(
            f"Learning rate schedule '{lr_schedule_cfg['type']}' not implemented!"
        )