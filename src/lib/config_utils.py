from omegaconf import DictConfig


def flatten(config: DictConfig, prefix="", flat_config=None):
    if flat_config is None:
        flat_config = {}

    for key, value in config.items():
        if isinstance(value, DictConfig):
            flatten(value, "", flat_config)
        else:
            flat_config[prefix + key] = value

    return flat_config


def flatten_config(config: DictConfig) -> DictConfig:
    flat_config = flatten(config)
    return DictConfig(flat_config)
