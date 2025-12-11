

import importlib
import importlib.util
import os


def load_configs_from_file(config_file_path: str, configs_var_name: str):
    """
    Load the 'configs' list from a specified Python file.

    Args:
        config_file_path: Path to the Python file containing configs

    Returns:
        The configs list from the specified file
    """
    # Convert to absolute path if relative
    config_file_path = os.path.abspath(config_file_path)

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("custom_configs", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, configs_var_name):
        raise AttributeError(f"Config file {config_file_path} does not contain a '{configs_var_name}' variable")

    return getattr(config_module, configs_var_name)