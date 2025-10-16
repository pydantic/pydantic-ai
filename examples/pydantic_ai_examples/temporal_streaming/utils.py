"""Utility functions for configuration management."""

import os
from copy import copy

import yaml


def recursively_modify_api_key(conf):
    """
    Recursively replace API key placeholders with environment variable values.

    This function traverses a configuration dictionary and replaces any keys
    containing 'api_key' with the corresponding environment variable value.
    It handles nested dictionaries and lists recursively.

    Args:
        conf: The configuration dictionary to process.

    Returns:
        A copy of the configuration with API keys replaced by environment variable values.
    """

    def inner(_conf):
        for key, value in _conf.items():
            if isinstance(value, dict):
                inner(value)
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    for item in value:
                        inner(item)
                else:
                    _conf[key] = [os.environ.get(str(v), v) for v in value]
            elif isinstance(value, str):
                _conf[key] = os.environ.get(value, value)
            else:
                _conf[key] = value

    copy_conf = copy(conf)
    inner(copy_conf)
    return copy_conf


def read_config_yml(path):
    """
    Read and process a YAML configuration file.

    This function reads a YAML file, processes it to replace API key placeholders
    with environment variable values, and returns the processed configuration.

    Args:
        path: The path to the YAML configuration file.

    Returns:
        dict: The parsed and processed YAML content as a Python dictionary.
    """
    with open(path) as f:
        configs = yaml.safe_load(f)
    recursively_modify_api_key(configs)
    return configs
