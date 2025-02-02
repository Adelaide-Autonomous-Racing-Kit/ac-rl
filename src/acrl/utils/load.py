from typing import Dict

from ruamel.yaml import YAML


def yaml(path: str) -> Dict:
    _yaml = YAML()
    with open(path) as file:
        params = _yaml.load(file)
    return params
