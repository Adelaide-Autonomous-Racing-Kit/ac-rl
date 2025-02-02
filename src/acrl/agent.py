import numpy as np

from aci.interface import AssettoCorsaInterface
from acrl.utils import load


class SACAgent(AssettoCorsaInterface):
    def __init__(self, config_path: str):
        self.cfg = load.yaml(config_path)
        super().__init__(self.cfg["aci"])
        self.setup()

    def behaviour(self, observation: Dict) -> np.array:
        return np.array([0.0, 0.0, 1.0])

    def teardown(self):
        pass

    def termination_condition(self, observation: Dict) -> bool:
        return False

    def restart_condition(self, observation: Dict) -> bool:
        return False

    def setup(self):
        pass
