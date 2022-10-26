import hydra
import os
import sys

from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional

from coin_game import CoinGame
from agents import DQNAgent


@hydra.main(config_path="../scripts", config_name="config")
def main(args: DictConfig):
    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
    print(config)



if __name__ == "__main__":
    main()