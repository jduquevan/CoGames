import hydra
import os
import sys

from omegaconf import DictConfig, OmegaConf

from coin_game import CoinGame


@hydra.main(config_path="../scripts", config_name="config")
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    print(args.agent.gamma)

if __name__ == "__main__":
    main()