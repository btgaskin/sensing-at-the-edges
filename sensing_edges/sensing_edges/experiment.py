"""Hydra entry point for sensing_edges training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .train import train


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
