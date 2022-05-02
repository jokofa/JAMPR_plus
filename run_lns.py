#
import sys
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
from lib.lns import LNS


@hydra.main(config_path="config_lns", config_name="config")
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    if cfg.no_warnings and not sys.warnoptions:
        warnings.filterwarnings("ignore")

    lns = LNS(cfg)
    lns.load_instance(cfg.data_path)
    lns.setup(time_limit=int(cfg.time_limit))
    lns.search()


if __name__ == "__main__":
    run()
