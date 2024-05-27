import time
from loguru import logger
from datetime import datetime
from pathlib import Path


def init_logging(model: str, reward_fn: str):
    par_folder = Path("outputs")
    date = f"{reward_fn}_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    saved_dir = par_folder / Path(model) / Path(date)
    saved_dir.mkdir(parents=True, exist_ok=True)
    file_name = saved_dir / Path("training.log")
    logger.add(str(file_name))
    return saved_dir