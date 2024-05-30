import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from configs.config import get_config

from .base_net import DQNNetWork, DuelingNetwork

from .dqn import DQN
from .double_dqn import DoubleDQN
from .dueling_dqn import DuelingDQN
from .double_dueling_dqn import DoubleDuelingDQN

allow_model_names = [
    "dqn",
    "double_dqn",
    "dueling_dqn",
    "double_dueling_dqn"
]


def build_model(args):
    config = get_config("configs/config.yaml")
    assert args["model_name"] in allow_model_names, f"Model with name: {args['model_name']} was not defined."
    if args["model_name"] == "dqn":
        agent = DQN
        args["net_type"] = "dqn"
    elif args["model_name"] == "double_dqn":
        agent = DoubleDQN
        args["net_type"] = "dqn"
    elif args["model_name"] == "dueling_dqn":
        agent = DuelingDQN
        args["net_type"] = "duel"
    elif args["model_name"] == "double_dueling_dqn":
        agent = DoubleDuelingDQN
        args["net_type"] = "duel"

    return agent(**args, **config["model"])
