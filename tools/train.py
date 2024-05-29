import argparse
import os
import sys
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import build_model


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Training Q-Learing based Method"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=10, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument(
        "-r",
        dest="reward",
        type=str,
        default="diff-waiting-time",
        choices=["diff-waiting-time", "average-speed", "queue", "pressure"],
        required=False,
        help="Reward function.\n",
    )
    prs.add_argument(
        "-name", 
        dest="name",
        default="dqn",
        choices=["dqn", "double_dqn", "dueling_dqn", "double_dueling_dqn"], 
        required=False, 
        help="Model name.\n"
    )
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("-train", action="store_true", default=False, help="Number of runs.\n")
    prs.add_argument("-model", dest="model", type=str, help="Model path (ONLY for test)")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split(".")[0]
    out_csv = f"outputs/2way-single-intersection/{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}_reward{args.reward}"

    gui_enable = False
    if not args.train:
        gui_enable = True
    
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=gui_enable,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        reward_fn=args.reward,
        sumo_warnings=False,
    )
    
    cfg = {
        "model_name": args.name,
        "env": env,
        "starting_state": env.reset(),
        "action_space": env.action_space,
        "trainPhase": args.train,
        "reward_fn": args.reward,
    }
    
    model = build_model(cfg)
    
    model.train()