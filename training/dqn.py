import argparse
import os
import sys
from datetime import datetime
import torch

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent, DQN
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
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
        default="wait",
        required=False,
        help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.\n",
    )
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("-train", action="store_true", default=False, help="Number of runs.\n")
    prs.add_argument("-model", dest="model")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split(".")[0]
    out_csv = f"outputs/2way-single-intersection/{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}_reward{args.reward}"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        reward_fn=args.reward,
        sumo_warnings=False,
    )
    model = DQN(
        env=env,
        starting_state=env.reset(),
        state_space=13,
        action_space=env.action_space,
        reward_fn=args.reward,
        trainPhase=args.train
    )
    
    if args.train:
        model.train(env=env)
    else:
        model.q_net.load_state_dict(torch.load(args.model))
        model.q_net.eval()
        model.play()
    # for run in range(1, args.runs + 1):
    #     initial_states = env.reset()

    #     done = {"__all__": False}
    #     infos = []
    #     if args.fixed:
    #         while not done["__all__"]:
    #             _, _, done, _ = env.step({})
    #     else:
    #         while not done["__all__"]:
    #             actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

    #             s, r, done, _ = env.step(action=actions)
    #             print(s)
    #             for agent_id in ql_agents.keys():
    #                 # print("agentid: ", agent_id)
    #                 # print("encode: ", env.encode(s[agent_id], agent_id))
    #                 # input()
    #                 ql_agents[agent_id].learn(next_state=s[agent_id], reward=r[agent_id])
    #     env.save_csv(out_csv, run)
    #     env.close()
