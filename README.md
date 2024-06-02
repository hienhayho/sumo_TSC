# Deep Q-learning - SUMO RL

This repo provide necessary code for running SUMO with DQN and its variants.

Now it supports:
- DQN
- Double DQN
- Dueling DQN
- Double Dueling DQN (D3QN)

## Installation

You can refer to [sumo_rl](https://github.com/LucasAlegre/sumo-rl?tab=readme-ov-file#install) or follow these instructions to install dependencies:

```bash
git clone https://github.com/hienhayho/sumo_TSC.git

cd sumo_TSC/

chmod +x install.sh

bash ./install.sh
```
> If it shows: **Permission denied**, run: `sudo bash ./install.sh`

## Usage

### Build Agent and Reward Function

There are 4 agents available, you can implement more agents and import it to [sumo_rl/agents/\__init\__.py](sumo_rl/agents/__init__.py). Make sure you add it to `build_model` function as well.

- `model_name`: name of `-name` flag in [tools/train.py](tools/train.py)
- `net_type`: type of core network. There are two available networks in [sumo_rl/agents/base_net.py](sumo_rl/agents/base_net.py), you can implement your own networks and add to `build_net` function.

To add new reward function, please refer to [sumo_rl/environment/traffic_signal.py](sumo_rl/environment/traffic_signal.py).

### Training Agent
This work is automatically run with single intersection environment in SUMO, you can refer to [here](https://github.com/LucasAlegre/sumo-rl/tree/main/sumo_rl/nets) to change environments.

For training, you can refer to [tools/train.py](tools/train.py). 

* Here an example for traing `dqn` with `diff-waiting-time` reward function:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -name dqn -s 10000
```

* Training AIO (all in one):
```bash
apt-get install make -y
make train
```
> If it shows: **Permission denied**, run: `sudo apt-get install make -y`

Model and logs is at `outputs/<model_name>/<reward_fn>_<time>`

### Demo
```bash
CUDA_VISIBLE_DEVICES=0 python tools/play.py -name <model_name> -r <reward_fn> -model <model_path>
```

Example:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/play.py -name dqn -r diff-waiting-time -model outputs/dqn/diff-waiting-time_2024_05_28_20_10_14/model_20240528.pth
```

## License

This code is developed from [sumo_rl](https://github.com/LucasAlegre/sumo-rl) by `hienhayho`.