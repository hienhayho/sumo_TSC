from tqdm import tqdm
from loguru import logger
from tqdm import tqdm
import torch
import numpy as np
from collections import deque
import torch.nn as nn
from torch.optim import Adam
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
from sumo_rl.util.logging import init_logging
import random
import time
import json

class NetWork(nn.Module):
    """Neural Network class."""

    def __init__(self, state_space, action_space):
        """Initialize neural network."""
        super(NetWork, self).__init__()
        self.fc1 = nn.Linear(state_space, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, action_space)
        self.relu = nn.ReLU()
        self.heInitilize()
        
    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def choose_action(self, state):
        state_x = torch.tensor(state["t"], dtype=torch.float32).to("cuda")
        y = self(state_x)
        return torch.argmax(y).item()
    
    def heInitilize(self):
        """He initialization."""
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

class DQN:
    """Q-learning Agent class."""

    def __init__(
        self, 
        env, 
        starting_state, 
        state_space, 
        action_space,
        reward_fn,
        trainPhase,
        max_steps=100000,
        batch_size=32,
        target_update_frequency=2,
        fill_mem_step=1000, 
        memory_size=50000,
        lr=0.0005,
        gamma=0.99,
        exploration_strategy=EpsilonGreedy()
    ):
        """Initialize Q-learning agent."""
        print(action_space.n)
        self.env = env
        self.max_epsilon = 1
        self.min_epsilon = 0.02
        self.state = starting_state["t"]
        self.state_space = state_space
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.trainPhase = trainPhase
        self.target_update_frequency = target_update_frequency
        self.action = None
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.acc_reward = 0
        self.q_net = NetWork(state_space=state_space, action_space=action_space.n).to("cuda")
        self.target_q_net = NetWork(state_space=state_space, action_space=action_space.n).to("cuda")
        self.memory_size = memory_size
        self.fill_mem_step = fill_mem_step
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        if trainPhase:
            self.saved_dir = init_logging("dqn", reward_fn=reward_fn)
    
    def fill_memory(self):
        logger.info("Starting fill...")
        memory = deque(maxlen=self.memory_size)
        state = self.env.reset()
        for step in tqdm(range(self.fill_mem_step)):
            action = self.env.action_space.sample()
            action = {"t": action}
            next_state, reward, done, info, truncated = self.env.step(action)
            experience = (state, action, reward, done, next_state, "fill")
            memory.append(experience)
            state = next_state
            if (self.trainPhase and truncated) or done["t"]:
                print(f"Step: {step} Done!")
                self.env.reset()
        logger.info(f"Fill memory: {self.fill_mem_step} samples")
        return memory
    
    def act(self, step, state):
        """Choose action based on Q-table."""
        epsilon = np.interp(step, [0, self.max_steps], [self.max_epsilon, self.min_epsilon])
        random_number = np.random.uniform(0,1)
        if random_number <= epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.q_net.choose_action(state)
        return {"t": action}
    
    def save(self):
        time_now = time.strftime("%Y%m%d")
        save_path = self.saved_dir / f"model_{time_now}.pth"
        torch.save(self.q_net.state_dict(), save_path)
        logger.info(f"Model is save at: {save_path}")
    
    @logger.catch
    def train(self, env):
        """Train agent."""
        memory = self.fill_memory()
        reward_per_episode = 0.0
        state = env.reset()
        all_rewards = []
        reward_buffer = deque(maxlen=1000)
        start = time.time()
        bar = tqdm(range(self.max_steps))
        logger.info("Start training....")
        for step in bar:
            action = self.act(step, state=state)
            next_state, reward, done, info, truncated = env.step(action)
            
            memory.append((state, action, reward, done, next_state, "train"))
            reward_per_episode += reward["t"]
            state = next_state
            
            if (self.trainPhase and truncated) or done["t"]:
                state = env.reset()
                reward_buffer.append(reward_per_episode)
                all_rewards.append((step, reward_per_episode))
                reward_per_episode = 0.0
            
            experiences = random.sample(memory, self.batch_size)
            states = [ex[0]["t"] for ex in experiences]
            actions = [ex[1]["t"] for ex in experiences]
            rewards = [ex[2]["t"] for ex in experiences]
            dones = [ex[3]["t"] for ex in experiences]
            next_states = [ex[4]["t"] for ex in experiences]

            
            states = torch.tensor(states, dtype=torch.float32).to("cuda")
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to("cuda") # (batch_size,) --> (batch_size, 1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to("cuda")
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to("cuda")
            next_states = torch.tensor(next_states, dtype=torch.float32).to("cuda")
            
            target_q_values = self.target_q_net(next_states)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * (1-dones) * max_target_q_values
            # Compute loss
            q_values = self.q_net(states)

            action_q_values = torch.gather(input=q_values, dim=1, index=actions).to("cuda")
            loss = torch.nn.functional.mse_loss(action_q_values, targets)

            # gradient descent for q-network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (step+1) % self.target_update_frequency == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # print training results
            if (step+1) % 10000 == 0:
                average_reward = np.mean(reward_buffer)
                logger.info(f'Episode: {len(all_rewards)} Step: {step+1} Average reward: {average_reward}')
        
        rewards = {
            "all_reward": all_rewards
        }
        reward_path = self.saved_dir / "reward.json"
        with open(reward_path, "w") as f:
            json.dump(reward, f, indent=4)
        
        end = time.time()
        logger.info(f"Time: {end - start}")
        logger.info(f"Reward was saved at: {str(reward_path)}")
        logger.info("Finish training!")
        self.save()
    
    @logger.catch
    def play(self):
        state = self.env.reset()
        done = {}
        done["t"] = False
        all_reward = 0.0
        while not done["t"]:
            action = self.q_net.choose_action(state)
            action = {"t": action}
            next_state, reward, done, info, truncated = self.env.step(action)
            print("next_state: ", next_state)
            input()
            state = next_state
            all_reward += reward["t"]
        print(all_reward)