import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.optim import Adam
from collections import deque
from abc import ABC, abstractmethod

device = "cuda" if torch.cuda.is_available() else "cpu"
from sumo_rl.agents.base_net import build_net
from sumo_rl.util.logging import init_logging, general_logging

class BaseAgent(ABC):
    """Double Deep Q-learning Agent class."""

    def __init__(
        self,
        model_name,
        env,
        starting_state,
        state_space,
        action_space,
        reward_fn,
        trainPhase,
        net_type,
        max_epsilon=1,
        min_epsilon=0.02,
        max_steps=100000,
        batch_size=32,
        target_update_frequency=500,
        fill_mem_step=1000,
        memory_size=50000,
        lr=0.0005,
        gamma=0.99,
    ):
        """Initialize Q-learning agent."""
        self.model_name = model_name
        self.env = env
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.state = starting_state["t"]
        self.state_space = state_space
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.trainPhase = trainPhase
        self.net_type = net_type
        self.target_update_frequency = target_update_frequency
        self.action = None
        self.gamma = gamma
        self.acc_reward = 0
        self.memory_size = memory_size
        self.fill_mem_step = fill_mem_step
        
        self.q_net = build_net(net_type=net_type, state_space=state_space, action_space=action_space.n).to(device)
        self.target_q_net = build_net(net_type=net_type, state_space=state_space, action_space=action_space.n).to(device)
        
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        if trainPhase:
            self.saved_dir = init_logging(self.model_name, reward_fn=reward_fn)
            kwargs = {
                "max_epsilon": max_epsilon,
                "min_epsilon": min_epsilon,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "target_update_frequency": target_update_frequency,
                "fill_mem_step": fill_mem_step,
                "memory_size": memory_size,
                "lr": lr,
                "gamma": gamma,
                "net_type": net_type
            }
            general_logging(self.model_name, kwargs)

    
    def fill_memory(self):
        logger.info(f"{self.model_name} - Starting fill {self.fill_mem_step} samples to memory ...")
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
                print(f"{self.model_name} - Step: {step} Done!")
                self.env.reset()
        logger.info(f"{self.model_name} - Fill memory: {self.fill_mem_step} samples")
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
        logger.info(f"{self.model_name} - Model is save at: {save_path}")
    
    @abstractmethod
    def compute_loss(self, actions, states, next_states, rewards, dones):
        raise NotImplementedError()
    
    @logger.catch
    def train(self):
        """Train agent."""
        memory = self.fill_memory()
        reward_per_episode = 0.0
        state = self.env.reset()
        all_rewards = []
        reward_buffer = deque(maxlen=1000)
        start = time.time()
        bar = tqdm(range(self.max_steps))
        logger.info(f"{self.model_name} - Start training....")
        for step in bar:
            action = self.act(step, state=state)
            next_state, reward, done, info, truncated = self.env.step(action)
            memory.append((state, action, reward, done, next_state, "train"))
            reward_per_episode += reward["t"]
            state = next_state
            
            if (self.trainPhase and truncated) or done["t"]:
                state = self.env.reset()
                reward_buffer.append(reward_per_episode)
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
            
            # Compute loss
            loss = self.compute_loss(actions=actions, states=states, next_states=next_states, rewards=rewards, dones=dones)

            # gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (step+1) % self.target_update_frequency == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # print training results
            if (step+1) % 5000 == 0:
                average_reward = np.mean(reward_buffer)
                all_rewards.append(average_reward)
                logger.info(f'{self.model_name} - Step: {step+1} - {self.reward_fn}: {average_reward}')
        
        rewards = {
            "all_reward": all_rewards
        }
        reward_path = self.saved_dir / "reward.json"
        with open(reward_path, "w") as f:
            json.dump(rewards, f, indent=4)
        
        end = time.time()
        logger.info(f"{self.model_name} - Time: {end - start}")
        logger.info(f"{self.model_name} - Reward was saved at: {str(reward_path)}")
        logger.info(f"{self.model_name} - Finish training!")
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
            state = next_state
            all_reward += reward["t"]
        print(all_reward)