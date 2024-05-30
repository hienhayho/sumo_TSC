import torch
import torch.nn as nn

allow_net_type = [
    "dqn",
    "duel"
]

def build_net(net_type, state_space, action_space):
    assert net_type in allow_net_type, f"Net type: {net_type} is not allowed."
    if net_type == "dqn":
        return DQNNetWork(state_space=state_space, action_space=action_space)
    elif net_type == "duel":
        return DuelingNetwork(state_space=state_space, action_space=action_space)

class DQNNetWork(nn.Module):
    """Neural Network class."""

    def __init__(self, state_space, action_space):
        """Initialize neural network."""
        super(DQNNetWork, self).__init__()
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
        
class DuelingNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(DuelingNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        # Separate streams for value and advantage
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_space)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combining value and advantage into Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def choose_action(self, state):
        state = torch.tensor(state["t"], dtype=torch.float32).unsqueeze(0).to("cuda")
        q_values = self(state)
        action = torch.argmax(q_values, dim=1).item()
        return action