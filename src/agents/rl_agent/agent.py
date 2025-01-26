import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .model import Connect4DQN
from .memory import ReplayMemory

class RLAgent:
    def __init__(
        self,
        symbol,
        input_dim=42,
        hidden_dim=256,
        output_dim=7,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        lr=0.001,
        memory_size=10000
    ):
        self.symbol = symbol
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = Connect4DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = Connect4DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
    
    def get_action(self, board, valid_moves, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_moves)
        
        state = self._board_to_tensor(board)
        with torch.no_grad():
            q_values = self.policy_net(state)
        
        mask = torch.zeros_like(q_values)
        mask[0, valid_moves] = 1
        q_values = q_values + (1 - mask) * -1e7  # Mask invalid moves
        
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(
            self._board_to_array(state),
            action,
            reward,
            self._board_to_array(next_state),
            done
        )
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _board_to_tensor(self, board):
        state = np.where(np.array(board) == self.symbol, 1, 
                        np.where(np.array(board) == ' ', 0, -1))
        return torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
    
    def _board_to_array(self, board):
        return np.where(np.array(board) == self.symbol, 1, 
                       np.where(np.array(board) == ' ', 0, -1)).flatten()
    
    def save_model(self, path='connect4_dqn.pth'):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path='connect4_dqn.pth'):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']