import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .model import Connect4DQN
from .memory import ReplayMemory
import os
import logging

# Logger einrichten
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        memory_size=10000,
        tau=0.005
    ):
        self.symbol = symbol
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Netzwerke initialisieren
        self.policy_net = Connect4DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = Connect4DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer und Memory
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)

        # Hyperparameter
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

    def get_action(self, board, valid_moves, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_moves)

        with torch.set_grad_enabled(training):
            state = self._board_to_tensor(board)
            q_values = self.policy_net(state)

        # Maskierung ungültiger Züge
        valid_indices = torch.tensor(valid_moves, device=self.device)
        q_values = q_values.clone()
        q_values[:, ~torch.isin(torch.arange(q_values.size(1), device=self.device), valid_indices)] = -float('inf')
        
        return torch.argmax(q_values).item()

    def update_target_net(self):
        """Soft Update der Target Network Parameter"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

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

        states, actions, rewards, next_states, dones = self._prepare_batch()

        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self._optimize_model(loss)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path='connect4_dqn.pth'):
        """Speichert NUR die essentiellen Komponenten"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': {
                'input_dim': 42,
                'hidden_dim': 256,
                'output_dim': 7,
                'gamma': self.gamma,
                'tau': self.tau
            }
        }, path)
        logger.info(f"Modell gespeichert unter {os.path.abspath(path)}")

    def load_model(self, path='connect4_dqn.pth'):
        """Lädt die essentiellen Komponenten ohne Memory"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Modell-Datei {path} nicht gefunden")

            checkpoint = torch.load(
                path, 
                map_location='cpu', 
                weights_only=True
            )
            
            required_keys = {
                'policy_state_dict', 
                'target_state_dict',
                'optimizer_state_dict',
                'epsilon'
            }
            
            if missing_keys := required_keys - checkpoint.keys():
                raise ValueError(f"Fehlende Schlüssel: {missing_keys}")

            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']

            # Device-Handling
            self.policy_net = self.policy_net.to(self.device)
            self.target_net = self.target_net.to(self.device)
            
            # Optimizer-Parameter übertragen
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.policy_net.eval()
            logger.info(f"Modell erfolgreich geladen von {os.path.abspath(path)}")

        except Exception as e:
            logger.error(f"Kritischer Lade-Fehler: {str(e)}")
            raise

    def _prepare_batch(self):
        """Vorbereitung des Trainingsbatches"""
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).unsqueeze(1).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        )

    def _optimize_model(self, loss):
        """Führt den Optimierungsschritt durch"""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def _board_to_tensor(self, board):
        """Konvertiert das Spielbrett in einen Tensor"""
        state = np.where(np.array(board) == self.symbol, 1, 
                        np.where(np.array(board) == ' ', 0, -1))
        return torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)

    def _board_to_array(self, board):
        """Konvertiert das Spielbrett in ein numpy-Array"""
        return np.where(np.array(board) == self.symbol, 1, 
                      np.where(np.array(board) == ' ', 0, -1)).flatten()