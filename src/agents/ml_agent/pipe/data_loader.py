import pickle
import torch
from torch.utils.data import Dataset

class Connect4Dataset(Dataset):
    def __init__(self, data_paths):
        '''
        the method [__init__] initializes the dataset by loading data from the specified paths.
        '''
        
        self.data = []
        
        # load custom data
        for path in data_paths:
            print(f"Loading data from {path}...")
            with open(path, "rb") as f:
                self.data += pickle.load(f)
            print(f"Loaded {len(self.data)} samples so far.")

        print(f"Total dataset size: {len(self.data)}")


    def __len__(self):
        '''
        the method [__len__] returns the total number of samples in the dataset.
        '''
        
        return len(self.data)


    def __getitem__(self, idx):
        '''
        the method [__getitem__] retrieves a sample from the dataset at the specified index [idx].
        It converts the board state to a 2D tensor format and returns it along with the move.
        '''
        
        state, move = self.data[idx]
        
        # Convert the board to a 2D tensor format
        state_tensor = self.board_to_tensor(state)
        return state_tensor, torch.tensor(move, dtype=torch.long)


    def convert_uci_to_board(self, features):
        '''
        the method [convert_uci_to_board] converts UCI data (1D array) to the expected 2D board format.
        '''
        
        board = []
        for row in range(6):
            board_row = []
            for col in range(7):
                cell = features[row * 7 + col]
                if cell == 0:
                    board_row.append(' ')  # Empty cell
                elif cell == 1:
                    board_row.append('●')  # Player 1
                elif cell == 2:
                    board_row.append('○')  # Player 2
            board.append(board_row)
        return board


    def board_to_tensor(self, state):
        '''
        the method [board_to_tensor] converts the board state to a tensor format.
        It handles both 2D and flattened state formats.
        '''
        
        # Handle both 2D and flattened state formats
        if not isinstance(state[0], list):  # Flattened state (e.g., from generated data)
            state = [state[i:i+7] for i in range(0, len(state), 7)]

        # Convert numerical values to channels
        channel_self = [[1.0 if cell == 1 else 0.0 for cell in row] for row in state]
        channel_opp = [[1.0 if cell == -1 else 0.0 for cell in row] for row in state]
        return torch.FloatTensor([channel_self, channel_opp])