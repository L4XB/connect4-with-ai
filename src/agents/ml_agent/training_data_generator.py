import pickle
from src.constants import *
from src.agents.mini_max_agent import MiniMaxAgent
from src.enviroment.game_board import GameBoard
from tqdm import tqdm


def board_to_vector(board, symbol):
    vec = []
    for row in board:
        for cell in row:
            if cell == symbol: vec.append(1)
            elif cell == " ": vec.append(0)
            else: vec.append(-1)
    return vec

def generate_data(num_games=1000, depth=4):
    agent1 = MiniMaxAgent(depth)
    agent1.set_symbol(PLAYER_ONE_SYMBOL)
    agent2 = MiniMaxAgent(depth)
    agent2.set_symbol(PLAYER_TWO_SYMBOL)

    data = []
    for _ in tqdm(range(num_games), desc="Generating games"):
        board = GameBoard()
        game_history = []
        current_agent = agent1

        while True:
            state = board_to_vector(board.board, current_agent.symbol)
            move = current_agent.get_move(board.board)
            game_history.append((state, move))

            board.insert_token(move, current_agent.symbol)
            if board.check_winner(current_agent.symbol) or board.is_draw():
                break

            current_agent = agent2 if current_agent == agent1 else agent1

        data.extend(game_history)

    with open(f"connect4_data_{depth}d_{num_games}g.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"Generated {len(data)} samples")


generate_data(num_games=5000, depth=3)