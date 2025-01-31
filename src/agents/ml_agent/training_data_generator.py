import pickle
from tqdm import tqdm
from src.constants import AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL
from src.agents.smart_agent import SmartAgent
from src.enviroment.game_board import GameBoard

def board_to_vector(board, symbol):
    # 2D-Liste zurückgeben
    vec = []
    for row in board:
        row_vec = []
        for cell in row:
            if cell == symbol: row_vec.append(1)
            elif cell == " ": row_vec.append(0)
            else: row_vec.append(-1)
        vec.append(row_vec)
    return vec

def augment_data(data):
    augmented = []
    for state, move in data:
        # Original
        augmented.append((state, move))
        # Gespiegelte Version
        mirrored_state = [row[::-1] for row in state]  # Zeilen spiegeln
        mirrored_move = 6 - move
        augmented.append((mirrored_state, mirrored_move))
    return augmented

def generate_data(num_games=1000, depth=2):
    agent1 = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
    agent2 = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL)

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
        pickle.dump(augment_data(data), f)
    print(f"Generated {len(data)} samples")

# Daten generieren
generate_data(num_games=15000, depth=3)