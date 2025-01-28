from evaluation.game_simulator import GameSimulator
from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from agents.smart_agent import SmartAgent
from agents.mini_max_agent import MiniMaxAgent
from agents.random_agent import RandomAgent

##### evaluations #####

# -> random agent vs. smart agent
# random_agent = RandomAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
# smart = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL)
# simulator = GameSimulator(random_agent, PLAYER_ONE_SYMBOL, smart, PLAYER_TWO_SYMBOL)
# simulator.simulate(500)

# -> smart agent vs. minimax agent
# smart = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
# minimax_agent = RandomAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL)
# simulator = GameSimulator(smart, PLAYER_ONE_SYMBOL, minimax_agent, PLAYER_TWO_SYMBOL)
# simulator.simulate(500)

# -> minimax agent vs. ml agent