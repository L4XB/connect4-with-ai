from src.evaluation.game_simulator import GameSimulator
from src.constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from src.agents.smart_agent import SmartAgent
from src.agents.mini_max_agent import MiniMaxAgent
from src.agents.random_agent import RandomAgent
from src.agents.ml_agent.agent import AIAgent

##### evaluations #####

# -> random agent vs. smart agent
# random_agent = RandomAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
# smart = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL)
# simulator = GameSimulator(random_agent, PLAYER_ONE_SYMBOL, smart, PLAYER_TWO_SYMBOL)
# simulator.simulate(500)

# -> smart agent vs. minimax agent
smart = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
minimax_agent = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL, max_depth = 3)
simulator = GameSimulator(smart, PLAYER_ONE_SYMBOL, minimax_agent, PLAYER_TWO_SYMBOL)
simulator.simulate(500)

# -> minimax agent vs. ml agent
# minimax_agent = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL, max_depth = 3)
# ai_agent = AIAgent("src/agents/ml_agent/models/connect4_model_full_trained.pth", PLAYER_TWO_SYMBOL)
# simulator = GameSimulator(minimax_agent, PLAYER_ONE_SYMBOL, ai_agent, PLAYER_TWO_SYMBOL)
# simulator.simulate(20)
