import logging
import random

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance


def scoreEvaluationFunction(currentGameState):
    """
    Stronger evaluation than raw score:
    - prefer high game score
    - prefer being closer to food/capsules
    - avoid active ghosts, chase scared ghosts
    """
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    score = currentGameState.getScore()
    pacman_pos = currentGameState.getPacmanPosition()

    food = currentGameState.getFood().asList()
    if food:
        closest_food = min(manhattanDistance(pacman_pos, f) for f in food)
        score += 8.0 / (closest_food + 1)
        score -= 0.25 * len(food)

    capsules = currentGameState.getCapsules()
    if capsules:
        closest_capsule = min(manhattanDistance(pacman_pos, c) for c in capsules)
        score += 3.0 / (closest_capsule + 1)
        score -= 1.5 * len(capsules)

    for ghost_state in currentGameState.getGhostStates():
        ghost_pos = ghost_state.getPosition()
        d = manhattanDistance(pacman_pos, ghost_pos)
        scared = ghost_state.scaredTimer

        if scared > 0:
            score += 6.0 / (d + 1)
        else:
            if d == 0:
                return float('-inf')
            if d < 2:
                score -= 30
            score -= 2.0 / d

    return score


class Q2_Agent(Agent):

    def __init__(self, evalFn='scoreEvaluationFunction', depth='3'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    @log_function
    def getAction(self, gameState: GameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.

            Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

            gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        logger = logging.getLogger('root')
        logger.info('MinimaxAgent')

        legal_actions = [a for a in gameState.getLegalActions(0) if a != Directions.STOP]
        if not legal_actions:
            legal_actions = gameState.getLegalActions(0)
        if not legal_actions:
            return Directions.STOP

        best_score = float('-inf')
        best_action = legal_actions[0]
        alpha = float('-inf')
        beta = float('inf')

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            score = self._alphabeta(successor, depth=0, agent_index=1, alpha=alpha, beta=beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action

    def _alphabeta(self, state: GameState, depth: int, agent_index: int, alpha: float, beta: float):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()

        if agent_index >= num_agents:
            return self._alphabeta(state, depth + 1, 0, alpha, beta)

        legal_actions = state.getLegalActions(agent_index)
        if not legal_actions:
            return self.evaluationFunction(state)

        if agent_index == 0:
            value = float('-inf')
            actions = [a for a in legal_actions if a != Directions.STOP] or legal_actions
            for action in actions:
                successor = state.generateSuccessor(agent_index, action)
                value = max(value, self._alphabeta(successor, depth, agent_index + 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        value = float('inf')
        for action in legal_actions:
            successor = state.generateSuccessor(agent_index, action)
            value = min(value, self._alphabeta(successor, depth, agent_index + 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value
