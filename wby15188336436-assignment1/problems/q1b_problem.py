import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1b_problem:
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startingGameState: GameState = gameState
        self.walls = gameState.getWalls()
        self.food = frozenset(gameState.getFood().asList())

    @log_function
    def getStartState(self):
        start = self.startingGameState.getPacmanPosition()
        return (start, self.food)

    @log_function
    def isGoalState(self, state):
        _, remaining_food = state
        return len(remaining_food) == 0

    @log_function
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        (x, y), remaining_food = state
        successors = []

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            if self.walls[next_x][next_y]:
                continue

            next_pos = (next_x, next_y)
            next_food = remaining_food
            if next_pos in remaining_food:
                next_food = frozenset(f for f in remaining_food if f != next_pos)

            successors.append(((next_pos, next_food), action, 1))

        return successors
