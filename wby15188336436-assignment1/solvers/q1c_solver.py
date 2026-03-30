#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1c_problem import q1c_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#

from game import Actions, Directions


def q1c_solver(problem: q1c_problem):
    position, remaining_food = problem.getStartState()
    remaining_food = set(remaining_food)
    actions = []

    while remaining_food:
        target = min(remaining_food, key=lambda food: util.manhattanDistance(position, food))
        path = _astar_path(problem.walls, position, target)
        if not path:
            break

        actions.extend(path)
        position = target
        remaining_food.discard(target)

    return actions


def _astar_path(walls, start, goal):
    if start == goal:
        return []

    frontier = util.PriorityQueue()
    frontier.push((start, 0), util.manhattanDistance(start, goal))

    best_g = {start: 0}
    parents = {start: (None, None)}
    closed = set()

    while not frontier.isEmpty():
        state, g_cost = frontier.pop()

        if state in closed:
            continue

        if g_cost != best_g.get(state):
            continue

        closed.add(state)

        if state == goal:
            return _reconstruct_path(state, parents)

        x, y = state
        for action in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            if walls[next_x][next_y]:
                continue

            successor = (next_x, next_y)
            new_g = g_cost + 1
            if new_g < best_g.get(successor, float('inf')):
                best_g[successor] = new_g
                parents[successor] = (state, action)
                frontier.push((successor, new_g), new_g + util.manhattanDistance(successor, goal))

    return []


def _reconstruct_path(goal_state, parents):
    path = []
    node = goal_state
    while True:
        parent, action = parents[node]
        if parent is None:
            break
        path.append(action)
        node = parent
    path.reverse()
    return path
