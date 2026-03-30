#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1b_problem import q1b_problem

def q1b_solver(problem: q1b_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    return result

#-------------------#
# DO NOT MODIFY END #
#-------------------#

class AStarData:
    def __init__(self):
        self.frontier = util.PriorityQueue()
        self.best_g = {}
        self.parents = {}
        self.closed = set()


def astar_initialise(problem: q1b_problem):
    astarData = AStarData()
    start = problem.getStartState()
    astarData.best_g[start] = 0
    astarData.parents[start] = (None, None)
    astarData.frontier.push((start, 0), astar_heuristic(start, problem.food))
    return astarData


def astar_loop_body(problem: q1b_problem, astarData: AStarData):
    if astarData.frontier.isEmpty():
        return True, []

    state, g_cost = astarData.frontier.pop()

    if state in astarData.closed:
        return False, None

    if g_cost != astarData.best_g.get(state):
        return False, None

    astarData.closed.add(state)

    if problem.isGoalState(state):
        return True, _reconstruct_path(state, astarData.parents)

    for successor, action, step_cost in problem.getSuccessors(state):
        new_g = g_cost + step_cost
        if new_g < astarData.best_g.get(successor, float('inf')):
            astarData.best_g[successor] = new_g
            astarData.parents[successor] = (state, action)
            f_cost = new_g + astar_heuristic(successor, problem.food)
            astarData.frontier.push((successor, new_g), f_cost)

    return False, None


def astar_heuristic(current, goals):
    if not goals:
        return 0
    return min(util.manhattanDistance(current, food) for food in goals)


def _reconstruct_path(goal_state, parents):
    actions = []
    node = goal_state
    while True:
        parent, action = parents[node]
        if parent is None:
            break
        actions.append(action)
        node = parent
    actions.reverse()
    return actions
