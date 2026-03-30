#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem

def q1a_solver(problem: q1a_problem):
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
        self.goal = None


def astar_initialise(problem: q1a_problem):
    astarData = AStarData()
    start = problem.getStartState()
    astarData.goal = problem.goal
    astarData.best_g[start] = 0
    astarData.parents[start] = (None, None)
    astarData.frontier.push((start, 0), astar_heuristic(start, astarData.goal))
    return astarData


def astar_loop_body(problem: q1a_problem, astarData: AStarData):
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
            f_cost = new_g + astar_heuristic(successor, astarData.goal)
            astarData.frontier.push((successor, new_g), f_cost)

    return False, None


def astar_heuristic(current, goal):
    return util.manhattanDistance(current, goal)


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
