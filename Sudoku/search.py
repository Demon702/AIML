import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    # util.raiseNotDefined()
    stack = util.Stack()
    visited = []
    stack.push(problem.start_values)
    state = problem.start_values
    while not stack.isEmpty() and not problem.isGoalState(state):
        state = stack.pop()
        # visited.append(convertStateToHash(state))
        for succ in problem.getSuccessors(state):
            # if convertStateToHash(succ[0]):
            stack.push(succ[0])
                # visited.append(convertStateToHash(succ[0]))
    # print(visited)
    if stack.isEmpty():
        return False
    else:
        return state

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    end_node = problem.G.node[problem.end_node]
    node = problem.G.node[state]
    return util.points2distance([[end_node['x'],0,0], [end_node['y'],0,0]],[[node['x'],0,0],[node['y'],0,0]])

    # util.raiseNotDefined()

def AStar_search(problem, heuristic=nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """
    # print(problem)
    # print(problem.G.node[problem.start_node])
    node = problem.start_node
    path = []
    path_cost = util.PriorityQueue()
    path_cost.push(problem.start_node, 0.0)
    distance = util.Counter()
    parent = util.Counter()
    distance[problem.start_node] = 0
    explored = []
    frontier = set()
    frontier.add(problem.start_node)
    while True:
        if path_cost.isEmpty():
            return False
        # print(path_cost)
        node = path_cost.pop()
        # print(node,par)
        if problem.isGoalState(node):
            
            break
        explored.append(node)
        frontier.remove(node)
        for child , edge, cost in problem.getSuccessors(node):
            heuristic_cost = heuristic(child, problem)
            # print(heuristic_cost)
            if child not in explored and child not in frontier:
                path_cost.push(child, distance[node] + cost + heuristic_cost)
                distance[child] = distance[node] + cost
                frontier.add(child)
                parent[child] = node
            elif child not in explored:
                path_cost.update(child, distance[node] + cost + heuristic_cost)
                distance[child] = min(distance[child], distance[node] + cost)
                # frontier.add(child)
                if distance[node] + cost <= distance[child]:
                    parent[child] = node
        # print("explored",explored)
        # print("distance",distance)


    path = [node] 
    while node!= problem.start_node:
        node = parent[node]
        path = [node] + path
    # print(path)
    return path
    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """

    

    # util.raiseNotDefined()