#path search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]
"""
#def graphSearch(problem, algorithm):
    
    from util import Stack
    from util import PriorityQueue
    fringe = PriorityQueue()
    explored = set()
    stk = Stack()
    fringe.push(problem.getStartState(),algorithm.priority)#pseudocode
    stk.push(problem.getStartState())
    while true:
        if not fringe:
            return None
        node = fringe.pop()
        explored.add(node)
        children = problem.getSuccessors(node.getStartState())
        for child in children:
            if algorithm == "DFS":
                
            elif algorithm == "BFS":
                
            elif algorithm == "UCS":
                
            elif algorithm == "ASS":
"""
def directions(stack):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    steps = []
    #this involves prepending, so it can be improved.
    while not stack.isEmpty():
        node = stack.pop()
        if node[1] == "South":
            steps.insert(0, s)
        if node[1] == "North":
            steps.insert(0, n)
        if node[1] == "East":
            steps.insert(0, e)
        if node[1] == "West":
            steps.insert(0, w)
    return steps

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    fringe = Stack()
    visited = set()
    path = Stack()
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []
    visited.add(start)
    path.push(start)
    children = problem.getSuccessors(start)

    for child in children:
        fringe.push(child)
        ds = DFSRec(problem,fringe,visited,path)
        if ds != None:
            break

    pathToGoal = directions(ds)
    return pathToGoal
    #util.raiseNotDefined()

def DFSRec(problem, fringe, visited, path):
        node = fringe.pop()
        visited.add(node[0])
        children = problem.getSuccessors(node[0])
        if problem.isGoalState(node[0]):
            path.push(node)
            return path
        elif len(problem.getSuccessors(node[0])) == 0:
            return None

        fringeList = stackToList(fringe)
        for child in children:
            if child[0] not in visited and child not in fringeList:
                fringe.push(child)
                path.push(node)
                ret = DFSRec(problem,fringe,visited,path)
                if ret is not None:
                    return path
        path.pop()
        return None

def stackToList(stk):
    stack = stk
    ret = []
    while not stack.isEmpty():
        ret.append(stack.pop())
    return ret

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    from util import Queue
    fringe = Queue()
    visited = set()
    path = {}
    start = problem.getStartState()
    visited.add(start)
    if problem.isGoalState(start[0]):
        return []
    fringe.push(start)

    #test dummy value on fringe
   # fringe.push(start)

    path[start] = start
    while True:
        if fringe.isEmpty():
            print "fringe empty"
            return None
        node = fringe.pop()
        if problem.isGoalState(node):
            return pathFormat(path,node,start)
        else:
            if node == problem.getStartState():
                children = problem.getSuccessors(node[:])

            else:
                children = problem.getSuccessors(node[0])
            if children:

                for child in children:
                    if child not in visited and child[0] != problem.getStartState():
                        fringe.push(child)
                        path[child] = node
                        visited.add(child)
                        if problem.isGoalState(child):
                            return pathFormat(path,child,start)
    return pathFormat(path,node,start)
            
    #util.raiseNotDefined()


def pathFormat(path,goal,start):
    ret = []
    child = goal
    while path[child] != start:
        ret.append(path[child][1])
        child = path[child]
    return ret

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
