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
    print "HEYY"
    print stack
    print "BYEE"
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
    fringe.push(start)
    ds = DFSRec(problem, fringe, visited, path)
    pathToGoal = directions(ds)
    return pathToGoal
    #util.raiseNotDefined()

def DFSRec(problem, fringe, visited, path):
        node = fringe.pop()
        print node
        print "HELLO"
        path.push(node)
        copyStack = path
        i = 0
        print "path:",
        while not copyStack.isEmpty():
            print copyStack.pop()
            print i
            i= i + 1
        visited.add(node)
        if problem.isGoalState(node):
            return path
        if not problem.isGoalState(node):
            if node != problem.getStartState():
                if len(problem.getSuccessors(node[0])) == 0:
                    path.pop()
                    return None
            else:
                if len(problem.getSuccessors(node[:])) == 0:
                    path.pop()
                    return None
        if node == problem.getStartState():
            for child in problem.getSuccessors(node[:]):
                if child not in visited:
                     fringe.push(child)
                     st = DFSRec(problem, fringe, visited, path)
                     if st:
                         return path
        else: 
            for child in problem.getSuccessors(node[0]):
                if child not in visited:
                     fringe.push(child)
                     st = DFSRec(problem, fringe, visited, path)
                     if st:
                         return path
        path.pop()
        return None

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"

    from util import Stack
    fringe = []
    explored = set()
    stk = Stack()
    fringe.append(problem.getStartState())
    stk.push(problem.getStartState())
    while true:
        if not fringe:
            return None
        node = fringe.pop()
        explored.add(node)
        children = problem.getSuccessors(node.getStartState())
       # for child in children:
            
    #util.raiseNotDefined()

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
