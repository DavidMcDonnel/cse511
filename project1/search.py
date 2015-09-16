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


class Child:
    def __init__(self):
        self.node = tuple()
        self.path = []
        self.cost = 0

    def create(self,n,lis,c):
        from copy import deepcopy
        self.node = deepcopy(n)
        self.path = lis[:]
        self.cost = c

    def addElmt(self,elmt):
        self.path.append(elmt)

    def getPath(self):
        return self.path

    def getNode(self):
        return self.node

    def getCost(self):
        return self.cost


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
        elif len(children) == 0:
            return None

        for child in children:
            if child[0] not in visited and child not in fringe.list:
                fringe.push(child)
                path.push(node)
                ret = DFSRec(problem,fringe,visited,path)
                if ret is not None:
                    return path
        path.pop()
        return None

class FringeObj:
    def __init__(self):
        self.parent = []
        self.state = []
        self.action = ""
    def create(self, s, p, a):
        self.state = s
        self.parent = p
        self.action = a
    def getState(self):
        return self.state
    def getParent(self):
        return self.parent
    def getAction(self):
        return self.action

def ListOfDicts(child):
    path = []
    while child.getParent():
        dir = child.getAction()
        path.insert(0, dir)
        child = child.getParent()
    return path
    
    
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    from util import Queue
    from game import Directions
    s = Directions.SOUTH
    n = Directions.NORTH
    e = Directions.EAST
    w = Directions.WEST
    st = Directions.STOP
    
    fringe = Queue()
    visited = set()
    startNode = problem.getStartState()
    start = FringeObj()
    start.create(startNode,[],st)
    visited.add(start.getState())
    fringe.push(start)

    while fringe.list:

        node = fringe.pop()

        if problem.isGoalState(node.getState()[0]):
            return ListOfDicts(node)
        else:
            if node.getState() == start.getState():
                children = problem.getSuccessors(node.getState())
            else:
                visited.add(node.getState())
                children = problem.getSuccessors(node.getState()[0])

            for childNode in children:
                child = FringeObj()
                child.create(childNode,node ,childNode[1])
                
                if child.getState() not in visited and child not in fringe.list:
                    if problem.isGoalState(child.getState()[0]):
                        return ListOfDicts(child)
                    fringe.push(child)
    return None

    #util.raiseNotDefined()


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    fringe = PriorityQueue()
    visited = set()
    startNode = problem.getStartState()
    start = Child()
    start.create(startNode,[],0)
    fringe.push(start,0)

    while fringe.heap:
        node = fringe.pop()
        if problem.isGoalState(node.getNode()[0]):
            return node.getPath()
        else:
            if node == start:
                visited.add(node.getNode()[:])
                children = problem.getSuccessors(node.getNode()[:])
            else:
                visited.add(node.getNode()[0])
                children = problem.getSuccessors(node.getNode()[0])
            for childNode in children:
                child = Child()
                child.create(childNode,node.getPath(),node.getCost()+childNode[-1])
                child.addElmt(child.getNode()[1])
                if child.getNode()[0] not in visited and child not in fringe.heap:
                    fringe.push(child,child.getCost())
    return None
    #util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    fringe = PriorityQueue()
    visited = set()
    startNode = problem.getStartState()
    start = Child()
    start.create(startNode,[],0)
    fringe.push(start,0)

    while fringe.heap:
        node = fringe.pop()
        if problem.isGoalState(node.getNode()[0]):
            return node.getPath()
        elif node.getNode() in visited:
            continue
        else:
            if node == start:
                children = problem.getSuccessors(node.getNode()[:])
                visited.add(start.getNode()[:])
            else:
                children = problem.getSuccessors(node.getNode()[0])
                visited.add(node.getNode()[0])
            for childNode in children:
                child = Child()
                h = heuristic(childNode[0],problem)
                child.create(childNode,node.getPath(),node.getCost()+childNode[-1] + h)
                child.addElmt(child.getNode()[1])
                if child.getNode()[0] not in visited and child not in fringe.heap:
                    fringe.push(child,child.getCost())
    return None
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
