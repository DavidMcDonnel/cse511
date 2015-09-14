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
        self.node = n
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
    fringeList = list()
    visited = set()
    startNode = problem.getStartState()
    start = Child()
    start.create(startNode,[],1)
    fringe.push(start)
    fringeList.append(start)

    while fringeList:
        node = fringe.pop()
        fringeList.remove(node)
        if problem.isGoalState(node.getNode()[0]):
            return node.getPath()
        else:
            if node == start:
                visited.add(node.getNode()[:])
            else:
                visited.add(node.getNode()[0])
            if node == start:
                children = problem.getSuccessors(node.getNode()[:])
            else:
                children = problem.getSuccessors(node.getNode()[0])
            for childNode in children:
                child = Child()
                child.create(childNode,node.getPath(),1)
                child.addElmt(child.getNode()[1])

                if child.getNode()[0] not in visited and child not in fringeList:
                    if problem.isGoalState(child.getNode()[0]):
                        return child.getPath()
                    fringe.push(child)
                    fringeList.append(child)
    return None

    #util.raiseNotDefined()


def pathFormat(parent,goal,start):
    # ret = []
    # child = goal
    # while path[child] != start:
    #     ret.append(path[child][1])
    #     child = path[child]
    # return ret
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringeList = list()
    visited = set()
    startNode = problem.getStartState()
    start = Child()
    start.create(startNode,[],0)
    fringe.push(start,0)
    fringeList.append(start)

    while fringeList:
        node = fringe.pop()
        fringeList.remove(node)
        if problem.isGoalState(node.getNode()[0]):
            return node.getPath()
        else:
            if node == start:
                visited.add(node.getNode()[:])
            else:
                visited.add(node.getNode()[0])
            if node == start:
                children = problem.getSuccessors(node.getNode()[:])
            else:
                children = problem.getSuccessors(node.getNode()[0])
            for childNode in children:
                child = Child()
                child.create(childNode,node.getPath(),node.getCost()+childNode[-1])
                child.addElmt(child.getNode()[1])

                if child.getNode()[0] not in visited and child not in fringeList:
                    if problem.isGoalState(child.getNode()[0]):
                        return child.getPath()
                    fringe.push(child,child.getCost())
                    fringeList.append(child)
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
    fringeList = list()
    visited = set()
    startNode = problem.getStartState()
    start = Child()
    start.create(startNode,[],0)
    fringe.push(start,0)
    fringeList.append(start)

    while fringeList:
        node = fringe.pop()
        fringeList.remove(node)
        if problem.isGoalState(node.getNode()[0]):
            return node.getPath()
        else:
            visited.add(node.getNode())
            if node == start:
                children = problem.getSuccessors(node.getNode()[:])
            else:
                children = problem.getSuccessors(node.getNode()[0])
            for childNode in children:
                child = Child()
                h = heuristic(childNode[0],problem)
                child.create(childNode,node.getPath(),node.getCost()+childNode[-1] + h)
                child.addElmt(child.getNode()[1])

                if child.getNode() not in visited and child not in fringeList:
                    if problem.isGoalState(child.getNode()[0]):
                        return child.getPath()
                    fringe.push(child,child.getCost())
                    fringeList.append(child)
    return None
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
