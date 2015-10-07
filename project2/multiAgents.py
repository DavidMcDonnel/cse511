# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html




from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]


    def distEval(self,newPos,x,y):
        from util import manhattanDistance
        dist = util.manhattanDistance(newPos,x)-util.manhattanDistance(newPos,y)
        return -1 if dist<0 else 1 if dist>0 else 0


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        ghostValue = 0
        foodValue = 0
        foodList = newFood.asList()
        foodList.sort(lambda x,y: util.manhattanDistance(newPos,x)-util.manhattanDistance(newPos,y))
        if len(foodList)>0:
            foodValue = util.manhattanDistance(newPos,foodList[0])
        ghosts = list(g.getPosition() for g in newGhostStates)
        if len(ghosts)>0:
            ghosts.sort(lambda x,y: self.distEval(newPos,x,y))
            dist = util.manhattanDistance(newPos,ghosts[0])
            ghostValue = -99 if dist==0 else 2*-1.0/dist

        return 2+ghostValue if foodValue==0 else ghostValue + 1.0/float(foodValue)


    def scoreEvaluationFunction(currentGameState):
        """
          This default evaluation function just returns the score of the state.
          The score is the same one displayed in the Pacman GUI.

          This evaluation function is meant for use with adversarial search agents
          (not reflex agents).
        """
        return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def value(self, gameState, agent, depth, evalFun):
        if agent >= (gameState.getNumAgents()):
            agent = 0
            depth = depth - 1
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return evalFun(gameState)
        elif agent == 0:
            return self.maxValue(gameState, agent, depth, evalFun)
        else:
            return self.minValue(gameState, agent, depth, evalFun)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        agent = 0
        actions = gameState.getLegalActions(agent)
        options = []
        successors = []
        best = []
        for action in actions:
            successors.append((action, gameState.generateSuccessor(agent, action)))
        for successor in successors:
            val = self.value(successor[1], agent + 1, self.depth, self.evaluationFunction)
            options.append((successor, val))
        maxV = float("-inf")
        for opt in options:
            if opt[1] > maxV:
                maxV = opt[1]
                best = opt
        return best[0][0]


        # util.raiseNotDefined()

    def maxValue(self, gameState, agent, depth, evalFun):
        maxV = float("-inf")
        successors = []
        best = []
        actions = gameState.getLegalActions(agent)
        for action in actions:
            successors.append(gameState.generateSuccessor(agent, action))
        for successor in successors:
            val = self.value(successor, agent + 1, depth, evalFun)
            if val > maxV:
                maxV = val
        return maxV

    def minValue(self, gameState, agent, depth, evalFun):
        minV = float("inf")
        successors = []
        best = []
        actions = gameState.getLegalActions(agent)
        for action in actions:
            successors.append(gameState.generateSuccessor(agent, action))
        for successor in successors:
            val = self.value(successor, agent + 1, depth, evalFun)
            if val < minV:
                minV = val
        return minV


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agent = 0
        options = []
        successors = []
        best = []
        maxBest = float("-inf")
        minBest = float("inf")
        actions = gameState.getLegalActions(agent)
        for action in actions:
            successors.append((action, gameState.generateSuccessor(agent, action)))
        for successor in successors:
            val = self.value(successor[1], agent + 1, self.depth, self.evaluationFunction, maxBest, minBest)
            options.append((successor, val))
        maxV = float("-inf")
        for opt in options:
            if opt[1] > maxV:
                maxV = opt[1]
                best = opt
        return best[0][0]

    def value(self, gameState, agent, depth, evalFun, maxBest, minBest):
        if agent >= gameState.getNumAgents():
            agent = 0
            depth = depth - 1
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return evalFun(gameState)
        elif agent == 0:
            return self.maxValue(gameState, agent, depth, evalFun, maxBest, minBest)
        else:
            return self.minValue(gameState, agent, depth, evalFun, maxBest, minBest)

    def maxValue(self, gameState, agent, depth, evalFun, maxBest, minBest):
        maxV = float("-inf")
        successors = []
        actions = gameState.getLegalActions(agent)
        for action in actions:
            successors.append(gameState.generateSuccessor(agent, action))
        for successor in successors:
            val = self.value(successor, agent + 1, depth, evalFun, maxBest, minBest)
            maxV = max(maxV, val)
            if maxV >= minBest:
                return maxV
            maxBest = max(maxBest, maxV)
        return maxV

    def minValue(self, gameState, agent, depth, evalFun, maxBest, minBest):
        minV = float("inf")
        successors = []
        actions = gameState.getLegalActions(agent)
        for action in actions:
            successors.append(gameState.generateSuccessor(agent, action))
        for successor in successors:
            val = self.value(successor, agent + 1, depth, evalFun, maxBest, minBest)
            minV = min(minV, val)
            if minV <= maxBest:
                return minV
            minBest = min(minBest, minV)
        return minV


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        agent = 0
        actions = gameState.getLegalActions(agent)
        options = []
        successors = []
        best = []
        for action in actions:
            successors.append((action, gameState.generateSuccessor(agent, action)))
        for succ in successors:
            val = self.value(succ[1], agent + 1, self.depth, self.evaluationFunction)
            options.append((succ, val))
        maxV = float("-inf")
        for opt in options:
            if opt[1] > maxV:
                maxV = opt[1]
                best = opt
        return best[0][0]

    def value(self, gameState, agent, depth, evalFun):
        if agent >= gameState.getNumAgents():
            agent = 0
            depth = depth - 1
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return evalFun(gameState)
        elif agent == 0:
            return self.maxValue(gameState, agent, depth, evalFun)
        else:
            return self.minValue(gameState, agent, depth, evalFun)

    def maxValue(self, gameState, agent, depth, evalFun):
        maxV = float("-inf")
        successors = []
        actions = gameState.getLegalActions(agent)
        for act in actions:
            successors.append(gameState.generateSuccessor(agent, act))
        for succ in successors:
            val = self.value(succ, agent + 1, depth, evalFun)
            maxV = max(maxV, val)
        return maxV

    def minValue(self, gameState, agent, depth, evalFun):
        v = 0
        successors = []
        actions = gameState.getLegalActions(agent)
        for act in actions:
            successors.append(gameState.generateSuccessor(agent, act))
        for succ in successors:
            p = 1 / len(successors)
            v = v + (p * self.value(succ, agent + 1, depth, evalFun))
        return v
        # util.raiseNotDefined()

    def betterEvaluationFunction(currentGameState):
        """
         Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
         evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
         """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


    # Abbreviation
    better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
