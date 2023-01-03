from os import DirEntry
from telnetlib import GA
from util import manhattanDistance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple
import numpy as np
from game import Agent
from pacman import GameState



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """
    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)


        return legalMoves[chosenIndex]

    

    def evaluationFunction(self,currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #Calculate distance to the nearest food
        newFoodList = np.array(newFood.asList())
        distanceToFood = [util.manhattanDistance(newPos, food) for food in newFoodList]
        min_food_distance = 0
        if len(newFoodList) > 0:
            min_food_distance = distanceToFood[np.argmin(distanceToFood)]

        #Calculate the distance to nearest ghost
        ghostPositions = np.array(successorGameState.getGhostPositions())
        distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
        min_ghost_distance = 0
        nearestGhostScaredTime = 0
        if len(ghostPositions) > 0:
            min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
            nearestGhostScaredTime = newScaredTimes[np.argmin(distanceToGhost)]
            # avoid certain death
            if min_ghost_distance <= 1 and nearestGhostScaredTime == 0:
                return -999999
            # eat a scared ghost
            if min_ghost_distance <= 1 and nearestGhostScaredTime > 0:
                return 999999

        value = successorGameState.getScore() - min_food_distance
        if nearestGhostScaredTime > 0:
            # follow ghosts if scared
            value -= min_ghost_distance
        else:
            value += min_ghost_distance
        return value

def scoreEvaluationFunction(successorGameState: GameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    # return currentGameState.getScore()
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #Calculate distance to the nearest food
    newFoodList = np.array(newFood.asList())
    distanceToFood = [util.manhattanDistance(newPos, food) for food in newFoodList]
    min_food_distance = 0
    if len(newFoodList) > 0:
        min_food_distance = distanceToFood[np.argmin(distanceToFood)]

    #Calculate the distance to nearest ghost
    ghostPositions = np.array(successorGameState.getGhostPositions())
    distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
    min_ghost_distance = 0
    nearestGhostScaredTime = 0
    if len(ghostPositions) > 0:
        min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
        nearestGhostScaredTime = newScaredTimes[np.argmin(distanceToGhost)]
        # avoid certain death
        if min_ghost_distance <= 1 and nearestGhostScaredTime == 0:
            return -999999
        # eat a scared ghost
        if min_ghost_distance <= 1 and nearestGhostScaredTime > 0:
            return 999999

    value = successorGameState.getScore() - min_food_distance
    if nearestGhostScaredTime > 0:
        # follow ghosts if scared
        value -= min_ghost_distance
    else:
        value += min_ghost_distance
    return value


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='4'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)



class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState) -> str:
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Don't forget to limit the search depth using self.depth. Also, avoid modifying
          self.depth directly (e.g., when implementing depth-limited search) since it
          is a member variable that should stay fixed throughout runtime.

        """

        def V_minimax(state:GameState,agent_index,current_depth)-> tuple:
          #-> (score,action)
          # the case that game is end
          if state.isLose() or state.isWin():
            return(state.getScore(),Directions.STOP)
          else:
            if current_depth == 0:
              return(scoreEvaluationFunction(state),Directions.STOP)

            #current_depth >0
            else:
              next_index = agent_index + 1
              next_depth = current_depth
              if next_index == state.getNumAgents() :
                next_index = 0
                next_depth -= 1
              assert agent_index < state.getNumAgents()

              if agent_index == 0: #max layer
                bestVal = -float('inf')
                bestMove = None
                for action in state.getLegalActions(agent_index):
                  value = V_minimax(state.generateSuccessor(agent_index,action),next_index,\
                    next_depth)[0]
                  if value > bestVal:
                    bestVal = value
                    bestMove = action
                return bestVal,bestMove
              else: #min layer
                bestVal = float('inf')
                bestMove = None
                for action in state.getLegalActions(agent_index):
                  value = V_minimax(state.generateSuccessor(agent_index,action),next_index,\
                    next_depth)[0]
                  if value < bestVal:
                    bestVal = value
                    bestMove = action
                return bestVal,bestMove



        score,action = V_minimax(gameState,self.index,self.depth)
        return action 


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState) -> str:
      """
        Returns the minimax action using self.depth and self.evaluationFunction
      """
      def alpha_beta(state:GameState,depth:int,alpha:float,beta:float,agent_index:int)->tuple:
        #alpha_beta ->(score,action)
        
        if state.isLose() or state.isWin(): return state.getScore(),Directions.STOP
        #######
        elif depth == 0:return scoreEvaluationFunction(state),Directions.STOP
        #######
        else:
          next_index = agent_index + 1
          next_depth = depth
          if next_index == state.getNumAgents():
            next_index = 0
            next_depth -= 1
          if agent_index == 0: #max level
            best_val = -float('inf')
            bestMove = None
            for action in state.getLegalActions():
              value = alpha_beta(state.generateSuccessor(agent_index,action)\
                ,next_depth,alpha,beta,next_index)[0]
              if value > best_val:
                best_val = value
                bestMove = action
              alpha = max(alpha,value)
              if value >= beta: break
            return best_val,bestMove
          else: #min level
            assert agent_index != 0 #checked
            assert agent_index < state.getNumAgents()
            best_val = float('inf')
            bestMove = None
            for action in state.getLegalActions(agent_index):
              value = alpha_beta(state.generateSuccessor(agent_index,action)\
                ,next_depth,alpha,beta,next_index)[0]
              
              if value < best_val:
                best_val = value
                bestMove = action
              beta = min(beta,value)
              if alpha >= value: break
            return best_val,bestMove


      alpha = -float('inf')
      beta  = float('inf')
      score,action = alpha_beta(gameState,self.depth,alpha,beta,self.index)
      return action 
        


class ExpectimaxAgent(MultiAgentSearchAgent):
    

    def getAction(self, gameState: GameState) -> str:
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def Vem(gameState, depth, index):
            legalMoves = gameState.getLegalActions(index)
            if gameState.isWin() or gameState.isLose() or len(legalMoves) == 0:
                return gameState.getScore()

            elif depth == 0:
                return self.evaluationFunction(gameState)

            if index == 0:  # Pacman's turn
                score_list = []
                for action in legalMoves:
                    score_list.append(
                        Vem(gameState.generateSuccessor(index, action), depth, index + 1))
                return max(score_list)

            elif index == gameState.getNumAgents() - 1:  # Last ghost turn
                score_list = []
                for action in legalMoves:
                    score_list.append(
                        Vem(gameState.generateSuccessor(index, action), depth-1, 0))
                return sum(score_list) / len(score_list)

            else:  # Non-last ghost turn
                score_list = []
                for action in legalMoves:
                    score_list.append(
                        Vem(gameState.generateSuccessor(index, action), depth, index+1))
                return sum(score_list) / len(score_list)
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions

        scores = []

        for action in legalMoves:
            scores.append(Vem(gameState.generateSuccessor(
                self.index, action), self.depth, self.index + 1))

        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

