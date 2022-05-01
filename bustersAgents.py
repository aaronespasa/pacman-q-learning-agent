from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
# from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
# from wekaI import Weka
import inference
import busters
import random

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        
        # Variables to perform inference
        self.countActions = 0
        self.nearestGhostPosition = tuple()
        self.nearestGhostDistance = float('inf')
        self.possibleDirections = 0

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        return self.chooseAction(gameState)

    def getPossibleDirections(self, legal):
        possibleDirections = 0

        for direction in legal:
            if direction == Directions.NORTH: possibleDirections += 8
            if direction == Directions.SOUTH: possibleDirections += 4
            if direction == Directions.WEST: possibleDirections += 2
            if direction == Directions.EAST: possibleDirections += 1

        return possibleDirections

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)
        self.countActions = 0
        self.nearestGhostPosition = tuple()
        self.nearestGhostDistance = float('inf')
        self.possibleDirections = 0
        self.directionTaken = ""
        self.legal = []

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        livingGhosts = gameState.getLivingGhosts()
        self.legal = [a for a in gameState.getLegalPacmanActions()]
        # "{N, S, W, E, X}" -> in binary format
        self.possibleDirections = self.getPossibleDirections(self.legal)
        self.countActions = self.countActions + 1

        ####### Sort the ghosts based on the distance to the pacman #######
        ###################################################################
        ghostDistancesDict = {}
        ghostDistances = gameState.data.ghostDistances
        for i, ghostDistance in enumerate(ghostDistances):
            # the first "ghost" is the pacman, so we avoid it doing i+1
            if livingGhosts[i+1] == True:
                ghostDistancesDict[i] = ghostDistance
        
        sortedDistances = dict(sorted(ghostDistancesDict.items(), key=lambda item: item[1]))

        nearestGhostKey = next(iter(sortedDistances)) # get the first element of the dict
        shortestDistance = sortedDistances[nearestGhostKey]
        self.nearestGhostDistance = shortestDistance
        nearestGhostPosition = gameState.getGhostPositions()[nearestGhostKey]
        self.nearestGhostPosition = nearestGhostPosition
        ###################################################################

        return KeyboardAgent.getAction(self, gameState)
    
    def printLineData(self, gameState):
        # gameState.data.agentStates[0].getDirection()
        self.futureScore = gameState.getScore()
        if gameState.data.agentStates[0].getDirection() == "Stop":
            self.directionTaken = self.legal[0][0]
        else:
            self.directionTaken = gameState.data.agentStates[0].getDirection()[0]
        
        return (
            ",".join(str(gameState.getPacmanPosition())[1:-1].split(", ")) + "," +
            str(self.possibleDirections) + "," +
            ",".join(str(self.nearestGhostPosition)[1:-1].split(", ")) + "," +
            str(self.nearestGhostDistance) + "," +
            str(1) + "," +
            str(3) + "," +
            str(gameState.data.layout.width - 1) + "," +
            str(gameState.data.layout.height - 1) + ","
        )

    def printFutureData(self, gameState):
        return (
            # N: North, S: South, E: East, W: West
            str(self.directionTaken) + "," +
            str(gameState.getScore()) + "\n"
        )
    def getScoreFromAgent(self, gameState):
        return gameState.getScore()
    
    def getNumAction(self):
        return self.countActions

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        self.nearestGhostPosition = tuple()
        self.nearestGhostDistance = float('inf')
        self.possibleDirections = 0
        self.legal = []
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
        
    def chooseAction(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        self.legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        
        self.possibleDirections = self.getPossibleDirections(self.legal)
        self.countActions = self.countActions + 1

        ####### Sort the ghosts based on the distance to the pacman #######
        ###################################################################
        ghostDistancesDict = {}
        ghostDistances = gameState.data.ghostDistances
        for i, ghostDistance in enumerate(ghostDistances):
            # the first "ghost" is the pacman, so we avoid it doing i+1
            if livingGhosts[i+1] == True:
                ghostDistancesDict[i] = ghostDistance
        
        sortedDistances = dict(sorted(ghostDistancesDict.items(), key=lambda item: item[1]))

        nearestGhostKey = next(iter(sortedDistances)) # get the first element of the dict
        shortestDistance = sortedDistances[nearestGhostKey]
        self.nearestGhostDistance = shortestDistance
        nearestGhostPosition = gameState.getGhostPositions()[nearestGhostKey]
        self.nearestGhostPosition = nearestGhostPosition
        ###################################################################
        

        #######       Check if the pacman has eaten the ghost       #######
        ###################################################################
        if shortestDistance == 0:
            gameState.setGhostNotLiving(nearestGhostKey)
            return self.legal[0]
        ###################################################################
        

        #######        Check which is the best step to take         #######
        ###################################################################
        bestStep = None
        bestStepValue = float('inf')
        for step in [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]:
            if step not in self.legal: continue

            newX, newY = Actions.getSuccessor(pacmanPosition, step)
            actualDistance = self.distancer.getDistance((newX, newY), nearestGhostPosition)

            if actualDistance < shortestDistance:
                return step

            # set the best step (this will be used just in the case that the distance from any step
            # will never be less than the shortestDistance to a ghost)
            if actualDistance < bestStepValue:
                bestStepValue = actualDistance
                bestStep = step
        ###################################################################
        
        # If new distance > shortest distance, take the step that reduces that new distance
        return bestStep

    def printLineData(self, gameState):
        self.futureScore = gameState.getScore()
        if gameState.data.agentStates[0].getDirection() == "Stop":
            self.directionTaken = self.legal[0][0]
        else:
            self.directionTaken = gameState.data.agentStates[0].getDirection()[0]
        
        # top left: (1, height-1)
        # bottom right: (width-1, 3)
        return (
            ",".join(str(gameState.getPacmanPosition())[1:-1].split(", ")) + "," +
            str(self.possibleDirections) + "," +
            ",".join(str(self.nearestGhostPosition)[1:-1].split(", ")) + "," +
            str(self.nearestGhostDistance) + "," +
            str(1) + "," +
            str(3) + "," +
            str(gameState.data.layout.width - 1) + "," +
            str(gameState.data.layout.height - 1) + ","
        )
    
    def printFutureData(self, gameState):
        return (
            # N: North, S: South, E: East, W: West
            str(self.directionTaken) + "," +
            str(gameState.getScore()) + "\n"
        )
    def getScoreFromAgent(self, gameState):
        return gameState.getScore()
    
    def getNumAction(self):
        return self.countActions