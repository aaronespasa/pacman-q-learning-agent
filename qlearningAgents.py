# qlearningAgents.py
# ------------------

from game import *
from game import Actions  

from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
    """
    def __init__(self, **args):
        "Initialize Q-values"
        ReinforcementAgent.__init__(self, **args)

        self.actions = {"north":0, "east":1, "south":2, "west":3, "exit":4}
        self.table_file = open("qtable.txt", "r+")      
        self.q_table = self.readQtable()
        self.epsilon = 1

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")    
            
    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        return state[0]+state[1]*4

    def getQValue(self, state, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]

        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
          return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
          return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
             return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        # Positions in the q table (row, column)
        # -> row = distance
        # -> column = action
        
        position = self.computePosition(state)
        action_column = self.actions[action]

        old_q_value = (1 - self.alpha) * self.getQValue(state, action)
        if len(self.getLegalActions(state)) == 0: # terminal state
            new_value = self.alpha * reward
        else: # non-terminal state
            new_value = self.alpha * \
                         (reward + self.discount * self.computeValueFromQValues(nextState))

        self.q_table[position][action_column] = old_q_value + new_value

        # self.printQtable()

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"
    # pillar la distancia al fantasma más cercano e ir a por el hasta que nos lo comamos

    # fantasmas vivos
    def __init__(self, epsilon=0.5, gamma=0.8, alpha=0.5, ghostAgents = None, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        QLearningAgent.__init__(self, **args)

        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining

        self.epsilon = epsilon

        print(f"epsilon{epsilon}; gamma{gamma}; alpha{alpha}")
        self.pacmanPositionLastLast = None
        self.lastlastAction = None

        self.actions = {"North":0, "East":1, "South":2, "West":3, "Stop":4}

        # distance items: [start_distance, final_distance]
        self.distances = [
            [1, 2],
            [2, 3],
            [3, 6],
            [6, 12],
            [12, 100]
        ]

        self.writeInitQtable()
    
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        # Avoid long paths
        reward = -1       
        if not self.lastState is None:
            if not state.isWin():              
                actual_distance = state.getDistanceNearestGhost()
                last_distance = self.lastState.getDistanceNearestGhost()

            # Reward Ghost Eaten
            # if state.isWin() or state.getPacmanPosition() in self.lastState.getGhostPositions(): reward = 30
            if (state.getScore() == self.lastState.getScore() + 199) or state.isWin(): reward = 30
            
            # Reward Food Eaten
            elif state.getNumFood() < self.lastState.getNumFood(): reward = 15

            # Punish Loop 
            elif state.getPacmanPosition() == self.pacmanPositionLastLast : reward = -15 
            #elif self.lastAction == Actions.reverseDirection(self.lastlastAction): reward = -15

            # Punish actions that go backwards and stop
            elif self.lastAction == 'Stop': reward = -10
            elif actual_distance > last_distance: reward = -5
            
            # Make Action
            self.observeTransition(self.lastState, self.lastAction, state, reward)

        return state

    def computeDiscretizedDistance(self, distance):
        for row_num in range(1, len(self.distances)):
            if self.distances[row_num-1][0] <= distance < self.distances[row_num-1][1]:
                return row_num
        
        # just in case the distance is greater than the last distance
        # in other words, distance > self.distances[-1][1]
        return len(self.distances)

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        state: ghost_direction_relative; number of living ghosts; legal actions booleans.
        """
        ghost_direction = state.getDirectionToNearestGhost() - 1
        # n_living = sum(state.getLivingGhosts()) - 1
        legal = -1
        for l in state.getLegalActions(): 
            if l == 'North': legal += 1
            if l == 'South': legal += 2
            if l == 'East': legal += 4
            if l == 'West': legal += 8
     
        #return 8*16 * n_living + (8 * legal + ghost_direction)
        return 8 * legal + ghost_direction
        
    def writeInitQtable(self):
        "Write qtable to disc"
        # initQTable = [[0 for state in range(12*18)] for action in range(5) ]
        num_actions = len(self.actions)
        #num_discretized_distances = len(self.distances)
        num_legal_actions = 15
        num_directions = 8
        num_living_ghosts = 4

        with open("qtable.ini.txt", "w", encoding="utf-8") as initTableFile:
            for _ in range(num_directions + 8 * num_legal_actions + 8*16*num_living_ghosts):
                line = "0.0 " * (num_actions - 1) + "0.0\n"
                initTableFile.write(line)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        #Update Position
        if self.lastState: self.pacmanPositionLastLast = self.lastState.getPacmanPosition()
        self.doAction(state,action)
        return action
