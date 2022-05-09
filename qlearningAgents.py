# qlearningAgents.py
# ------------------

from game import *
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

    def __init__(self, epsilon=0.5,gamma=0.8,alpha=0.5, ghostAgents = None, numTraining=0, **args):
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

        self.pacmanPositionLastLast = None

        self.actions = {"North":0, "East":1, "South":2, "West":3, "Stop":4}

        # distance items: [start_distance, final_distance]
        self.distances = [
            [1, 2],
            [2, 3],
            [3, 6],
            [6, 12],
            [12, 100]
        ]

        self.index = 0  # This is always Pacman
        self.ghostNearestDistance = float('inf')

        self.writeInitQtable()
    
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        reward = 1
        # if state.isWin():
        #     discrete_distance = 1
        # else:
        #     distance = state.getDistanceNearestGhost()
        #     discrete_distance = self.computeDiscretizedDistance(distance)
        
        if not self.lastState is None:
            # if state.isWin():
            #     discrete_distance = 1
            # else:
            #     distance = self.lastState.getDistanceNearestGhost()
            #     discrete_distance = self.computeDiscretizedDistance(distance)
            
            # if the pacman position is the same than the position of any other ghost: reward = 1
            # in any other case: reward = 0

            # print("-----------------")

            # for ghost_position in self.lastState.getGhostPositions():
            #     # get the next action of state
            #     # print("Pacman position: " + str(state.getPacmanPosition()))
            #     # print("Ghost position: " + str(ghost_position))
            #     # print()

                # if ghost_position == state.getPacmanPosition():
                #     reward = 1
                #     break
            
            # print("-----------------")
            
            # if reward == 1:
            #     print("reward is 1")
            # if discrete_distance == 1:
            #     reward = 50
            # elif discrete_distance == 2:
            #     reward = 12
            # elif discrete_distance == 3:
            #     reward = 6
            # elif discrete_distance == 4:
            #     reward = 2
            # elif discrete_distance == 5:
            #     reward = 1

            # reward = 1
            if self.pacmanPositionLastLast == state.getPacmanPosition():
                reward = -1

            self.pacmanPositionLastLast = self.lastState.getPacmanPosition()

            # reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)

        return state


    # def observationFunction(self, state):
    #     reward = 0
        
    #     if not self.lastState is None:
    #         if state.isWin():
    #             previous_discrete_distance = 1
    #             current_discrete_distance = 1
    #         else:
    #             previous_distance = self.lastState.getDistanceNearestGhost()
    #             previous_discrete_distance = self.computeDiscretizedDistance(previous_distance)

    #             current_distance = state.getDistanceNearestGhost()
    #             current_discrete_distance = self.computeDiscretizedDistance(current_distance)
        
    #         if current_discrete_distance < previous_discrete_distance:
    #             # Si pacman se acerca al fantasma, pasando de un rango de distancias discretizadas a otro menor
    #             reward += 2

    #             if state.isWin():
    #                 reward += 5

    #         elif current_discrete_distance == previous_discrete_distance:
    #             # Si la distancia al fantasma sigue en el mismo rango de distancias discretizadas
    #             # Puede haberse acercado, si se movio en la direccion correcta
    #             ghost_direction = state.getDirectionToNearestGhost()
    #             last_action = self.lastAction   
    #             # Si el fantasma está al norte
    #             if (last_action == "North" and ghost_direction == 2):
    #                 reward += 2
    #             # Si el fantasma está al sur
    #             if (last_action == "South" and ghost_direction == 6):
    #                 reward += 2   
    #             # Si el fantasma está al este
    #             if (last_action == "East" and ghost_direction == 8):
    #                 reward += 2
    #             # Si el fantasma está al oeste
    #             if (last_action == "West" and ghost_direction == 4):
    #                 reward += 2

    #             # Si el fantasma está al noreste y pacman se movio al norte o al este
    #             if ((ghost_direction == 1) and (last_action == "North" or last_action == "East")):
    #                 reward += 2
    #             # Si el fantasma está al sureste y pacman se movio al sur o al este
    #             if ((ghost_direction == 7) and (last_action == "South" or last_action == "East")):
    #                 reward += 2
    #             # Si el fantasma está al noroeste y pacman se movio al norte o al oeste
    #             if ((ghost_direction == 3) and (last_action == "North" or last_action == "West")):
    #                 reward += 2
    #             # Si el fantasma está al suroeste y pacman se movio al sur o al oeste        
    #             if ((ghost_direction == 5) and (last_action == "South" or last_action == "West")):
    #                 reward += 2
    #             else:
    #             # Si se movio en la direccion equivocada
    #                 reward -= 1
            
    #         elif current_discrete_distance > previous_discrete_distance:
    #             # Si pacman se aleja del fantasma, pasando de un rango de distancias discretizadas a otro mayor
    #             reward -= 1

    #         self.observeTransition(self.lastState, self.lastAction, state, reward)
        
    #     return state

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

        Args:
            state: (x,y) position of the pacman
        """
        num_directions = 8
        # distance = state.getDistanceNearestGhost()
        # discrete_distance = self.computeDiscretizedDistance(distance)
        ghost_direction = state.getDirectionToNearestGhost()
        legalActions = state.getLegalActions()

        # we assume there will be always at least one legal action
        value = 0
        for action in legalActions:
            if action == 'North':
                value += 1
            elif action == 'South':
                value += 2
            elif action == 'East':
                value += 4
            elif action == 'West':
                value += 8
        
        # print(f"Distance: {discrete_distance}, Direction: {ghost_direction}")

        # Maximum values:
        # discrete_distance_max = len(self.distances) = 5 -> d - 1 = 4
        # (d - 1) * num_directions = 4 * 8 = 32
        # 32 + 8 - 1 = 39
        
        # distance_direction = (discrete_distance - 1) * num_directions + ghost_direction - 1
        # return (value - 1 ) * 40 + distance_direction
        # direction = num_directions + ghost_direction - 1
        return (value - 1) * num_directions + ghost_direction - 1 # 112 + 8 - 1 = 119
        
    def writeInitQtable(self):
        "Write qtable to disc"
        # initQTable = [[0 for state in range(12*18)] for action in range(5) ]
        num_actions = len(self.actions)
        #num_discretized_distances = len(self.distances)
        num_legal_actions = 15
        num_directions = 8

        with open("qtable.ini.txt", "w", encoding="utf-8") as initTableFile:
            for _ in range(num_legal_actions * num_directions):
                line = "0.0 " * (num_actions - 1) + "0.0\n"
                initTableFile.write(line)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
