<h1 id="title">Creating a Pac-Man Agent using RL (Q-Learning)</h1>

<p align="center">
  <img src="https://lh3.googleusercontent.com/H8hhcUas7f9Pi4aMLTQfSTVk1wwE1d_SPYYGldXn9S8GARJis2ED4EpnIfXzfBhTP8KZM64bFnmgowpU3Ct7b7OznwcRakNOM3mB2KRr=s660" alt="pac-man banner" />
</p>
  
Welcome to the Pac-Man Agent using Reinforcement Learning (Q-Learning) repository! This project aims to create an intelligent Pac-Man agent using Q-Learning, a popular reinforcement learning technique. The agent will be able to navigate through various Pac-Man mazes, avoiding ghosts and consuming power pellets.

<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#getting-started">Getting Started</a></li>
  <li><a href="#file-structure">File Structure</a></li>
  <li><a href="#agent-design">Agent Design</a>
  <ul>
      <li><a href="#state-space">State Space</a></li>
      <li><a href="#reward-function">Reward Function</a></li>
      <li><a href="#training-process">Training Process</a></li>
    </ul>
  </li>
  <li><a href="#conclusions-and-challenges">Conclusions and Challenges</a></li>
</ul>

<h2 id="overview">Overview</h2>
<p>
In this project, we use Q-Learning to train a Pac-Man agent that can navigate different mazes while avoiding ghosts and consuming power pellets. The agent learns an optimal policy by exploring the environment and receiving rewards for its actions. We have designed the state space, reward function, and training process to achieve an intelligent agent capable of handling various mazes.
</p>

<h2 id="getting-started">Getting Started</h2>
To get started with the project, follow these steps:

1. Clone the repository to your local machine:

```bash
$ git clone https://github.com/username/pacman-q-learning-agent.git
$ cd pacman-q-learning-agent
```

2. Install Python 3.9 or higher.
3. Install the required dependencies:

```bash
$ pip install -r requirements.txt
```

4. Run the main Pac-Man game:


```bash
$ python pacman.py
```

<h2 id="file-structure">File Structure</h2>
<pre>
pacman-q-learning-agent
├── README.md
├── busters.py
├── bustersAgents.py
├── bustersGhostAgents.py
├── crawler.py
├── distanceCalculator.py
├── environment.py
├── featureExtractors.py
├── game.py
├── ghostAgents.py
├── grading.py
├── graphicsCrawlerDisplay.py
├── graphicsDisplay.py
├── graphicsGridworldDisplay.py
├── graphicsUtils.py
├── gridworld.py
├── inference.py
├── keyboardAgents.py
├── labyrinths
├── layout.py
├── layouts
│   ├── <All the layout (.lay) files>
├── learningAgents.py
├── pacman.py
├── pacmanAgents.py
├── projectParams.py
├── qlearningAgents.py
├── qtable.8dir.txt
├── qtable.ini.txt
├── qtable.txt
├── testClasses.py
├── testParser.py
├── textDisplay.py
├── textGridworldDisplay.py
├── util.py
├── valueIterationAgents.py
└── write_init
</pre>

<h2 id="agent-design">Agent Design</h2>
The agent's design consists of three main components: state space, reward function, and the training process.

<h3 id="state-space">State Space</h3>
The state space includes the following attributes:

1. Relative position of the agent to the closest ghost (encoded as 1 to 8).
2. Legal action combinations that the agent can take (encoded as 1 to 15).

<h3 id="reward-function">Reward Function</h3>
The reward function consists of positive and negative rewards based on the agent's actions:

<ul>
  <li>
    Positive rewards:
    <ul>
      <li>Matching the agent's position with a ghost: +30 reward (as it is the objective we want the agent to learn).</li>
    </ul>
  </li>
  <li>
    Negative rewards
    <ul>
      <li>Matching the agent's antepenultimate position and the current position: -15 reward (to prevent loops).</li>
      <li>The agent's position does not match a ghost, but it is approaching: -1 reward.</li>
    </ul>
  </li>
</ul>

<h3 id="training-process">Training Process</h3>
The agent is trained progressively through different mazes, starting from the simplest maze and gradually moving towards more complex mazes. This ensures that the agent generalizes its learning across different environments.

<h2 id="conclusions-and-challenges">Conclusions and Challenges</h2>
Throughout the development of this project, we have faced several challenges, mainly related to selecting the appropriate state space attributes and designing the reward function. We have learned the importance of choosing the right attributes and reward function for the agent to learn efficiently.

We have also realized the importance of balancing exploration and exploitation during the training process. The agent should explore enough to learn the optimal policy and then gradually increase its exploitation to make more informed decisions.
