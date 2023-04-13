  <h1 id="title">Creating a Pac-Man Agent using RL (Q-Learning)</h1>
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#file-structure">File Structure</a></li>
    <li><a href="#reinforcement-function">Reinforcement Function</a></li>
    <li><a href="#state-attribute-selection">State Attribute Selection</a></li>
    <li><a href="#conclusions-and-challenges">Conclusions and Challenges</a></li>
  </ul>

  <h2 id="overview">Overview</h2>
  <p>
    This project aims to create an autonomous Pac-Man agent using Reinforcement Learning (specifically, Q-Learning). The agent learns how to play the game by exploring different states and actions, receiving rewards or penalties depending on its performance.
  </p>

  <h2 id="getting-started">Getting Started</h2>
  <p>
    To get started with the project, clone the repository and follow the instructions to set up the required dependencies. Once everything is set up, you can run the main script to see the Pac-Man agent in action.
  </p>

  <h2 id="file-structure">File Structure</h2>
  <pre>
.
├── README.md
├── __pycache__
│   ├── (various .pyc files)
├── (various .py files)
├── labyrinths
├── layouts
│   ├── (various .lay files)
└── write_init
  </pre>

  <h2 id="reinforcement-function">Reinforcement Function</h2>
  <p>
    The implemented reinforcement function provides our agent with negative rewards (penalties) in the following cases: when the agent's antepenultimate position matches the current position, and when the agent's position does not match a ghost's position but is getting closer. Our agent is given positive rewards (prizes) when its position matches a ghost's position.
  </p>

  <h2 id="state-attribute-selection">State Attribute Selection</h2>
  <p>
    Throughout the implementation process, we experimented with various attributes to describe each state. The final model includes the relative position of the agent to the closest ghost and the legal actions the agent can take.
  </p>

  <h2 id="conclusions-and-challenges">Conclusions and Challenges</h2>
  <p>
    Developing this project allowed us to appreciate the importance of choosing the right state attributes and designing an effective reinforcement function. A suboptimal choice can significantly hinder the agent's learning process. Furthermore, we discovered that balancing exploration and exploitation during the learning process is crucial for obtaining a well-performing agent.
  </p>
