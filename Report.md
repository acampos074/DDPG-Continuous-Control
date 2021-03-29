## Report
### Project description
The goal of this project is to train an agent to move to target locations for as long as possible. A reward of +0.1 is provided for each time step the agent's hand remains in the target position. The task is episodic, and the environment is considered solved once the agent obtains an average score of +30 over 100 consecutive episodes.

The agent is a double-jointed arm, the observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. The action space is a four dimensional vector corresponding to torques applied to two joints. Every entry in the action space is a number between -1 and 1.

### Learning algorithm

The algorithm used to solve this problem is the Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/abs/1509.02971)). This is a model-free, off-policy, and type of actor-critic algorithm.

The agent was trained for a fixed number of episodes and a fixed episode length. For a given state, the actor determines the optimal action deterministically.  [Ornstein-Uhlenbeck process](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823) noise was added to the action output from the local actor network to build an exploration policy. The critic uses a state-action pair (were the action is determined by the actor) to estimate the optimal action-value function. This action-value function is then used to train the actor. Fig. 1 illustrates a schematic representation of this actor-critic interaction.

![alt-text](https://raw.githubusercontent.com/acampos074/DDPG-Continuous-Control/master/Figures/actor_critic.png)

**Figure 1 | Schematic illustration of the DDPG algorithm.** DDPG is a type of actor-critic method. The actor observes a state and determines the best action. The critic observes a state-action pair and determines an action-value function.

The algorithm uses three key ideas:
* Experience Replay
* Soft Updates
* Fixed Q-Targets  

Experience replay helps to break correlations from sequential observations. During training, a set of random experiences are fetched from the memory buffer and these are used to break these correlations. To improve the stability of training, two sets of networks were used (e.g. local and target) for the actor and the critic. Fig. 2 summarizes the network architecture used for the actor and the critic.

![alt-text](https://raw.githubusercontent.com/acampos074/DDPG-Continuous-Control/master/Figures/DNN_v2.png)

**Figure 2 | Neural network architecture.** The input to the actor neural network consist of a 33x1 tensor of states, followed by two fully connected hidden layers with 256 and 128 nodes. The output layer consist of a 4x1 tensor of actions. Each hidden layer is followed by a rectified linear unit (ReLU) activation function. The output layer is followed by a hyperbolic tangent activation function. A batch size of 512 was used to compute each stochastic gradient decent update.

Next, a soft update approach helps to slowly adjust the weights of the target network using the weights of the local network. Fig. 3 illustrates how each network is updated.

![alt-text](https://raw.githubusercontent.com/acampos074/DDPG-Continuous-Control/master/Figures/actor_critic_local_target.png)

**Figure 3 | Actor-Critic Local and Target Networks.** The target networks use a soft update approach to increase the stability of training. The local actor network minimizes the loss function based on the action-value function determined by the local critic network. The local critic network minimizes the loss function of the mean square error between the target and expected action-value functions.

Lastly, fixed Q-Targets help to break correlations with the targets (i.e. it helps solve the problem of training with a moving target). Thus, the target weights of both actor and critic are updated less often than the weights of their corresponding local networks. Table 1 lists all the parameters used in this implementation.

#### **Table 1 | List of hyperparameters and their values**
| **Hyperparameter**      | **Value** | **Description**     |
| :---        |    :---   |  :--- |
| `BUFFER_SIZE`      | 100000       | Size of the replay memory buffer. [Adam algorithm](https://arxiv.org/abs/1412.6980) (a variant of stochastic gradient decent (SGD) algorithm) updates are sampled from this buffer.    |
| `BATCH_SIZE`   | 512        | Number of training cases used to compute each SGD update.      |
| `GAMMA`   | 0.99        | Gamma is the discount factor used in the Q-learning update.     |
| `TAU`   | 0.001  |  Tau is an interpolation parameter used to update the weights of the target network. |
| `LR_ACTOR`   | 0.0001  |  Learning rate of the actor.  |
|`LR_CRITIC`   | 0.0001  |  Learning rate of the critic. |
| `THETA_NOISE`   | 0.3  | Long-term mean of the [Ornstein-Uhlenbeck process](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823).  |
|`UPDATE_EVERY`   | 4  | The number of actions taken by the agent between successive SGD updates.  |


Fig. 4 illustrates the temporal evolution of the agent's score-per-episode.

![alt-text](https://raw.githubusercontent.com/acampos074/DDPG-Continuous-Control/master/Figures/scores.png)

**Figure 4 | Training curve tracking the agent's score.** The average scores (orange line) shows that the agent is able to receive a score of at least +30 over 380 episodes.
### Ideas of Future Work
Other ideas to further improve the agent's performance include:
* TNPG
* TRPO
* PPO


Another idea to further improve the training efficiency is to modify the DDPG algorithm to train multiple agents at the same time.
