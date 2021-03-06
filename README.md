# Project 2: Continuous Control

## Project Details
The goal of this project is to train a double-jointed arm to move to target locations using the Deep Deterministic Policy Gradients algorithm ([DDPG](https://arxiv.org/abs/1509.02971)).

![alt text](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

## Environment Details
The environment consists of a single agent, an observation space of 33 dimensions (which contains the arm's position, rotation, velocity and angular velocities), and a continuous action space. At each time step, the agent outputs an action vector of four dimensions (which corresponds to the torque applied to the both joints), and these values range between -1 and 1.

The task of the agent is to maintain its position at the target location for as many time steps as possible. The agent receives a reward of +0.1 for each time step the agent's hand is in the goal position.

The task is episodic, and to solve the environment the agent must get an average score greater than 30 over 100 consecutive episodes.

## Getting Started
You will need to set up your python environment.
1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__:
    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    - __Windows__:
    ```bash
    conda create --name drlnd python=3.6
    activate drlnd
    ```
2. Perform a minimal install of [OpenAI](https://github.com/openai/gym) `gym` with:
```
pip install gym
```
  * Install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control)
  * Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d)
3. Clone the Udacity's Deep Reinforcement Learning repository
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
4. Create an [IPython kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for `drlnd` environment
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
5. Before running code in a notebook, change the kernel to `drlnd` environment by using the drop-down `Kernel` menu.
6. For this project you will need to download the pre-built environment prepared by Udacity, and you can download it from one of the links below. You need to download the file that matches your operating system:

   - Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
   - Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
   - Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
   - Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

7. Download this repository within your working directory.

## Dependencies
* [Python 3.6](https://www.python.org/downloads/release/python-360/)
* [UnitiyEnvironment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
* [Pytorch](https://pytorch.org/)
* [Numpy](http://www.numpy.org/)
* [Deque](https://docs.python.org/3/library/collections.html)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Summary](https://github.com/sksq96/pytorch-summary/)

## How to use

1. Activate the `drlnd` conda Environment
```
source activate drlnd
```
2. Open the `Continuous_Control.ipynb` Jupyter notebook
3. Run the cells under steps 1 & 2.
4. Run cells under steps 5 & 6 if you want to train your agent. Else, run cell 7 to watch a trained agent using the weights in this repository `checkpoint_actor.path`.

## Files
* `Continuous_Control.ipynb` This is the Jupyter notebook that contains the implementation of the DDPG algorithm.
* `ddpg_agent.py` This Python file contains two major classes: `Agent` and `ReplayBuffer`. The `Agent` class contains an `act` method used to return an action for a given state and current policy. Noise was sampled from an [Ornstein-Uhlenbeck](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823) process to improve the learning efficiency in a continuous action space. The `learn` method is used to update both the actor and critic network parameters given a batch of experience tuples. The `ReplayBuffer` class has an `add` method to add a new experience to the memory buffer, and a `sample` method used to randomly fetch a batch of experiences from memory.
* `model.py` This Python file contains the deep neural networks (DNN) defined for the actor and the critic. Both networks have two hidden layers (first layer has 256 nodes and the second 128 nodes). The actor DNN maps 33 input states to a four dimensional vector of actions. A hyperbolic tangent activation function is used at the output of this network. The critic network maps state and action pairs to Q-values and a linear activation function is used at the output.
* `checkpoint_actor.pth` & `checkpoint_critic.pth` These files contain the DNN weights of the trained actor and critic, respectively.


## License
The source code is released under an [MIT license.](https://opensource.org/licenses/MIT)
## Acknowledgements
I would like to thank the Udacity community for the technical support and for providing coding exercises that helped me understand the implementation of this algorithm.

## Author
Andres Campos
