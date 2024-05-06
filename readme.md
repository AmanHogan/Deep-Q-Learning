# Elevator Scheduling using Deep Q-Learning

## Summary/Objective (CSE 6369 - Reinforcement Learning)
Consider an extended version of the elevator problem from Project 2 where the number of possible persons to be waiting or in elevators is 10 and the capacity of each elevator is limited to 2 persons. This problem will have a significantly larger state space. Design an MDP for this problem and build a Deep Q-Learning agent to learn an elevator policy for this problem. You can use libraries to build the neural network. Here is a link to the previous Elevator MDP: https://github.com/AmanHogan/Q-SARSA-and-LAMDA-Learning

## Prerequisites
- A Windows 10 or Mac OS computer
- Internet access and administrator access
- Vscode
- `Python 2.7xx` or higher
- `matplotlib` and `numpy`

# How to Run / Usage
- Clone or download this repository
- Move the cloned or downloaded repo to a safe location like your Desktop
- Open the project in a code editor like Vscode
- run the program using: `python main.py`

There are optional command line paramaters you can use to change the behavior of the program:

1. `--num_floors` or `-nf`: Number of Floors
2. `--start_floors` or `-sf`: Start Floors
3. `--start_prob` or `-sp`: Start Floors Chance
4. `--exit_floors` or `-ef`: Exit Floors
5. `--exit_prob` or `-ep`: Exit Floors Chance
6. `--floors` or `-f`: Floors
7. `--floors_zero` or `-fz`: Floors Including Zero
8. `--iterations` or `-i`: Number of Iterations
9. `--timestep` or `-t`: Timestep
10. `--epsilon` or `-e`: Exploration Rate
11. `--gamma` or `-g`: Discount Factor
12. `--alpha` or `-a`: Learning Rate
13. `--arrival_rate` or `-ar`: Arrival Rate
14. `--epochs`: Epochs
15. `--max_steps`: Max Steps
16. `--buffer_capacity`: Max Replay Buffer Size
17. `--batch_size`: Size of Batch
18. `--epsilon_decay`: Epsilon Decay. Default at .998
19. `--min_epsilon`: Minimum Epsilon
20. `--verbose` or `-v`: A verbose level of 0, will coument nothing. A verbose level of 1, will print to the console but not log to a logfile. A verbose level of 2 prints out all output into the logfile.

## Authors
- Aman Hogan-Bailey

## Contributions and Referenes
- The University of Texas at Arlington
- Manfred Huber (2242-CSE-6369-001)
