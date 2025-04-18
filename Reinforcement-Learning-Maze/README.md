### Improving maze using drqModel

The values for the penalties and rewards are defined in class *Maze* in *maze.py*:
```python
    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze
```
![Maze](https://github.com/erikdelange/Reinforcement-Learning-Maze/blob/master/bestmove.png)

Package *models* contains the following models:
1. *RandomModel* is the simplest model and just selects the next move randomly. It is [dumb all over](https://www.youtube.com/watch?v=DR_wf92A8E4) and learns nothing. You are lucky if you get to the exit using this model.
2. *QTableModel* uses a table to record the value of each (state, action) pair. For a state the highest value indicates the most desirable action. These values are constantly refined during training. This is a fast way to learn a policy.
3. *SarsaTableModel* uses a similar setup as the previous model, but takes less risk during learning (= on-policy learning).
4. *QTableTraceModel* is an extension of the QTableModel. It speeds up learning by keeping track of previously visited state-action pairs, and updates their values as well although with a decaying rate.
5. *SarsaTableTraceModel* is a variant of SarsaTableModel but adds an eligibility trace, just as QTableTraceModel. 
6. *QReplayNetworkModel* is a simple neural network which learns the relation between a state and the corresponding values by replaying previous moves. It is significantly slower than all other models, and an overkill for a problem with such a small state space. As an extra feature after learning it saves the model to disk so this can be loaded later for a next game. This is typically how you would use a neural network in a real world situation where training is separated from actual use. 
7. *drQModel* is a good model than other models
The table below gives an impression of the relative performance of each of these models (on my PC):

| Model | Trained | Average no of episodes | Average training time |
| --- | --- | --- | --- | 
| QTableModel | 10 times | 149.5 | 16.5 sec |
| QTableTraceModel | 10 times | 87.0 | 5.2 sec |
| SarsaTableModel | 10 times | 114.0 | 11.7 sec |
| SarsaTableTraceModel | 10 times | 73.0 | 5.5 sec |
| QReplayNetworkModel | 10 times | 113.5 | 4 min 48 sec |
| drQModel | 10 times | 60.5 | 4 min 2 sec |

Requires matplotlib, numpy, keras and tensorflow.
******HOW TO RUN CODE
run "python main.py"
