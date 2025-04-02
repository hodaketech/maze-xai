# maze-XRL
**_Reward Composition_** is a concept in Reinforcement Learning (RL) and reward optimization systems. It refers to the process of constructing a reward function by combining multiple reward components.

1. Combine multiple objectives: In many RL problems, the goal is not to optimize a single factor but to balance multiple objectives.
2. Flexibly adjust strategies: By using Reward Composition, we can change the weight of each component to modify the agent’s behavior.
3. Enable easy scalability: When a new factor needs optimization, we can simply add a new reward component without redesigning the entire system.

## Result
The result obtained is the shortest path of the "agent" in the maze. It is drawn from the Q-Values obtained in the Q-Learning algorithm.
We need to provide specific maze data. The code will train for 1000 random episodes to find the Q-Values (up, down, left, right) so that at each point, the "agent" can decide which direction to go next.
After training agents with 1000 episodes, the result is included in a heat map (Heatmap) that indicates the maze, the synthetic Q values ​​are included in the Excel file, and the shortest path from the starting point (0, 0) to the destination point (9, 9).

## General Formula & Result of Demo
[./maze-XRL/screenshots/]
[./maze-XRL/excel-results/]

## Punishment and Reward
- Reaching the goal (line 79): +
- Hitting a blocked (line 81): -
- Turning around (line 77): -
- Consecutive safe actions (line 92): +

*Note: The value of point is not fixed, it can be adjusted based on the large of the maze and the difficulty of the game.*

**Recommend**

_In grid 10x10:_
> - Reaching the goal: +30
> - Hitting a blocked: -1
> - Turning around: -1
> - 2 Consecutive safe actions: +2  

_In grid 20x20:_
> - Reaching the goal: +1000
> - Hitting a blocked: -2
> - Turning around: -2
> - 2 Consecutive safe actions: +10  

## How to run this code ?
1. Clone the repository:  
`git clone https://github.com/nguyenthanhtin0712/maze-XRL.git`
2. Install the required packages:  
`pip3 install numpy matplotlib openpyxl gym`  
3. Choose **drQ-main.py** to run
4. Check the grid_size to change the size of the maze (line 189)
5. Run the code:  
> If the grid_size=10   
`python drQ-main.py maze10.txt`  
> If the grid_size=20  
`python drQ-main.py maze20.txt`  
6. The result will be displayed in the console, you can compare it with the result in the **_screenshot_**.

*Note: The colors on the map will change depending on the Q-Values based on a color scale (reflecting reward and penalty variations).*



