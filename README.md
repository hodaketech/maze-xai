# maze-XRL
**Reward Composition** is a concept in Reinforcement Learning (RL) and reward optimization systems. It refers to the process of constructing a reward function by combining multiple reward components.

1. Combine multiple objectives: In many RL problems, the goal is not to optimize a single factor but to balance multiple objectives.
2. Flexibly adjust strategies: By using Reward Composition, we can change the weight of each component to modify the agent’s behavior.
3. Enable easy scalability: When a new factor needs optimization, we can simply add a new reward component without redesigning the entire system.

## General Formula & Result
[./maze-XRL/screenshots]

## Punishment and Reward
- Reaching the goal: +
- Hitting a blocked: -
- Turning around: -
- Consecutive safe actions: +

*Note: The value of point is not fixed, it can be adjusted based on the large of the maze and the difficulty of the game.*

## How to run this code ?



