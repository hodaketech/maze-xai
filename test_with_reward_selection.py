import numpy as np
import random
import matplotlib.pyplot as plt

class RewardPathFinder:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.end = (grid_size - 1, grid_size - 1)
        self.reward_point = (2, 3)  # Reward point
        self.blocked_points = self.random_blocked_points()
        self.q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions: up, down, left, right
        self.epsilon = 1.0  # exploration rate
        self.alpha = 0.1    # learning rate
        self.gamma = 0.9    # discount factor

    def random_blocked_points(self):
        points = set()  
        while len(points) < 3:  # Ensure two blocked points
            point = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if point != self.start and point != self.end and point != self.reward_point:
                points.add(point)
        return list(points)

    def reset(self):
        return self.start

    def is_terminal(self, state):
        return state == self.end

    def get_actions(self, state):
        actions = []
        for action in range(4):
            next_state = self.take_action(state, action)
            if next_state not in self.blocked_points and 0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size:
                actions.append(action)
        return actions

    def take_action(self, state, action):
        if action == 0:  # up
            return (state[0] - 1, state[1])
        elif action == 1:  # down
            return (state[0] + 1, state[1])
        elif action == 2:  # left
            return (state[0], state[1] - 1)
        elif action == 3:  # right
            return (state[0], state[1] + 1)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.get_actions(state))
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
        td_delta = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.alpha * td_delta

    def train(self, episodes):
        for episode in range(episodes):
            state = self.reset()
            total_reward = 0
            while not self.is_terminal(state):
                action = self.choose_action(state)
                next_state = self.take_action(state, action)
                if self.is_terminal(next_state):
                    reward = 10  # reward for reaching the end
                elif next_state == self.reward_point:
                    reward = 1  # lower reward for reaching the reward point
                elif next_state in self.blocked_points:
                    reward = -100  # penalty for hitting a blocked point
                else:
                    reward = 0  # neutral reward for other moves

                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    def visualize(self):
        q_values = np.max(self.q_table, axis=2)
        plt.imshow(q_values, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Q-Values Heatmap')
        
        # Marking the start, end, and blocked points
        plt.scatter(self.start[1], self.start[0], marker='o', color='green', label='Start')
        plt.scatter(self.end[1], self.end[0], marker='x', color='blue', label='End')
        plt.scatter(self.reward_point[1], self.reward_point[0], marker='*', color='cyan', s=100, label='Reward')
        for point in self.blocked_points:
            plt.scatter(point[1], point[0], marker='s', color='red', label='Blocked' if point == self.blocked_points[0] else "")
        
        plt.legend()
        plt.grid(True)  # Add grid for better visualization
        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))
        plt.show()

# Parameters
grid_size = 5
robot = RewardPathFinder(grid_size)

# Training the robot
robot.train(1000)

# Visualizing the results
robot.visualize()
