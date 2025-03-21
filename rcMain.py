import numpy as np
import random
import matplotlib.pyplot as plt
import sys

class RewardPathFinder:
    def __init__(self, grid_size, maze_file):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.end = (grid_size - 1, grid_size - 1)
        self.blocked_points = self.load_maze(maze_file)
        self.q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions: up, down, left, right
        self.epsilon = 1.0  # exploration rate
        self.alpha = 0.1    # learning rate
        self.gamma = 0.9    # discount factor
        self.best_path = []  # To store the best path taken
        self.consecutive_safe_actions = 0  # Counter for consecutive safe actions

    def load_maze(self, maze_file):
        blocked_points = []
        with open(maze_file, 'r') as file:
            for i, line in enumerate(file):
                for j, char in enumerate(line.strip().split()):
                    if char == '#':
                        blocked_points.append((i, j))
        return blocked_points

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

    def train(self, episodes, log_random_episode=False):
        best_total_reward = -float('inf')
        for episode in range(episodes):
            state = self.reset()
            total_reward = 0
            current_path = [state]  # Track the current path taken
            last_action = None  # Track the last action taken
            self.consecutive_safe_actions = 0  # Reset counter at the start of each episode
            while not self.is_terminal(state):
                action = self.choose_action(state)
                next_state = self.take_action(state, action)
                current_path.append(next_state)  # Add the state to the current path
                reward = 0
                
                # Check for 180-degree turn penalty
                if last_action is not None and ((action == 0 and last_action == 1) or (action == 1 and last_action == 0) or (action == 2 and last_action == 3) or (action == 3 and last_action == 2)):
                    reward -= 1  # Penalty for turning around
                elif self.is_terminal(next_state):
                    reward += 30  # reward for reaching the end
                elif next_state in self.blocked_points:
                    reward -= 1  # penalty for hitting a blocked point (walls)
                    self.consecutive_safe_actions = 0  # Reset counter
                else:
                    reward += 0  # neutral reward for other moves
                    self.consecutive_safe_actions += 1  # Increment counter for safe actions
                    if self.consecutive_safe_actions == 2:
                        reward += 1  # Reward for 3 consecutive safe actions
                        self.consecutive_safe_actions = 0  # Reset counter
                    elif self.consecutive_safe_actions == 3:
                        reward += 2
                        self.consecutive_safe_actions = 0

                self.update_q_table(state, action, reward, next_state)
                state = next_state
                last_action = action  # Update the last action taken
                total_reward += reward
            
            # Check if the current episode is the best one
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                self.best_path = current_path  # Update the best path
            if episode == 999 and log_random_episode:  # Log episode 1000
                print(f"Logging actions for the last episode:")
                for i in range(len(current_path) - 1):
                    action = self.choose_action(current_path[i])  # Choose action based on current state
                    next_state = self.take_action(current_path[i], action)  # Determine next state
                    q_value = self.q_table[current_path[i][0], current_path[i][1], action]  # Get Q-value for the action
                    print(f"State: {current_path[i]}, Action: {action} (Q-value: {q_value}), Next State: {next_state}")
                    print(f"Choosing action based on Q-value: {q_value}")
                    if next_state == self.end:
                        print("Reached the end!")
                    elif next_state in self.blocked_points:
                        print("Hit a blocked point!")
                    else:
                        print("Continuing...")
                        print(f"Choosing action based on Q-value: {q_value}")
            print(f"Episode {episode + 1}: Total Reward: {total_reward}\n")
            if episode == episodes - 1:  # Print Q-Table after the last episode
                print("Final Q-Table:")
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        print(f"State {i, j}: Q-Values: {self.q_table[i, j]}")

    def get_shortest_path(self):
        state = self.start
        shortest_path = [state]
        visited = set()

        while state != self.end:
            visited.add(state)
            best_action = np.argmax(self.q_table[state[0], state[1]])
            next_state = self.take_action(state, best_action)

            if next_state in visited or next_state in self.blocked_points or not (0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size):
                break

            shortest_path.append(next_state)
            state = next_state

        return shortest_path

    def visualize(self, show_q_values=False):
        q_values = np.max(self.q_table, axis=2)
        plt.imshow(q_values, cmap='hot', interpolation='nearest')
        if show_q_values:
            plt.colorbar(label='Q-Values')
        else:
            plt.colorbar(label='Heatmap')
        plt.title('Heatmap of Actions' if not show_q_values else 'Q-Values Heatmap')
        
        # Marking the start, end, and blocked points
        plt.scatter(self.start[1], self.start[0], marker='o', color='green', label='Start')
        plt.scatter(self.end[1], self.end[0], marker='x', color='blue', label='End')
        for point in self.blocked_points:
            plt.scatter(point[1], point[0], marker='s', color='red', label='Blocked' if point == self.blocked_points[0] else "")
        
        # Draw the shortest path
        shortest_path = self.get_shortest_path()
        if shortest_path:
            path_x, path_y = zip(*shortest_path)
            plt.plot(path_y, path_x, marker='o', color='cyan', label='Shortest Path')
        
        plt.legend()
        plt.grid(True)  # Add grid for better visualization
        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))
        plt.show()
        plt.close()  # Close the figure to prevent it from displaying multiple times

        if show_q_values:
            print("Q-Values for each action:")
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    print(f"State {i, j}: Q-Values: {self.q_table[i, j]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rcMain3.py <maze_file>")
        sys.exit(1)

    maze_file = sys.argv[1]
    grid_size = 10
    agent = RewardPathFinder(grid_size, maze_file)

    # Training the robot
    agent.train(1000)

    # Visualizing the results
    agent.visualize()
    
# cd Blackbox
# python rcMain.py maze.txt