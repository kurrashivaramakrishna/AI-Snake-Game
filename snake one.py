# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 00:06:54 2024

@author: Shiva
"""



import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# Pygame
pygame.init()

# Constants
GRID_SIZE = 30
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
WALL_COUNT = 0
FOOD_SPAWN_RATE = 0
FOOD_SPAWN_RATE = 0
# Environment
class SnakeEnvironment:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.snake1_positions = [[0, 0], [0, 1]]
        self.snake2_positions = [[grid_size - 1, grid_size - 1], [grid_size - 1, grid_size - 2]]
        self.food_positions = []
        self.wall_positions = self.generate_walls()
        self.score1 = 0
        self.score2 = 0
        self.spawn_food()
    def is_collision(self, snake, action):
        new_head = self.move_snake(snake, action)
        if new_head in snake or new_head in self.wall_positions:
            return True
        return False
    
    def reset(self):
        self.snake1_positions = [[0, 0], [0, 1]]
        self.snake2_positions = [[self.grid_size - 1, self.grid_size - 1], [self.grid_size - 1, self.grid_size - 2]]
        self.food_positions = []
        self.spawn_food()
        self.score1 = 0
        self.score2 = 0

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for pos in self.snake1_positions:
            state[pos[0], pos[1]] = 1
        for pos in self.snake2_positions:
            state[pos[0], pos[1]] = 0.5
        for pos in self.food_positions:
            state[pos[0], pos[1]] = 0.8
        for pos in self.wall_positions:
            state[pos[0], pos[1]] = 0.3
        return state.flatten()
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_closest_food(self, snake_head):
     min_distance = float('inf')
     closest_food = None
     for food in self.food_positions:
        distance = self.heuristic(snake_head, food)
        if distance < min_distance:
            min_distance = distance
            closest_food = food
     return closest_food

    def shortest_path(self, start, target):
       
        start = tuple(start)  
        target = tuple(target)
        open_set = set()
        open_set.add(start)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, target)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            if current == target:
                return self.get_direction_from_path(came_from, start, target)

            open_set.remove(current)
            
            
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = current[0] + dx, current[1] + dy
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, target)
                        if neighbor not in open_set:
                            open_set.add(neighbor)
            
        return None  # Return None if there is no path to the target
    
    def draw_rounded_snake(screen, snake_positions, color):
        for i, pos in enumerate(snake_positions):
            x, y = pos[1] * CELL_SIZE, pos[0] * CELL_SIZE
            if i == 0:
            # Head of the snake, draw a circle
                pygame.draw.circle(screen, color, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 2)
            else:
            # Body of the snake, draw rectangles and circles for turns
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                if i < len(snake_positions) - 1:
                # Check for turns and draw circles
                    next_pos = snake_positions[i + 1]
                    if next_pos[0] != pos[0] or next_pos[1] != pos[1]:
                        pygame.draw.circle(screen, color, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 2)
    
    def get_direction_from_delta(self, dx, dy):
        # Convert dx, dy to a direction
        if dx == -1: return 1  # Up
        elif dx == 1: return 3  # Down
        elif dy == -1: return 0  # Left
        elif dy == 1: return 2  # Right
        return None
    
    def get_direction_from_path(self, came_from, start, target):
        # Reconstructs the path and returns the first move direction
        current = target
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        if path:
            next_step = path[0]
            dx, dy = next_step[0] - start[0], next_step[1] - start[1]
            return self.get_direction_from_delta(dx, dy)
        return None

    def step(self, action1, action2):
        if action1 is not None:
            new_head1, reward1, done1 = self.update_snake(self.snake1_positions, action1)
        else:
            new_head1, reward1, done1 = self.snake1_positions[0], 0, False

        if action2 is not None:
            new_head2, reward2, done2 = self.update_snake(self.snake2_positions, action2)
        else:
            new_head2, reward2, done2 = self.snake2_positions[0], 0, False

        done = done1 or done2

        if random.uniform(0, 1) < FOOD_SPAWN_RATE:
            self.spawn_food()

        self.score1 += reward1
        self.score2 += reward2

        return self.get_state(), reward1, reward2, done

    def move_snake(self, snake_positions, action):
        if action == 0:
            return [snake_positions[0][0], (snake_positions[0][1] - 1) % self.grid_size]
        elif action == 1:
            return [(snake_positions[0][0] - 1) % self.grid_size, snake_positions[0][1]]
        elif action == 2:
            return [snake_positions[0][0], (snake_positions[0][1] + 1) % self.grid_size]
        elif action == 3:
            return [(snake_positions[0][0] + 1) % self.grid_size, snake_positions[0][1]]

    def update_snake(self, snake_positions, action):
        new_head = self.move_snake(snake_positions, action)

    # Check for collisions with obstacles and the other snake
        if new_head in snake_positions or new_head in self.wall_positions or new_head in self.snake1_positions:
            return new_head, 0, True  # Game over

        if new_head[0] < 0 or new_head[0] >= self.grid_size or new_head[1] < 0 or new_head[1] >= self.grid_size:
            return new_head, 0, True  # Game over

    # Add new head position
        snake_positions.insert(0, new_head)

    # Check if snake has eaten food
        if new_head in self.food_positions:
            self.food_positions.remove(new_head)
            self.spawn_food()
            return new_head, 1, False  # Snake ate food, increase score

    # If no food is eaten, remove the last segment
        snake_positions.pop()
        return new_head, 0, False

    def spawn_food(self):
        empty_positions = [
            [i, j]
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if [i, j] not in self.snake1_positions
               and [i, j] not in self.snake2_positions
               and [i, j] not in self.food_positions
               and [i, j] not in self.wall_positions
        ]

        for _ in range(2):  # Spawn two pieces of food
            if empty_positions:
                new_food = random.choice(empty_positions)
                self.food_positions.append(new_food)
                empty_positions.remove(new_food)

    def generate_walls(self):
        walls = set()
        while len(walls) < WALL_COUNT:
            wall_position = random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2)
            if wall_position not in walls and wall_position not in self.snake1_positions and wall_position not in self.snake2_positions:
                walls.add(wall_position)
        return list(walls)

# Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = []
        self.replay_buffer = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.batch_size = 32

    def select_action(self, state, snake_positions):
        safe_actions = [action for action in range(4) if not env.is_collision(snake_positions, action)]

        if not safe_actions:  # If no safe actions, stop to avoid collision
            return None

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(safe_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                # Filter out unsafe actions
                safe_q_values = [q_values[i] if i in safe_actions else -float('inf') for i in range(4)]
                return torch.argmax(torch.tensor(safe_q_values)).item()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)

    # Filter out samples where action is None
        filtered_minibatch = [sample for sample in minibatch if sample[1] is not None]

        if not filtered_minibatch:
            return  # Skip this round of training if no valid samples

        states, actions, rewards, next_states, dones = zip(*filtered_minibatch)

    # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

    # Rest of the code in replay() remains the same


    def update_epsilon(self, episode):
        self.epsilon = max(0.01, self.initial_epsilon - 0.001 * episode)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Environment, agent, and Pygame
env = SnakeEnvironment(grid_size=GRID_SIZE)
state_size = env.grid_size * env.grid_size
action_size = 4
agent = DQNAgent(state_size, action_size)

# Pygame window
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Snake Game")

# Main game loop
# Main game loop
clock = pygame.time.Clock()
running = True

# Initialize actions for both snakes


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    current_state = env.get_state()

    # AI control for both snakes
    action1 = agent.select_action(current_state, env.snake1_positions)
    action2 = agent.select_action(current_state, env.snake2_positions)

    # Check if actions are None (no safe move), and skip the step if so
    if action1 is None and action2 is None:
        continue
    if action1 is None and action2 is None:
        continue

    next_state, reward1, reward2, done = env.step(action1, action2)

    if action1 is not None:
        agent.remember(current_state, action1, reward1, next_state, done)

    if action2 is not None:
        agent.remember(current_state, action2, reward2, next_state, done)

    agent.replay()

    next_state, reward1, reward2, done = env.step(action1, action2)
    agent.remember(current_state, action1, reward1, next_state, done)
    agent.replay()

    if pygame.time.get_ticks() % 500 == 0:
        agent.update_target_network()            
    # User input for Snake 1
    closest_food = env.find_closest_food(env.snake1_positions[0])
    if closest_food is not None:
        direction_to_food = env.shortest_path(env.snake1_positions[0], closest_food)
        if direction_to_food is not None:
            action1 = direction_to_food
        else:
            action1 = agent.select_action(env.get_state())  # Fallback to DQN if no path found

    # AI control for Snake 2
    closest_food = env.find_closest_food(env.snake2_positions[0])
    if closest_food is not None:
        direction_to_food = env.shortest_path(env.snake2_positions[0], closest_food)
        if direction_to_food is not None:
            action2 = direction_to_food
        else:
            action2 = agent.select_action(env.get_state())  # Fallback to DQN if no path found

    next_state, reward1, reward2, done = env.step(action1, action2)
    agent.remember(env.get_state(), action1, reward1, next_state, done)
    agent.replay()

    if pygame.time.get_ticks() % 500 == 0:
        agent.update_target_network()

    screen.fill((128, 0, 255))
    
    
    for pos in env.snake1_positions:
        pygame.draw.rect(screen, (0, 255, 0), (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for pos in env.snake2_positions:
        pygame.draw.rect(screen, (0, 0, 255), (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for pos in env.food_positions:
        pygame.draw.rect(screen, (255, 0, 1), (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for pos in env.wall_positions:
        pygame.draw.rect(screen, (255, 0, 1), (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
   
    font = pygame.font.Font(None, 36)
    
    score1_text = font.render("Score Snake 1: {}".format(env.score1), True, (255, 255, 255))
    score2_text = font.render("Score Snake 2: {}".format(env.score2), True, (255, 255, 255))
    screen.blit(score1_text, (10, 10))
    screen.blit(score2_text, (10, 50))
    
    pygame.display.flip()
    clock.tick(2.5)

    if env.score1 >= 200:
        print("User reached 2 points. Game over.")
        running = False

pygame.quit()










# this is extended code for the snake
