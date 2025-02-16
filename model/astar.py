import heapq
import random


weather_penalties = {
    "clear": 0,
    "light_wind": 2,
    "strong_wind": 5,  # Reduced penalty to avoid too many blocked paths
    "snow": float('inf'),  # Impassable
    "hail": float('inf')   # Impassable
}


def generate_grid():
    """
    Generates a 10x10 grid filled with zeros (open space).
    Ensures there is at least one guaranteed open path.
    """
    grid = [[0 for _ in range(10)] for _ in range(10)]

    # Randomly place some obstacles (1s) but keep the main diagonal open
    for _ in range(random.randint(5, 15)):  # Add between 5 to 15 obstacles
        x, y = random.randint(0, 9), random.randint(0, 9)
        if (x, y) != (0, 0) and (x, y) != (9, 9):  # Don't block start/goal
            grid[x][y] = 1  # Mark obstacle

    return grid


def generate_weather():
    """
    Generates a 10x10 grid of random weather conditions but ensures (0,0) and (9,9) are clear.
    """
    weather_types = (
        ["clear"] * 60 +  # 60% chance for clear skies
        ["light_wind"] * 20 +  # 20% chance for light wind
        ["strong_wind"] * 15 +  # 15% chance for strong wind
        ["snow"] * 3 +  # 3% chance for snow (rare)
        ["hail"] * 2  # 2% chance for hail (very rare)
    )

    weather_grid = [[random.choice(weather_types) for _ in range(10)] for _ in range(10)]

    # Ensure start and goal positions are always clear
    weather_grid[0][0] = "clear"
    weather_grid[9][9] = "clear"

    return weather_grid


class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (x, y)
        self.parent = parent  # Parent node for path reconstruction
        self.g = 0  # Cost from start node
        self.h = 0  # Estimated cost to goal
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f  # Required for heapq (priority queue)


def astar(grid, start, goal, weather_data):
    open_list = []
    closed_list = set()

    start_node = Node(start)
    goal_node = Node(goal)

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Reverse path from start to goal

        # Define possible moves (Up, Down, Left, Right + Diagonals)
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        random.shuffle(moves)  # Randomize move order to add variety

        for move in moves:
            new_x, new_y = current_node.position[0] + move[0], current_node.position[1] + move[1]
            new_position = (new_x, new_y)

            if new_x < 0 or new_y < 0 or new_x >= len(grid) or new_y >= len(grid[0]):
                continue

            if grid[new_x][new_y] == 1:
                continue

            if new_position in closed_list:
                continue

            weather_condition = weather_data[new_x][new_y]
            weather_penalty = weather_penalties[weather_condition]

            if weather_penalty == float('inf'):
                continue

            new_node = Node(new_position, current_node)
            new_node.g = current_node.g + 1 + weather_penalty
            new_node.h = abs(new_x - goal_node.position[0]) + abs(new_y - goal_node.position[1])
            new_node.f = new_node.g + new_node.h

            existing_node = next((node for node in open_list if node.position == new_node.position), None)
            if existing_node and existing_node.g <= new_node.g:
                continue

            heapq.heappush(open_list, new_node)

    return None  # No path found


grid = generate_grid()
weather_data = generate_weather()

start = (0, 0)
goal = (9, 9)

path = astar(grid, start, goal, weather_data)


if path:
    print("Path Found:", path)
else:
    print("No Path Found!")
