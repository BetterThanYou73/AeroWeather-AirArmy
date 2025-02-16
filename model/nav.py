#!/usr/bin/env python
import os
import sys
import math
import random
import numpy as np
import requests
import heapq
import pickle
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier

#############################################
# GLOBAL VARIABLES
#############################################
GRID_SIZE = 15
WEATHER_API_KEY = "b865f8ab64997428792213c1280ae895" 
WIND_THRESHOLD = 6
RAIN_THRESHOLD = 1.5

# Use maximum workers available (for I/O-bound tasks)
MAX_WORKERS = (os.cpu_count() or 4) * 2

DEBUG = True  # Set to True to show debug maps

#############################################
# HELPER FUNCTIONS: Bounding Box & Grid Conversion
#############################################
def define_bounding_box(lat_start, lon_start, lat_goal, lon_goal, extra_margin=0.2):
    min_lat = min(lat_start, lat_goal)
    max_lat = max(lat_start, lat_goal)
    min_lon = min(lon_start, lon_goal)
    max_lon = max(lon_start, lon_goal)
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    min_lat -= lat_range * extra_margin
    max_lat += lat_range * extra_margin
    min_lon -= lon_range * extra_margin
    max_lon += lon_range * extra_margin
    return (min_lat, max_lat, min_lon, max_lon)

def latlon_to_grid(lat, lon, min_lat, max_lat, min_lon, max_lon, grid_size):
    lat = max(min_lat, min(lat, max_lat))
    lon = max(min_lon, min(lon, max_lon))
    frac_y = (lat - min_lat) / (max_lat - min_lat) if (max_lat - min_lat) != 0 else 0
    frac_x = (lon - min_lon) / (max_lon - min_lon) if (max_lon - min_lon) != 0 else 0
    gy = int(frac_y * (grid_size - 1))
    gx = int(frac_x * (grid_size - 1))
    return (gy, gx)

def grid_to_latlon(gy, gx, min_lat, max_lat, min_lon, max_lon, grid_size):
    frac_y = gy / (grid_size - 1)
    frac_x = gx / (grid_size - 1)
    lat = min_lat + frac_y * (max_lat - min_lat)
    lon = min_lon + frac_x * (max_lon - min_lon)
    return (lat, lon)

#############################################
# HELPER FUNCTIONS: Weather API Calls (Parallel)
#############################################
def fetch_cell_data(args):
    (y, x, min_lat, max_lat, min_lon, max_lon,
     grid_size, wind_thresh, rain_thresh, api_key) = args
    lat_step = (max_lat - min_lat) / grid_size
    lon_step = (max_lon - min_lon) / grid_size
    cell_lat = min_lat + (y + 0.5) * lat_step
    cell_lon = min_lon + (x + 0.5) * lon_step

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={cell_lat}&lon={cell_lon}&appid={api_key}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    wind_data = data.get("wind", {})
    wind_speed = wind_data.get("speed", 0.0)
    wind_deg = wind_data.get("deg", 0)
    radians = math.radians(wind_deg)
    wx = wind_speed * math.sin(radians)
    wy = -wind_speed * math.cos(radians)
    rain_1h = 0.0
    if "rain" in data and "1h" in data["rain"]:
        rain_1h = data["rain"]["1h"]
    snow_1h = 0.0
    if "snow" in data and "1h" in data["snow"]:
        snow_1h = data["snow"]["1h"]
    blocked = 1 if (wind_speed > wind_thresh or (rain_1h + snow_1h) > rain_thresh) else 0
    return (y, x, blocked, (wy, wx))

def generate_environment_multi_owm_concurrent(min_lat, max_lat, min_lon, max_lon,
                                               grid_size=GRID_SIZE,
                                               wind_threshold=WIND_THRESHOLD,
                                               rain_threshold=RAIN_THRESHOLD,
                                               api_key=WEATHER_API_KEY,
                                               max_workers=MAX_WORKERS):
    env = np.zeros((grid_size, grid_size), dtype=int)
    wind = np.zeros((grid_size, grid_size, 2), dtype=float)
    tasks = []
    for y in range(grid_size):
        for x in range(grid_size):
            tasks.append((y, x, min_lat, max_lat, min_lon, max_lon, grid_size, wind_threshold, rain_threshold, api_key))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_cell_data, tasks))
    for (y, x, blocked, (wy, wx)) in results:
        env[y, x] = blocked
        wind[y, x] = (wy, wx)
    return env, wind

#############################################
# HELPER FUNCTIONS: A* and Training Data
#############################################
def get_neighbors(node, grid_size):
    (y, x) = node
    for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < grid_size and 0 <= nx < grid_size:
            yield (ny, nx)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def a_star(env, wind, start, goal):
    grid_size = env.shape[0]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {(y,x): float('inf') for y in range(grid_size) for x in range(grid_size)}
    f_score = {(y,x): float('inf') for y in range(grid_size) for x in range(grid_size)}
    g_score[start] = 0
    f_score[start] = heuristic(start, goal)
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for nxt in get_neighbors(current, grid_size):
            cy, cx = current
            ny, nx = nxt
            if env[ny, nx] == 1:
                continue
            step_cost = 1.0 + np.hypot(*wind[ny, nx])
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score[nxt] = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_set, (f_score[nxt], nxt))
    return None

def path_to_state_action_pairs(path, wind, goal):
    pairs = []
    for i in range(len(path)-1):
        cy, cx = path[i]
        ny, nx = path[i+1]
        if ny < cy:
            action = 0  # up
        elif ny > cy:
            action = 1  # down
        elif nx < cx:
            action = 2  # left
        else:
            action = 3  # right
        wy, wx = wind[cy, cx]
        gy, gx = goal
        state = (cy, cx, wy, wx, gy, gx)
        pairs.append((state, action))
    return pairs

#############################################
# FALLBACK ROUTE FUNCTION
#############################################
def apply_action(cy, cx, action):
    if action == 0:
        return (cy-1, cx)
    elif action == 1:
        return (cy+1, cx)
    elif action == 2:
        return (cy, cx-1)
    else:
        return (cy, cx+1)

def model_based_route_with_fallback(model, env, wind, start, goal, max_steps=200):
    current = start
    path = []
    grid_size = env.shape[0]
    for _ in range(max_steps):
        path.append(current)
        if current == goal:
            return path
        cy, cx = current
        wy, wx = wind[cy, cx]
        gy, gx = goal
        state = np.array([cy, cx, wy, wx, gy, gx], dtype=float).reshape(1, -1)
        primary_action = model.predict(state)[0]
        fallback_order = [0, 1, 2, 3]
        fallback_order.remove(primary_action)
        candidate_actions = [primary_action] + fallback_order
        picked_action = None
        for action in candidate_actions:
            ny, nx = apply_action(cy, cx, action)
            if not (0 <= ny < grid_size and 0 <= nx < grid_size):
                continue
            if env[ny, nx] == 1:
                continue
            picked_action = action
            break
        if picked_action is None:
            return None
        else:
            ny, nx = apply_action(cy, cx, picked_action)
            current = (ny, nx)
    return None

#############################################
# MAP VISUALIZATION FUNCTION
#############################################
def visualize_latlon_route_on_map(latlon_route, title="LatLon Route", pad=100):
    """
    Visualize the lat/lon route on a contextily basemap, forcing a square aspect ratio
    and adding 'pad' units of extra space around all edges in Web Mercator coordinates.
    """
    import contextily as ctx
    import matplotlib.pyplot as plt
    from pyproj import Transformer

    if not latlon_route:
        print("No route to display on map.")
        return

    # Convert lat/lon -> Web Mercator
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = [], []
    for (lat, lon) in latlon_route:
        x, y = transformer.transform(lon, lat)  # (lon, lat) order for transform
        xs.append(x)
        ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Determine the max range to force a square bounding box
    range_x = x_max - x_min
    range_y = y_max - y_min
    range_max = max(range_x, range_y)

    # Center about midpoints
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    # Adjust bounding box so width == height == range_max
    x_min_new = x_mid - 0.5 * range_max
    x_max_new = x_mid + 0.5 * range_max
    y_min_new = y_mid - 0.5 * range_max
    y_max_new = y_mid + 0.5 * range_max

    # Add some padding around edges
    x_min_new -= pad
    x_max_new += pad
    y_min_new -= pad
    y_max_new += pad

    fig, ax = plt.subplots(figsize=(8,8))

    # Plot the route
    ax.plot(xs, ys, 'ro-', label='Route')

    # Force the new bounding box
    ax.set_xlim(x_min_new, x_max_new)
    ax.set_ylim(y_min_new, y_max_new)

    # Add the basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print("Falling back to OSM Mapnik due to error:", e)
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik)

    # Force a square aspect ratio
    ax.set_aspect("equal", "box")

    ax.set_title(title)
    ax.legend()
    plt.show()



#############################################
# PRINT ROUTE AS JSON-LIKE STRUCTURE
#############################################
def find_closest_point_index(latlon_route, user_lat, user_lon):
    """
    Returns the index of the route point in 'latlon_route' closest
    to (user_lat, user_lon) in simple lat/lon Euclidean distance.
    """
    best_idx = None
    best_dist_sq = float("inf")
    for i, (lat_c, lon_c) in enumerate(latlon_route):
        # We'll do a quick approximate distance in lat/lon degrees
        dy = lat_c - user_lat
        dx = lon_c - user_lon
        dist_sq = dx*dx + dy*dy
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_idx = i
    return best_idx


def link_start_goal_to_closest(latlon_route, start_lat, start_lon, goal_lat, goal_lon):
    """
    Finds the closest route point to the user's start lat/lon,
    and the closest route point to the user's goal lat/lon,
    then replaces those points with the actual user lat/lon.
    """
    if not latlon_route:
        return latlon_route
    idx_start = find_closest_point_index(latlon_route, start_lat, start_lon)
    latlon_route[idx_start] = (start_lat, start_lon)
    idx_goal = find_closest_point_index(latlon_route, goal_lat, goal_lon)
    latlon_route = latlon_route[idx_start:idx_goal]
    latlon_route.append((goal_lat, goal_lon))
    return latlon_route


def print_route_as_json(latlon_route):
    for idx, (lat, lon) in enumerate(latlon_route, 1):
        print({
            "name": f"Point {idx}",
            "coords": [lat, lon]
        })

#############################################
# nav_call FUNCTION: Main Entry Point
#############################################
def nav_call(start_lat, start_lon, goal_lat, goal_lon):
    # 1) Build environment around start/goal
    (min_lat, max_lat, min_lon, max_lon) = define_bounding_box(start_lat, start_lon, goal_lat, goal_lon, extra_margin=0.3)
    env, wind_field = generate_environment_multi_owm_concurrent(
        min_lat, max_lat, min_lon, max_lon,
        grid_size=GRID_SIZE,
        wind_threshold=WIND_THRESHOLD,
        rain_threshold=RAIN_THRESHOLD,
        api_key=WEATHER_API_KEY,
        max_workers=MAX_WORKERS
    )

    # 2) Convert user lat/lon to grid coordinates
    start_grid = latlon_to_grid(start_lat, start_lon, min_lat, max_lat, min_lon, max_lon, GRID_SIZE)
    goal_grid  = latlon_to_grid(goal_lat, goal_lon, min_lat, max_lat, min_lon, max_lon, GRID_SIZE)
    print("nav_call: Start grid:", start_grid, "Goal grid:", goal_grid)

    # 3) Run A* to get the expert route
    astar_path = a_star(env, wind_field, start_grid, goal_grid)
    print("A* path (grid coords):", astar_path)
    if not astar_path or len(astar_path) < 2:
        print("No valid A* route found. Exiting.")
        return []

    # 4) Build training data from the A* route and train the model
    pairs = path_to_state_action_pairs(astar_path, wind_field, goal_grid)
    X_data, y_data = [], []
    for (st, ac) in pairs:
        X_data.append(st)
        y_data.append(ac)
    X_np = np.array(X_data, dtype=float)
    y_np = np.array(y_data, dtype=int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_np, y_np)
    print("Trained model on route of length:", len(astar_path))
    
    # Save training data and model for later updates
    with open("training_data.pkl", "wb") as f:
        pickle.dump((X_np, y_np), f)
    with open("route_model_multi.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved model and training data.")

    # 5) Use the model to get a fallback route
    fallback_route = model_based_route_with_fallback(model, env, wind_field, start_grid, goal_grid, max_steps=200)
    print("Model fallback route (grid coords):", fallback_route)
    if fallback_route is None:
        print("No route found by the model. Possibly blocked environment.")
        return []

    # 6) Convert fallback route to lat/lon
    latlon_route = [grid_to_latlon(gy, gx, min_lat, max_lat, min_lon, max_lon, GRID_SIZE) for (gy, gx) in fallback_route]
    # Instead of forcibly snapping, link to the closest route points
    latlon_route = link_start_goal_to_closest(latlon_route, start_lat, start_lon, goal_lat, goal_lon)

    # 7) Print final route in JSON-like structure
    print("Final route in lat/lon:")
    print_route_as_json(latlon_route)

    return latlon_route


#############################################
# MAIN: For direct running
#############################################
if __name__ == "__main__":
    try:
        start_lat = float(input("Enter start latitude (default 53.521568): ") or "53.521568")
        start_lon = float(input("Enter start longitude (default -113.509583): ") or "-113.509583")
        goal_lat = float(input("Enter goal latitude (default 51.043966): ") or "51.043966")
        goal_lon = float(input("Enter goal longitude (default -114.060203): ") or "-114.060203")
    except Exception as e:
        print("Invalid input. Using default coordinates.")
        start_lat, start_lon = 53.521568, -113.509583
        goal_lat, goal_lon = 51.043966, -114.060203

    DEBUG = input("Enable debug visualization? (y/n, default n): ").strip().lower() == "y"

    route = nav_call(start_lat, start_lon, goal_lat, goal_lon)
    print("Returned route (lat/lon):")
    for point in route:
        print(point)