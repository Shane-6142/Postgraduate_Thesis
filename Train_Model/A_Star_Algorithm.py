import numpy as np
import heapq

def build_grid_from_meta(meta, rows=10, cols=10):
    grid = np.zeros((rows, cols), dtype=np.int32)

    for x, y in meta["obstacles"]:
        grid[y, x] = 1

    return grid

def A_Star_trajectory(grid, start, goal):
    
    rows, cols = grid.shape
    Sqrt2 = np.sqrt(2)

    def in_bounds(pos):
        return 0 <= pos[0] < cols and 0 <= pos[1] < rows

    def is_walkable(pos):
        return grid[pos[1], pos[0]] == 0
    
    def h_score(pos, goal): # heuristic
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        return (Sqrt2 - 1) * min(dx, dy) + max(dx, dy)
    
    def get_neighbors(pos):
        directions = [
            [-1, 0],[1, 0],[0, -1],[0, 1],
            [-1, -1],[-1, 1],[1, -1],[1, 1]
        ]
        neighbors = []
        for d in directions:
            new_pos = (pos[0] + d[0], pos[1] + d[1])
            if not in_bounds(new_pos) or not is_walkable(new_pos):
                continue

            if d[0] != 0 and d[1] != 0:
                side1 = (pos[0] + d[0], pos[1])
                side2 = (pos[0], pos[1] + d[1])
                if in_bounds(side1) and in_bounds(side2):
                    if (not is_walkable(side1)) and (not is_walkable(side2)):
                        continue

            neighbors.append(new_pos)

        return neighbors
    
    open_heap = []
    previous_pos = {}
    g_score = {start: 0.0}
    closed_set = set()
    heapq.heappush(open_heap, (h_score(start, goal) + g_score[start], start))

    while open_heap:
        current_f, current_pos = heapq.heappop(open_heap)
        if current_pos in closed_set:
            continue
        if current_pos == goal:
            path = [current_pos]
            while current_pos in previous_pos:
                current_pos = previous_pos[current_pos]
                path.append(current_pos)
            path.reverse()
            return path

        closed_set.add(current_pos)

        for neighbor in get_neighbors(current_pos):
            if (neighbor[0] != current_pos[0] and neighbor[1] != current_pos[1]):
                cost = Sqrt2
            else:
                cost = 1
            tentative_g = g_score[current_pos] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                previous_pos[neighbor] = current_pos
                g_score[neighbor] = tentative_g
                f_score = g_score[neighbor] + h_score(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return

def get_step(path):
        step = []
        direction_map = {
            (0, 1): "right",
            (0, -1): "left",
            (-1, 0): "up",
            (1, 0): "down",
            (-1, 1): "right_up",
            (-1, -1): "left_up",
            (1, 1): "right_down",
            (1, -1): "left_down"
        }
        for i in range(1, len(path)):
            prev_pos = path[i - 1]
            curr_pos = path[i]
            d_x = curr_pos[0] - prev_pos[0]
            d_y = curr_pos[1] - prev_pos[1]
            action = direction_map.get((d_y, d_x), "unknown")
            step.append(action)

        # if target_type != "exit":
        #     step.append("pick")

        return step
