import numpy as np
import heapq


class AStarPlanner:
    def __init__(self, costs=None):
        if costs is None:
            costs = {
                "rover_track": 0.5,
                "soil": 1.0,
                "bedrock": 2.0,
                "sand": 3.0,
                "large_rock": 10,
            }
        self.costs = costs

    def plan(self, seg_map):
        label_map = {
            0: "soil",
            1: "bedrock",
            2: "sand",
            3: "large_rock",
            4: "rover_track",
        }
        cost_grid = np.vectorize(lambda x: self.costs[label_map[x]])(seg_map)
        h, w = seg_map.shape
        g_score = np.full((h, w), np.inf)
        came_from = {}
        frontier = []
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for row in range(h):
            g_score[row, 0] = cost_grid[row, 0]
            heapq.heappush(frontier, (g_score[row, 0], row, 0))

        while frontier:
            cost, row, col = heapq.heappop(frontier)

            if col == w - 1:
                return self.reconstruct_path(came_from, row, col)

            for move in offsets:
                nrow, ncol = row + move[0], col + move[1]
                if nrow < 0 or nrow >= h or ncol < 0 or ncol >= w:
                    continue
                new_g = cost + cost_grid[nrow, ncol]
                if new_g < g_score[nrow, ncol]:
                    g_score[nrow, ncol] = new_g
                    came_from[(nrow, ncol)] = (row, col)
                    heapq.heappush(frontier, (new_g + (w - 1 - ncol), nrow, ncol))
        return None

    def reconstruct_path(self, came_from, row, col):
        path = [(row, col)]
        while (row, col) in came_from:
            row, col = came_from[(row, col)]
            path.append((row, col))

        path.reverse()
        return path
