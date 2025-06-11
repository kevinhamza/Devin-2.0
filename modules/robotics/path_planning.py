# Devin/modules/robotics/path_planning.py
# Purpose: Provides path planning capabilities for a robot to navigate in a 2D environment.
#          Implements the A* (A-star) search algorithm on an occupancy grid.

import logging
import heapq # Efficient priority queue for A* open set
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import numpy as np

# Configure basic logging
logger = logging.getLogger("PathPlanner")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

# Type hint for a location on the grid (row, col)
GridLocation = Tuple[int, int]
# Type hint for a path
PathType = List[GridLocation]

@dataclass(order=True)
class PriorityNode:
    """A node used in the A* priority queue for sorting."""
    priority: float # The f-score (g_score + h_score)
    # The item itself needs a __lt__ method if priorities are equal, or just use another sortable field
    item: Any=field(compare=False)

class PathPlanner:
    """
    Finds an optimal path on a 2D occupancy grid using the A* algorithm.
    """

    def __init__(self, occupancy_grid: np.ndarray):
        """
        Initializes the PathPlanner with a map of the environment.

        Args:
            occupancy_grid (np.ndarray): A 2D numpy array where:
                - 0 represents free, traversable space.
                - 1 (or any non-zero value) represents an occupied, non-traversable obstacle.
        """
        if not isinstance(occupancy_grid, np.ndarray) or occupancy_grid.ndim != 2:
            raise ValueError("occupancy_grid must be a 2D NumPy array.")
        
        self.grid = occupancy_grid
        self.height, self.width = occupancy_grid.shape
        logger.info(f"PathPlanner initialized with a {self.width}x{self.height} grid.")

    def _is_valid_location(self, loc: GridLocation) -> bool:
        """Checks if a location is within grid boundaries and is not an obstacle."""
        r, c = loc
        return (0 <= r < self.height and 0 <= c < self.width) and (self.grid[r, c] == 0)

    def _get_neighbors(self, current_loc: GridLocation) -> List[GridLocation]:
        """
        Gets the valid, traversable neighbors of a grid location.
        Considers 8-directional movement (including diagonals).
        """
        r, c = current_loc
        neighbors = []
        # Possible moves: [dr, dc, move_cost]
        # Straight moves cost 1.0, diagonal moves cost sqrt(2) ~= 1.414
        possible_moves = [
            (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1), # Straight
            (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1) # Diagonal
        ]
        
        for next_loc in possible_moves:
            if self._is_valid_location(next_loc):
                neighbors.append(next_loc)
        return neighbors

    def _heuristic(self, loc_a: GridLocation, loc_b: GridLocation) -> float:
        """
        Calculates the heuristic (estimated distance) between two points.
        Uses Euclidean distance for a more accurate estimate than Manhattan distance.
        """
        (r1, c1) = loc_a
        (r2, c2) = loc_b
        return np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
        # Alternative: Manhattan distance (faster but less accurate for diagonal movement)
        # return abs(r1 - r2) + abs(c1 - c2)

    def _reconstruct_path(self, came_from: Dict[GridLocation, GridLocation], current: GridLocation) -> PathType:
        """Traces the path backwards from the goal to the start."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1] # Reverse to get path from start to goal

    def plan_path_astar(self, start: GridLocation, goal: GridLocation) -> Optional[PathType]:
        """
        Finds a path from start to goal using the A* search algorithm.

        Args:
            start (GridLocation): The starting coordinates (row, col).
            goal (GridLocation): The goal coordinates (row, col).

        Returns:
            Optional[PathType]: A list of (row, col) tuples representing the path,
                                or None if no path is found.
        """
        logger.info(f"Planning path from {start} to {goal} using A* algorithm...")

        if not self._is_valid_location(start):
            logger.error(f"Start location {start} is invalid (out of bounds or on an obstacle).")
            return None
        if not self._is_valid_location(goal):
            logger.error(f"Goal location {goal} is invalid (out of bounds or on an obstacle).")
            return None
        if start == goal:
            return [start] # Path is just the start point

        # The set of nodes already evaluated
        closed_set: Set[GridLocation] = set()
        
        # The set of discovered nodes that are not yet evaluated.
        # Implemented as a priority queue (min-heap) for efficiency.
        # Items are (f_score, node)
        open_set: List[PriorityNode] = [PriorityNode(priority=self._heuristic(start, goal), item=start)]
        
        # came_from[n] is the node immediately preceding it on the cheapest path from start to n.
        came_from: Dict[GridLocation, GridLocation] = {}
        
        # g_score[n] is the cost of the cheapest path from start to n currently known.
        g_score: Dict[GridLocation, float] = defaultdict(lambda: float('inf'))
        g_score[start] = 0.0
        
        # f_score[n] represents our current best guess as to how cheap a path from start to goal
        # can be if it goes through n. f_score[n] = g_score[n] + h(n).
        f_score: Dict[GridLocation, float] = defaultdict(lambda: float('inf'))
        f_score[start] = self._heuristic(start, goal)

        # Map to keep track of items in the priority queue for efficient updates/removals
        open_set_map = {start: open_set[0]}

        while open_set:
            # Get the node in open_set having the lowest f_score value
            current_node = heapq.heappop(open_set).item
            
            if current_node not in open_set_map: # If already removed (e.g., duplicate with higher f_score)
                continue
            
            del open_set_map[current_node] # Remove from map as we process it

            if current_node == goal:
                logger.info(f"Path found! Length: {len(self._reconstruct_path(came_from, current))} steps.")
                return self._reconstruct_path(came_from, current)

            closed_set.add(current_node)

            for neighbor in self._get_neighbors(current_node):
                if neighbor in closed_set:
                    continue # Ignore neighbors already evaluated

                # The distance from start to a neighbor
                # d(current,neighbor) is the weight of the edge from current to neighbor
                # Here, straight moves cost 1, diagonal moves cost sqrt(2)
                move_cost = np.sqrt((current[0] - neighbor[0])**2 + (current[1] - neighbor[1])**2)
                tentative_g_score = g_score[current] + move_cost

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one. Record it!
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_map:
                        entry = PriorityNode(priority=f_score[neighbor], item=neighbor)
                        heapq.heappush(open_set, entry)
                        open_set_map[neighbor] = entry
                    # If neighbor is already in open_set with a higher f_score,
                    # a proper priority queue implementation would update its priority.
                    # Since we are using a basic heapq and checking on pop, we just add the new, better path.
                    # A more optimized version might use a structure that supports priority updates.

        logger.warning(f"No path found from {start} to {goal}.")
        return None # No path was found
