# Devin/modules/robotics/environment_mapping.py
# Purpose: Provides functionality to build a 2D occupancy grid map of an
#          environment using sensor data (e.g., LiDAR), a key component of SLAM.

import logging
import time
from typing import List, Tuple, Any
import numpy as np

# --- Conceptual Placeholders for Imported Modules & Data ---
Pose = Tuple[float, float, float] # (x_meters, y_meters, theta_radians)
LidarScan = List[Tuple[float, float]] # [(angle_radians, distance_meters), ...]
# --- End of Conceptual Placeholders ---

# Configure basic logging
logger = logging.getLogger("EnvironmentMapping")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

class EnvironmentMapper:
    """
    Builds and maintains a 2D occupancy grid map of the environment.
    """
    def __init__(self, map_size_pixels: Tuple[int, int] = (500, 500), map_resolution: float = 0.05):
        """
        Initializes the environment mapper.

        Args:
            map_size_pixels (Tuple[int, int]): The size of the map in pixels (width, height).
            map_resolution (float): The size of each pixel in meters (e.g., 0.05 m/pixel).
        """
        self.map_size = map_size_pixels
        self.resolution = map_resolution
        
        # The log-odds representation of the map.
        # Initialize with 0, representing 50% probability (unknown).
        self.log_odds_map = np.zeros(self.map_size, dtype=np.float32)
        
        # Log-odds values for updating the map.
        # These values are added/subtracted from the grid cells.
        self.log_odds_occupied = np.log(0.85 / (1 - 0.85)) # P(occ) = 85%
        self.log_odds_free = np.log(0.40 / (1 - 0.40))     # P(occ) = 40% -> P(free) = 60%
        
        # Saturation limits to prevent infinite values
        self.log_odds_min = -10.0
        self.log_odds_max = 10.0
        
        logger.info(f"EnvironmentMapper initialized with a {self.map_size[0]}x{self.map_size[1]} map at {self.resolution} m/pixel.")

    def _world_to_map_coords(self, world_coords: Tuple[float, float]) -> Tuple[int, int]:
        """Converts world coordinates (meters) to map grid coordinates (pixels)."""
        wx, wy = world_coords
        # Assume (0,0) in world coords is the center of the map
        mc = int(wx / self.resolution + self.map_size[0] / 2)
        mr = int(-wy / self.resolution + self.map_size[1] / 2) # Y is often inverted in image coordinates
        return (mc, mr)

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        A Python implementation of the Bresenham line algorithm.
        Returns all the grid cells that form a line between two points.
        """
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points

    def update_map(self, robot_pose: Pose, lidar_scan: LidarScan):
        """
        Updates the occupancy grid map with a new LiDAR scan.

        Args:
            robot_pose (Pose): The current pose of the robot (x, y, theta).
            lidar_scan (LidarScan): The latest scan data from the LiDAR.
        """
        robot_x, robot_y, robot_theta = robot_pose
        robot_map_coords = self._world_to_map_coords((robot_x, robot_y))

        # We will update a set of cells to avoid redundant updates
        cells_to_update_free = set()
        cells_to_update_occupied = set()

        for angle_rad, distance_m in lidar_scan:
            # Calculate the world coordinates of the laser hit point
            hit_angle = robot_theta + angle_rad
            hit_x = robot_x + distance_m * np.cos(hit_angle)
            hit_y = robot_y + distance_m * np.sin(hit_angle)
            hit_map_coords = self._world_to_map_coords((hit_x, hit_y))
            
            # Trace a line from the robot to the hit point
            line_points = self._bresenham_line(
                robot_map_coords[0], robot_map_coords[1],
                hit_map_coords[0], hit_map_coords[1]
            )
            
            # The points along the line (except the last one) are free space
            for p in line_points[:-1]:
                cells_to_update_free.add(p)
            
            # The last point is occupied
            cells_to_update_occupied.add(line_points[-1])
            
        # Perform the log-odds updates
        for x, y in cells_to_update_free:
            if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                self.log_odds_map[y, x] -= self.log_odds_free
        for x, y in cells_to_update_occupied:
            if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                 self.log_odds_map[y, x] += self.log_odds_occupied

        # Clamp the values to prevent them from growing infinitely
        np.clip(self.log_odds_map, self.log_odds_min, self.log_odds_max, out=self.log_odds_map)
        logger.debug(f"Map updated with {len(lidar_scan)} laser points.")

    def get_occupancy_grid(self, occupied_thresh: float = 0.65, free_thresh: float = 0.196) -> np.ndarray:
        """
        Converts the internal log-odds map to a standard occupancy grid
        for use by a path planner.

        Returns:
            np.ndarray: A grid where 1=occupied, 0=free, -1=unknown.
        """
        # Convert log-odds back to probabilities
        prob_map = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds_map))
        
        grid = np.full(self.map_size, -1, dtype=np.int8) # -1 is unknown
        grid[prob_map > occupied_thresh] = 1 # 1 is occupied
        grid[prob_map < free_thresh] = 0 # 0 is free
        
        return grid

# --- Example Usage ---
if __name__ == "__main__":
    print("=========================================================")
    print("=== Environment Mapper Prototype 🗺️ ===")
    print("=========================================================")

    # 1. Initialize the mapper
    mapper = EnvironmentMapper(map_size_pixels=(200, 200), map_resolution=0.1)

    # 2. Simulate the robot moving and scanning
    print("\nSimulating a robot moving down a hallway and mapping it...")
    # The hallway is 10 meters long. Walls are at y = +1 and y = -1.
    robot_path_simulation = [(x, 0) for x in np.arange(0, 5, 0.5)] # Move 5 meters in 0.5m steps

    for i, (robot_x, robot_y) in enumerate(robot_path_simulation):
        robot_pose = (robot_x, robot_y, 0.0) # Moving straight, no rotation
        
        # Simulate a 180-degree LiDAR scan
        sim_scan = []
        for angle_deg in range(-90, 91, 5):
            angle_rad = np.deg2rad(angle_deg)
            # Wall at y=1 (to the left) or y=-1 (to the right)
            if -90 <= angle_deg <= 90:
                dist = 1.0 / np.cos(angle_rad) if np.cos(angle_rad) != 0 else 100.0
                sim_scan.append((angle_rad, abs(dist)))

        # Update the map with this new scan
        mapper.update_map(robot_pose, sim_scan)
        print(f"  Simulation step {i+1}/{len(robot_path_simulation)}: Mapped from pose {robot_pose[:2]}")
        time.sleep(0.1)

    # 3. Get the final map and visualize it
    final_map = mapper.get_occupancy_grid()
    
    # Simple text-based visualization
    print("\n--- Final Generated Map ---")
    # Downsample for display
    display_map = final_map[::5, ::5]
    for row in display_map:
        row_str = ""
        for cell in row:
            if cell == 1: row_str += "█" # Occupied
            elif cell == 0: row_str += " " # Free
            else: row_str += "." # Unknown
        print(row_str)
    print("-------------------------")

    # A real application would save this map to a file or pass it to the PathPlanner
    # e.g., np.save("generated_map.npy", final_map)
    print("\nMap generation complete. The map can now be used for path planning.")


    print("\n=========================================================")
    print("=== Environment Mapper Prototype Complete ===")
    print("=========================================================")
