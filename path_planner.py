#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose
from bittle_msgs.msg import AprilTag, Yolo, BittlePath, BittlePathJSON
import numpy as np
import json
import heapq
import math
from std_msgs.msg import Int32

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        qos_profile = QoSProfile(depth=10)

        # Declare a parameter for the mission goal fallback (e.g., the next target goal)
        self.declare_parameter("mission_goal", [1.0, 1.0])

        # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        self.create_subscription(AprilTag, '/april_tag/bittle_pose', self.bittlebot_callback, qos_profile)
        self.create_subscription(Yolo, '/yolo/goals', self.goal_callback, qos_profile)
        # Subscribe to LLM stage changes for re-planning candidate paths.
        self.create_subscription(Int32, '/llm_stage', self.stage_callback, qos_profile)
        # New: Subscribe to buffer updates
        self.create_subscription(Int32, '/buffer_update', self.buffer_update_callback, qos_profile)

        # Publishers
        self.path_pub = self.create_publisher(BittlePath, '/bittlebot/path', qos_profile)
        self.json_pub = self.create_publisher(BittlePathJSON, '/bittlebot/path_json', qos_profile)
        self.path_vis_pub = self.create_publisher(Path, '/bittlebot/path_visualization', qos_profile)

        # Map dimensions and resolution.
        self.map_width = 320
        self.map_height = 240
        self.map_resolution = 0.0053  # meters per cell

        self.map_info = Pose()  # Using default Pose for origin

        # Internal state
        self.occupancy_grid = None           # 2D numpy array
        self.bittlebot_position = None       # in grid coordinates (x, y)
        self.candidate_goals = []            # Each candidate: { "world": [x, y], "grid": (gx, gy) }
        self.candidate_paths = []            # List of candidate paths computed via A*

        self.get_logger().info("PathPlanner node started for multiple candidate goals.")

        # Timer to periodically check if the current paths remain valid
        self.path_check_timer = self.create_timer(2.0, self.check_current_path_validity)

        # New: Variables to track buffer updates and waiting state
        self.current_buffer = None
        self.wait_for_buffer_update = False

    def stage_callback(self, msg: Int32):
        self.get_logger().info(f"Received LLM stage update: {msg.data}. Waiting for buffer update before re-planning candidate paths.")
        # Set flag to wait for the new buffer value
        self.wait_for_buffer_update = True

    def buffer_update_callback(self, msg: Int32):
        self.current_buffer = msg.data
        self.get_logger().info(f"Buffer updated to: {self.current_buffer}")
        # If we are waiting for the buffer update, re-plan candidate paths now
        if self.wait_for_buffer_update:
            self.plan_paths_for_candidates()
            self.wait_for_buffer_update = False

    def map_callback(self, msg: OccupancyGrid):
        self.occupancy_grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.get_logger().info("Occupancy grid updated.")

    def bittlebot_callback(self, msg: AprilTag):
        grid_x = int(msg.position[0] / self.map_resolution)
        grid_y = int(msg.position[1] / self.map_resolution)
        grid_x = max(0, min(self.map_width - 1, grid_x))
        grid_y = max(0, min(self.map_height - 1, grid_y))
        self.bittlebot_position = (grid_x, grid_y)
        self.get_logger().info(f"BittleBot position (grid): {self.bittlebot_position}")

    def goal_callback(self, msg: Yolo):
        """
        Update candidate goals from the YOLO message.
        Each goal is defined by 4 floats in msg.xywh: [cx, cy, w, h].
        If no valid detections are found, use a fallback mission goal from a ROS parameter.
        """
        self.candidate_goals = []  # Clear previous candidates
        num_goals = len(msg.xywh) // 4

        if num_goals == 0:
            self.get_logger().warn("No valid goal detections received from YOLO; checking fallback mission goal.")
            mission_goal = self.get_parameter("mission_goal").get_parameter_value().double_array_value
            if mission_goal and len(mission_goal) >= 2:
                goal_x, goal_y = mission_goal[0], mission_goal[1]
                goal_x_grid = int(goal_x / self.map_resolution)
                goal_y_grid = int(goal_y / self.map_resolution)
                candidate = {
                    "world": [goal_x, goal_y],
                    "grid": (goal_x_grid, goal_y_grid)
                }
                self.candidate_goals.append(candidate)
                self.get_logger().info(f"Using fallback mission goal: world=({goal_x:.2f}, {goal_y:.2f}), grid=({goal_x_grid}, {goal_y_grid})")
            else:
                self.get_logger().warn("No fallback mission goal available from parameter 'mission_goal'.")
        else:
            for i in range(num_goals):
                idx = i * 4
                world_x = msg.xywh[idx]
                world_y = msg.xywh[idx+1]
                goal_x_grid = int(world_x / self.map_resolution)
                goal_y_grid = int(world_y / self.map_resolution)
                candidate = {
                    "world": [world_x, world_y],
                    "grid": (goal_x_grid, goal_y_grid)
                }
                self.candidate_goals.append(candidate)
                self.get_logger().info(
                    f"Candidate goal {i}: world=({world_x:.2f}, {world_y:.2f}), grid=({goal_x_grid}, {goal_y_grid})"
                )
        # After updating candidate goals, compute candidate paths.
        self.plan_paths_for_candidates()

    def plan_paths_for_candidates(self):
        """
        Runs A* from the BittleBot position to each candidate goal,
        smooths each path, and publishes the result as JSON.
        """
        if (self.occupancy_grid is None or
            self.bittlebot_position is None or
            len(self.candidate_goals) == 0):
            self.get_logger().warn("Missing data for path planning (occupancy grid, robot position, or candidate goals).")
            return

        self.candidate_paths = []  # Clear previous candidate paths
        start = self.bittlebot_position

        # Tolerance: if the candidate goal is very close to the robot (in cells), ignore occupancy blockage.
        GOAL_OCCUPANCY_TOLERANCE_CELLS = 2

        for i, candidate in enumerate(self.candidate_goals):
            goal = candidate["grid"]

            # Check map bounds
            if not (0 <= goal[0] < self.map_width and 0 <= goal[1] < self.map_height):
                self.get_logger().error(f"Candidate goal {i} {goal} is out of bounds!")
                continue

            # Ensure the goal cell is free or near the robot (tolerance for when the goal is reached)
            if self.occupancy_grid[goal[1], goal[0]] != 0:
                if self.bittlebot_position is not None:
                    dist_cells = math.hypot(goal[0] - self.bittlebot_position[0], goal[1] - self.bittlebot_position[1])
                    if dist_cells > GOAL_OCCUPANCY_TOLERANCE_CELLS:
                        self.get_logger().error(f"Candidate goal {i} {goal} is blocked by an obstacle!")
                        continue
                    else:
                        self.get_logger().info(f"Candidate goal {i} {goal} is near the robot ({dist_cells:.2f} cells away); ignoring occupancy blockage.")
                else:
                    self.get_logger().error("Bittlebot position not available for occupancy check.")
                    continue

            # Plan raw path with A*
            raw_path = self.a_star(start, goal)
            if raw_path:
                # Smooth the path before converting to world coordinates
                smoothed_path = self.smooth_path(raw_path)
                metrics = self.compute_path_metrics(smoothed_path)
                candidate_path = {
                    "goal": candidate["world"],
                    "grid_goal": goal,
                    "path": self.convert_path_to_world(smoothed_path),
                    "metrics": metrics
                }
                self.candidate_paths.append(candidate_path)
                self.get_logger().info(
                    f"Candidate path {i} computed: length={metrics['path_length']:.2f}, "
                    f"obstacle_count={metrics['obstacle_count']}, "
                    f"min_clearance={metrics['min_clearance']:.3f}, "
                    f"avg_clearance={metrics['avg_clearance']:.3f}."
                )
            else:
                self.get_logger().warn(f"A* failed to find a valid path for candidate goal {i}.")

        # Publish all candidate paths as JSON
        self.publish_candidate_paths()

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor, step_cost in self.get_neighbors_8(current):
                tentative_g = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def get_neighbors_8(self, node):
        (x, y) = node
        neighbors_8 = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),
            (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)
        ]
        valid_neighbors = []
        for nx, ny in neighbors_8:
            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                if self.occupancy_grid[ny, nx] == 0:
                    cost = 1.0 if (nx == x or ny == y) else 1.4142
                    valid_neighbors.append(((nx, ny), cost))
        return valid_neighbors

    def heuristic(self, node, goal):
        (x1, y1) = node
        (x2, y2) = goal
        return math.hypot(x2 - x1, y2 - y1)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def compute_clearance_for_point(self, x, y):
        """
        Compute the Euclidean distance (in meters) from the given grid cell (x,y)
        to the nearest obstacle cell in the occupancy grid.
        """
        min_dist_cells = float('inf')
        for i in range(self.map_height):
            for j in range(self.map_width):
                if self.occupancy_grid[i, j] != 0:
                    dist = math.hypot(j - x, i - y)
                    if dist < min_dist_cells:
                        min_dist_cells = dist
        return min_dist_cells * self.map_resolution if min_dist_cells != float('inf') else float('inf')

    def compute_path_metrics(self, grid_path):
        path_length = 0.0
        obstacle_count = 0
        clearances = []
        for i in range(1, len(grid_path)):
            x1, y1 = grid_path[i - 1]
            x2, y2 = grid_path[i]
            segment_len_cells = math.hypot(x2 - x1, y2 - y1)
            path_length += segment_len_cells * self.map_resolution
            if not self.is_valid_cell(x2, y2):
                obstacle_count += 1
            clearance = self.compute_clearance_for_point(x2, y2)
            clearances.append(clearance)

        min_clearance = min(clearances) if clearances else float('inf')
        avg_clearance = sum(clearances)/len(clearances) if clearances else float('inf')
        return {
            "path_length": path_length,
            "obstacle_count": obstacle_count,
            "min_clearance": min_clearance,
            "avg_clearance": avg_clearance
        }

    def convert_path_to_world(self, grid_path):
        world_path = []
        for (x, y) in grid_path:
            world_x = x * self.map_resolution
            world_y = y * self.map_resolution
            world_path.append([world_x, world_y])
        return world_path

    def smooth_path(self, path):
        if len(path) < 3:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.check_line_of_sight(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def check_line_of_sight(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        for _ in range(n):
            if not self.is_valid_cell(x, y):
                return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    def is_valid_cell(self, x, y):
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            return (self.occupancy_grid[y, x] == 0)
        return False

    def publish_candidate_paths(self):
        data = {"candidate_paths": self.candidate_paths}
        json_msg = BittlePathJSON()
        json_msg.json_data = json.dumps(data)
        self.json_pub.publish(json_msg)
        self.get_logger().info("Published candidate paths.")

        if self.candidate_paths:
            chosen = min(self.candidate_paths, key=lambda cp: cp["metrics"]["path_length"])
            self.publish_visualization(chosen["path"])

    def publish_visualization(self, world_path):
        nav_path = Path()
        nav_path.header.stamp = self.get_clock().now().to_msg()
        nav_path.header.frame_id = "map"
        for coord in world_path:
            pose = PoseStamped()
            pose.pose.position.x = coord[0]
            pose.pose.position.y = coord[1]
            pose.pose.position.z = 0.0
            nav_path.poses.append(pose)
        self.path_vis_pub.publish(nav_path)
        self.get_logger().info("Published visualization of chosen candidate path.")

    def check_current_path_validity(self):
        if not self.candidate_paths or self.occupancy_grid is None:
            return

        for cp in self.candidate_paths:
            for point in cp["path"]:
                grid_x = int(point[0] / self.map_resolution)
                grid_y = int(point[1] / self.map_resolution)
                if not self.is_valid_cell(grid_x, grid_y):
                    self.get_logger().info("A candidate path is invalidated by a new obstacle. Replanning.")
                    self.plan_paths_for_candidates()
                    return

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("PathPlanner node shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
