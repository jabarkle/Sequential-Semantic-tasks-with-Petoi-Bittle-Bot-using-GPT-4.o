#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from bittle_msgs.msg import Yolo, AprilTag
from std_msgs.msg import Int32
import math

class OccupancyGridPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_grid_publisher')
        qos_profile = QoSProfile(depth=10)

        # Subscribers for YOLO obstacles and AprilTag for robot pose
        self.sub_yolo = self.create_subscription(
            Yolo,
            '/yolo/obstacles',
            self.yolo_callback,
            qos_profile
        )
        self.sub_apriltag = self.create_subscription(
            AprilTag,
            '/april_tag/bittle_pose',
            self.apriltag_callback,
            qos_profile
        )
        # Subscriber for dynamic buffer updates from the LLM node.
        self.buffer_sub = self.create_subscription(
            Int32,
            '/buffer_update',
            self.buffer_update_callback,
            qos_profile
        )

        # Publishers for the occupancy grid and visualization.
        self.pub_grid = self.create_publisher(OccupancyGrid, '/map', qos_profile)
        self.pub_visualization = self.create_publisher(OccupancyGrid, '/visualization', qos_profile)

        # Map dimensions and resolution.
        self.map_width = 320
        self.map_height = 240
        self.map_resolution = 0.0053  # meters per cell

        self.map_info = MapMetaData()
        self.map_info.resolution = self.map_resolution
        self.map_info.width = self.map_width
        self.map_info.height = self.map_height
        self.map_info.origin = Pose()
        self.map_info.origin.position.x = 0.0
        self.map_info.origin.position.y = 0.0
        self.map_info.origin.position.z = 0.0
        self.map_info.origin.orientation.w = 1.0

        # Robot position from AprilTag (world coordinates)
        self.bittlebot_position = None
        self.buffer_cells = 0  # Default buffer; can be updated via the LLM.

        # Threshold: ignore obstacles that are too close to the robot.
        self.robot_obstacle_ignore_dist = 0.15

        # Store last received YOLO message for republishing purposes.
        self.last_yolo_msg = None

        # Timer to periodically republish the occupancy grid using the latest YOLO data.
        self.publish_timer = self.create_timer(1.0, self.publish_grid)

        self.get_logger().info("OccupancyGridPublisher node started with dynamic buffer control.")

    def apriltag_callback(self, apriltag_msg: AprilTag):
        self.bittlebot_position = [apriltag_msg.position[0], apriltag_msg.position[1]]

    def buffer_update_callback(self, msg: Int32):
        self.buffer_cells = msg.data
        self.get_logger().info(f"Buffer updated to: {self.buffer_cells}")
        # Republish the grid immediately using the current YOLO obstacles (if available).
        if self.last_yolo_msg is not None:
            self.process_yolo(self.last_yolo_msg)

    def yolo_callback(self, yolo_msg: Yolo):
        self.last_yolo_msg = yolo_msg
        self.process_yolo(yolo_msg)

    def process_yolo(self, yolo_msg: Yolo):
        # 1) Create an all-free occupancy grid.
        occupancy_data = [0 for _ in range(self.map_width * self.map_height)]

        # 2) Convert each YOLO bounding box to an occupied region with the specified buffer.
        arr = yolo_msg.xywh
        for i in range(0, len(arr), 4):
            center_x_m = arr[i]
            center_y_m = arr[i + 1]
            width_m    = arr[i + 2]
            height_m   = arr[i + 3]

            # If robot position is known, skip marking obstacles that are too close to it.
            if self.bittlebot_position is not None:
                robot_x, robot_y = self.bittlebot_position
                dist = math.hypot(center_x_m - robot_x, center_y_m - robot_y)
                if dist < self.robot_obstacle_ignore_dist:
                    self.get_logger().info(
                        f"Ignoring obstacle at ({center_x_m:.2f}, {center_y_m:.2f}) as it is {dist:.2f} m from the robot."
                    )
                    continue

            center_x_cell = int(center_x_m / self.map_resolution)
            center_y_cell = int(center_y_m / self.map_resolution)
            half_w_cells  = int((width_m / self.map_resolution) / 2.0)
            half_h_cells  = int((height_m / self.map_resolution) / 2.0)

            x_min = center_x_cell - half_w_cells - self.buffer_cells
            x_max = center_x_cell + half_w_cells + self.buffer_cells
            y_min = center_y_cell - half_h_cells - self.buffer_cells
            y_max = center_y_cell + half_h_cells + self.buffer_cells

            # Clamp coordinates to map bounds.
            x_min = max(0, x_min)
            x_max = min(self.map_width - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(self.map_height - 1, y_max)

            # Mark these cells as occupied (value 100).
            for yy in range(y_min, y_max + 1):
                for xx in range(x_min, x_max + 1):
                    idx = yy * self.map_width + xx
                    occupancy_data[idx] = 100

        # 3) Construct and publish the OccupancyGrid message.
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"
        grid_msg.info = self.map_info
        grid_msg.data = occupancy_data

        self.pub_grid.publish(grid_msg)
        self.pub_visualization.publish(grid_msg)
        self.get_logger().info("Published occupancy grid with current buffer.")

    def publish_grid(self):
        # Periodically republish occupancy grid using the latest YOLO obstacles data.
        if self.last_yolo_msg is not None:
            self.process_yolo(self.last_yolo_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
