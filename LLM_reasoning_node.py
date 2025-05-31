#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from nav_msgs.msg import OccupancyGrid
import json
import math
import os
import openai
import time

from bittle_msgs.msg import AprilTag, Yolo, BittlePathJSON, BittlePathGPTJSON

LOG_FILE = "/home/jesse/Desktop/openai_responses.json"

def append_to_log(data: dict):
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []
    logs.append(data)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

class LLMReasoning(Node):
    def __init__(self):
        super().__init__('llm_reasoning_node')
        
        # Mission stages: 1 = collect resource, 2 = navigate to final goal.
        self.current_stage = 1
        self.decision_locked = False
        self.candidate_paths = []
        self.selected_candidate_path = []
        self.robot_position = None   # [x, y, theta]
        self.obstacles = []
        self.map_resolution = 0.0053

        # Pure pursuit thresholds
        self.WP_THRESHOLD = 0.15
        self.LOOKAHEAD_DIST = 0.1
        self.ANGLE_THRESHOLD = 0.25
        self.path_index = 0

        # For stage transitions
        self.stage2_start_position = None

        # Persistent mission list with two goals (example coordinates)
        self.mission_goals = [
            {"name": "resource_goal", "position": [0.5, 0.5]},  # Task one
            {"name": "final_goal", "position": [1.0, 1.0]}        # Task two
        ]
        self.current_mission_index = 0

        # Variables to store LLM decision before locking in candidate path
        self.chosen_candidate_index = None
        self.pending_buffer_value = None
        self.decision_lock_timer = None

        # -----------------------
        #   SUBSCRIPTIONS
        # -----------------------
        self.create_subscription(BittlePathJSON, '/bittlebot/path_json', self.path_json_callback, 10)
        self.create_subscription(AprilTag, '/april_tag/bittle_pose', self.apriltag_callback, 10)
        self.create_subscription(Yolo, '/yolo/obstacles', self.obstacles_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # -----------------------
        #   PUBLISHERS
        # -----------------------
        self.cmd_pub = self.create_publisher(String, '/bittle_cmd', 10)
        self.gpt_path_pub = self.create_publisher(BittlePathGPTJSON, '/bittlebot/path_gpt_json', 10)
        self.buffer_pub = self.create_publisher(Int32, '/buffer_update', 10)
        self.stage_pub = self.create_publisher(Int32, '/llm_stage', 10)

        # -----------------------
        #   TIMERS
        # -----------------------
        self.goal_check_timer = self.create_timer(0.5, self.goal_check_callback)
        self.control_timer = self.create_timer(0.4, self.control_timer_callback)

        # One-shot timer for delaying the candidate lock-in (initialized as None)
        self.decision_timer = None

        # OpenAI API key setup (consider securing this key in production)
        openai.api_key = ""

        self.get_logger().info("LLM Reasoning node started. Stage 1: Collect Resource.")

    # ------------------------------------------------------------------
    #   SUBSCRIPTION CALLBACKS
    # ------------------------------------------------------------------

    def apriltag_callback(self, msg: AprilTag):
        self.robot_position = [msg.position[0], msg.position[1], msg.position[2]]
        self.get_logger().debug(f"Updated robot position: {self.robot_position}")

    def obstacles_callback(self, msg: Yolo):
        obs_list = []
        arr = msg.xywh
        for i in range(0, len(arr), 4):
            obs_list.append({
                "center": [round(arr[i], 3), round(arr[i+1], 3)],
                "size": [round(arr[i+2], 3), round(arr[i+3], 3)]
            })
        self.obstacles = obs_list

    def map_callback(self, msg: OccupancyGrid):
        self.map_resolution = msg.info.resolution

    def path_json_callback(self, msg: BittlePathJSON):
        """
        When candidate paths are published by the PathPlanner,
        start a one-shot timer (if not already running) to delay the LLM decision.
        """
        try:
            data = json.loads(msg.json_data)
            if "candidate_paths" in data:
                self.candidate_paths = data["candidate_paths"]
                self.get_logger().info(f"Received {len(self.candidate_paths)} candidate paths.")
                if not self.decision_locked and self.candidate_paths:
                    if self.decision_timer is None:
                        self.decision_timer = self.create_timer(1.0, self.delayed_trigger_llm_decision)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error decoding path JSON: {e}")

    def delayed_trigger_llm_decision(self):
        if self.decision_timer is not None:
            self.decision_timer.cancel()
            self.decision_timer = None
        self.trigger_llm_decision()

    # ------------------------------------------------------------------
    #   LLM DECISION LOGIC
    # ------------------------------------------------------------------

    def trigger_llm_decision(self):
        """
        Prompt the GPT model for a candidate selection.
        Instead of prescribing a fixed buffer, the LLM is asked to determine
        the appropriate occupancy grid buffer based on the candidate path's metrics.
        """
        if self.current_stage == 1:
            system_prompt = (
                "You are an LLM controlling a BittleBot in a 2D grid. The mission is to collect a resource. "
                "There are two goals in your environment. The goal a little further away with a single skinny obstacle is the resource goal and is your target. "
                "For this mission, you should only go to the goal with a single skinny obstacle near it and avoid the other goal. "
                "Based on the candidate path metrics (including path length, obstacle count, clearances, and qualitative description "
                "of obstacles), select the candidate that travels to the resource marker and then decide on the appropriate occupancy grid buffer: "
                "if the path is direct with minimal obstacles, set the buffer to 0; if it requires turns or obstacle avoidance, set it to 20. "
                "Return your decision strictly in JSON format: "
                '{"mode":"candidate_selection", "selected_candidate":<index>, "buffer":<value>}. Do not include any extra fields.'
            )
        elif self.current_stage == 2:
            system_prompt = (
                "You are an LLM controlling a BittleBot in a 2D grid. The mission is now to navigate to the final goal while maintaining safe clearance from obstacles. "
                "Among the candidate paths provided, select the one that ensures the safest navigation. "
                "Based on the candidate path metrics (including path length, obstacle count, and clearances), determine whether the path is sufficiently direct "
                "(set buffer 0) or if it requires turns or obstacle avoidance (set buffer 20). "
                "Return your decision strictly in JSON format: "
                '{"mode":"candidate_selection", "selected_candidate":<index>, "buffer":<value>}. Do not include any extra fields.'
            )
        else:
            self.get_logger().error("Unknown stage for LLM decision.")
            return

        # Build context data with candidate path metrics.
        context_data = {"candidate_paths": []}
        for i, cp in enumerate(self.candidate_paths):
            m = cp.get("metrics", {})
            candidate_info = {
                "index": i,
                "path_length": m.get("path_length", 0),
                "obstacle_count": m.get("obstacle_count", 0),
                "min_clearance": m.get("min_clearance", 0),
                "avg_clearance": m.get("avg_clearance", 0)
            }
            if "aspect_ratio" in cp:
                candidate_info["aspect_ratio"] = cp["aspect_ratio"]
            context_data["candidate_paths"].append(candidate_info)

        self.get_logger().info(f"LLM context data: {json.dumps(context_data)}")

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(context_data)}
                ],
                max_tokens=256,
                temperature=0.0
            )
            raw_reply = response.choices[0].message.content
            self.get_logger().info(f"LLM Response: {raw_reply}")
            append_to_log({
                "stage": self.current_stage,
                "prompt": {"system": system_prompt, "user": context_data},
                "response": raw_reply
            })
            time.sleep(1.0)  # Delay to avoid API rate limits
        except Exception as e:
            self.get_logger().error(f"Error during LLM API call: {e}")
            return

        mode, selection, buffer_value = self._parse_llm_reply(raw_reply)
        if mode != "candidate_selection" or selection < 0 or selection >= len(self.candidate_paths):
            self.get_logger().error("LLM returned an invalid candidate selection. Stopping.")
            self._publish_command("stop")
            return

        # Save the candidate selection and buffer value, then publish the buffer update
        self.chosen_candidate_index = selection
        self.pending_buffer_value = buffer_value
        self.publish_buffer_update(buffer_value)
        
        # Now, schedule a one-shot timer to lock in the candidate path after a delay
        self.decision_lock_timer = self.create_timer(1.0, self.lock_in_candidate_path)

    def lock_in_candidate_path(self):
        if self.decision_lock_timer is not None:
            self.decision_lock_timer.cancel()
            self.decision_lock_timer = None
        if self.chosen_candidate_index is not None and self.chosen_candidate_index < len(self.candidate_paths):
            chosen_candidate = self.candidate_paths[self.chosen_candidate_index]
            self.selected_candidate_path = chosen_candidate.get("path", [])
            self.decision_locked = True
            self.path_index = 0
            out_msg = BittlePathGPTJSON()
            out_msg.json_data = json.dumps({"candidate_paths": [chosen_candidate]})
            self.gpt_path_pub.publish(out_msg)
            self.get_logger().info(
                f"Stage {self.current_stage} locked choice: candidate {self.chosen_candidate_index}, buffer {self.pending_buffer_value}"
            )
        else:
            self.get_logger().error("Candidate index out of range after buffer update.")

    def _parse_llm_reply(self, raw):
        try:
            data = json.loads(raw.strip())
            mode = data.get("mode", "")
            selected_candidate = int(data.get("selected_candidate", -1))
            buffer_value = int(data.get("buffer", -1))
            return mode, selected_candidate, buffer_value
        except Exception as e:
            self.get_logger().error(f"Could not parse LLM reply as JSON: {e}")
            return "", -1, -1

    def publish_buffer_update(self, buffer_value):
        msg = Int32()
        msg.data = buffer_value
        self.buffer_pub.publish(msg)
        self.get_logger().info(f"Buffer updated to {buffer_value}")

    # ------------------------------------------------------------------
    #   GOAL CHECK & STAGE TRANSITION
    # ------------------------------------------------------------------

    def goal_check_callback(self):
        if self.decision_locked and self.selected_candidate_path and self.robot_position:
            path_goal = self.selected_candidate_path[-1]
            rx, ry, _ = self.robot_position
            distance = math.hypot(path_goal[0] - rx, path_goal[1] - ry)
            current_mission_goal = self.mission_goals[self.current_mission_index]["position"]
            mission_distance = math.hypot(current_mission_goal[0] - rx, current_mission_goal[1] - ry)
            if self.current_stage == 2 and self.stage2_start_position is not None:
                start_rx, start_ry, _ = self.stage2_start_position
                movement = math.hypot(rx - start_rx, ry - start_ry)
                if movement < 0.1:
                    return
            if distance < self.WP_THRESHOLD or mission_distance < self.WP_THRESHOLD:
                self.get_logger().info(f"Mission goal reached (distance: {mission_distance:.2f}).")
                if self.current_stage == 1:
                    self.current_mission_index += 1
                    self.transition_to_stage2()
                elif self.current_stage == 2:
                    self.get_logger().info("Final goal reached. Mission complete.")
                    self._publish_command("stop")

    def transition_to_stage2(self):
        self.get_logger().info("Transitioning to Stage 2: Resetting decision and candidate paths.")
        self.decision_locked = False
        self.selected_candidate_path = []
        self.candidate_paths = []
        self.stage2_start_position = self.robot_position
        self.current_stage = 2
        stage_msg = Int32()
        stage_msg.data = self.current_stage
        self.stage_pub.publish(stage_msg)
        self.get_logger().info(f"Published stage update: {self.current_stage}")

    # ------------------------------------------------------------------
    #   CONTROL LOOP (PURE PURSUIT)
    # ------------------------------------------------------------------

    def control_timer_callback(self):
        if self.decision_locked and self.selected_candidate_path:
            cmd = self._compute_pure_pursuit(self.selected_candidate_path)
            self._publish_command(cmd)

    def _compute_pure_pursuit(self, path):
        if not path or not self.robot_position:
            return "stop"
        rx, ry, rtheta = self.robot_position
        final_wp = path[-1]
        if math.hypot(final_wp[0] - rx, final_wp[1] - ry) < self.WP_THRESHOLD:
            return "stop"
        if self.path_index >= len(path):
            return "stop"
        cx, cy = path[self.path_index]
        dist_to_wp = math.hypot(cx - rx, cy - ry)
        if dist_to_wp < self.WP_THRESHOLD:
            self.path_index += 1
            if self.path_index >= len(path):
                return "stop"
        target_idx = self.path_index
        best_idx = target_idx
        while target_idx < len(path):
            tx, ty = path[target_idx]
            if math.hypot(tx - rx, ty - ry) >= self.LOOKAHEAD_DIST:
                best_idx = target_idx
                break
            best_idx = target_idx
            target_idx += 1
        tx, ty = path[best_idx]
        dx = tx - rx
        dy = ty - ry
        desired_theta = math.atan2(dy, dx)
        angle_error = self._normalize_angle(desired_theta - rtheta)
        if abs(angle_error) > self.ANGLE_THRESHOLD:
            return "turn_right" if angle_error > 0 else "turn_left"
        else:
            return "move_forward"

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _publish_command(self, cmd: str):
        if cmd not in {"move_forward", "turn_left", "turn_right", "stop"}:
            cmd = "stop"
        msg = String()
        msg.data = json.dumps({"command": cmd})
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"Published command: {cmd}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMReasoning()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("LLMReasoning node shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
