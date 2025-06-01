# Sequential-Semantic-Tasks-with-Petoi-Bittle-Bot-using-GPT-4o

This repository builds on **"Semantic Intelligence: GPT-4 + A* Path Planning for Low-Cost Robotics"** by extending the system to sequential autonomous tasks. The robot now executes multi-stage missions—such as collecting a resource before navigating to a final destination—while GPT-4 dynamically adjusts obstacle buffers based on semantic context.

## Table of Contents
- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [Key Differences from Previous Work](#key-differences-from-previous-work)
- [Sequential Task Architecture](#sequential-task-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Installation & Setup](#installation--setup)
- [Running Sequential Experiments](#running-sequential-experiments)
- [Multi-Stage Mission Types](#multi-stage-mission-types)
- [Dynamic Buffer Management](#dynamic-buffer-management)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
The enhanced system demonstrates autonomous execution of sequential semantic tasks where GPT-4 reasons about mission stages, goal priorities, and safety requirements. The robot transitions between multiple objectives while adapting its navigation strategy to environmental conditions and task demands.

## Key Capabilities
- **Two-Stage Mission Execution** — Resource collection followed by final-goal navigation  
- **Dynamic Stage Transitions** — Automatic progression between mission phases  
- **Context-Aware Buffer Adjustment** — GPT-4 modifies obstacle clearance on the fly  
- **Goal Proximity Detection** — Prevents redundant goal detections  
- **Persistent Mission Memory** — Maintains task state across stages  

## Key Differences from Previous Work

### Enhanced LLM Reasoning Node
- Stage-based decision making (resource vs. final navigation)  
- GPT-4 determines buffer values (0 or 20 cells) per path complexity  
- Mission state tracking and delayed-decision timers  

### Updated Path Planner
- Subscribes to `/buffer_update` for dynamic inflation  
- Stage-aware replanning and goal-tolerance logic  

### Enhanced Occupancy Grid Publisher
- Real-time buffer inflation and robot-proximity filtering  

### Improved YOLO Node
- Filters goal detections when robot is already on target  

## Sequential Task Architecture

### Stage 1 — Resource Collection
- **Objective:** Reach resource goal (identified via "skinny obstacle")  
- **Buffer Logic:** 0 cells for direct paths; 20 cells for complex routes  

### Stage 2 — Final Destination
- **Objective:** Safely reach final goal location  
- **Buffer Logic:** Conservative (often 20 cells) in congested areas  

**Stage Transition:** Triggered when robot is within 0.15 m of current goal; system resets decision locks and publishes stage updates.

## Hardware Requirements
(Same as previous project)
- Petoi Bittle + Raspberry Pi Zero 2 W  
- Overhead camera with streaming server  
- AprilTag (ID = 1)  
- Laptop/desktop for ROS 2 computation  

## Software Dependencies

### Laptop Requirements
```bash
# ROS 2 Humble
sudo apt update && sudo apt install ros-humble-desktop
# Python packages
pip install openai ultralytics opencv-python apriltag cv-bridge
# Additional ROS 2 packages
sudo apt install ros-humble-nav-msgs ros-humble-sensor-msgs

### Additional Message Types
- `std_msgs/Int32` — buffer updates and stage notifications  
- Updated `BittlePathJSON` / `BittlePathGPTJSON` — richer fields for path coordination  

## Installation & Setup

### 1. Clone Repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/yourusername/sequential-semantic-navigation.git
cd ~/ros2_ws
colcon build
source install/setup.bash
