# `storm_pick_place_demo`

Pick-and-place demo for Franka Panda using:

- High-level planner: STORM MPPI via `storm_agimus_bridge`
- Low-level controller: `linear_feedback_controller` (ros2_control torque LFC)
- Visualization: RViz2 + TF + markers

## Prerequisites

- ROS 2 Humble
- System Python can import Isaac Sim + Isaac Lab:
  - `python -c "import isaacsim, isaaclab; print('ok')"`
- STORM repo available at `/workspace/storm` (launch files set `STORM_ROOT` + `PYTHONPATH` automatically)

## Build

```bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select storm_agimus_bridge storm_pick_place_demo
source install/setup.bash
```

## Run (Simulation)

```bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
source install/setup.bash

# If controller activation fails (common when the ROS 2 daemon is misbehaving in this environment):
ros2 daemon stop

ros2 launch storm_pick_place_demo pick_place_sim_isaacsim.launch.py \
  armed:=true with_rviz:=true gz_headless:=true
```

Notes:

- `storm_agimus_bridge` publishes `control` (`linear_feedback_controller_msgs/Control`) which drives the torque LFC.
- Tune MPPI + LFC parameters in `storm_agimus_bridge/config/storm_mppi_planner_params.yaml`.

## Run (Real Robot)

Safety gate is disabled by default. The robot only moves when `enable_motion:=true`.

```bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
source install/setup.bash

ros2 launch storm_pick_place_demo pick_place_real.launch.py \
  enable_motion:=true with_rviz:=true
```

## Key Topics

- Input state: `sensor` (`linear_feedback_controller_msgs/Sensor`)
- Planner output: `control` (`linear_feedback_controller_msgs/Control`)
- Task goals: `/goal_pose` (`geometry_msgs/PoseStamped`) (published by the task manager)
- Objects: `/storm_pick_place/objects_markers` (`visualization_msgs/MarkerArray`)

## Troubleshooting

- Controller never activates / no `sensor`:
  - Run `ros2 daemon stop` and relaunch.
- STORM fails to import under `ros2 launch`:
  - Ensure `/workspace/storm` exists and `storm_kit/` is present.
  - Launch files set `STORM_ROOT=/workspace/storm` and prepend it to `PYTHONPATH`.
