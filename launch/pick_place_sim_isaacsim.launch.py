from pathlib import Path

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import Command, EnvironmentVariable, FindExecutable, LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from agimus_demos_common.launch_utils import generate_default_franka_args, generate_include_launch
from agimus_demos_common.static_transform_publisher_node import static_transform_publisher_node


def launch_setup(context: LaunchContext, *args, **kwargs) -> list[LaunchDescriptionEntity]:
    arm_id = LaunchConfiguration("arm_id")
    armed = LaunchConfiguration("armed")
    with_rviz = LaunchConfiguration("with_rviz")
    armed_param = ParameterValue(armed, value_type=bool)

    # Force simulation mode for this launch file (agimus uses Gazebo).
    set_sim_cfg = [
        SetLaunchConfiguration("use_gazebo", "true"),
        SetLaunchConfiguration("use_rviz", with_rviz),
        SetLaunchConfiguration("robot_ip", ""),
    ]

    rviz_cfg = PathJoinSubstitution([FindPackageShare("storm_pick_place_demo"), "rviz", "pick_place.rviz"])
    lfc_params = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "linear_feedback_controller_params.yaml"]
    )

    franka_robot_launch = generate_include_launch(
        "franka_common_lfc.launch.py",
        extra_launch_arguments={"rviz_config_path": rviz_cfg, "linear_feedback_controller_params": lfc_params},
    )

    env_xacro = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "urdf", "environment.urdf.xacro"]
    )
    environment_description = ParameterValue(
        Command([PathJoinSubstitution([FindExecutable(name="xacro")]), " ", env_xacro]),
        value_type=str,
    )
    environment_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="environment_publisher",
        output="screen",
        remappings=[("robot_description", "environment_description")],
        parameters=[{"robot_description": environment_description, "use_sim_time": True}],
    )

    arm_id_str = context.perform_substitution(arm_id)
    base_frame = f"{arm_id_str}_link0"
    tf_env = static_transform_publisher_node(
        frame_id=base_frame,
        child_frame_id="robot_attachment_link",
        xyz=(0.0, 0.0, 0.0),
        rot_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    wait_for_non_zero_joints_node = Node(
        package="agimus_demos_common",
        executable="wait_for_non_zero_joints_node",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    obj_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "object_manager_params.yaml"]
    )
    object_manager_node = Node(
        package="storm_pick_place_demo",
        executable="object_manager_node",
        name="object_manager",
        parameters=[{"use_sim_time": True}, obj_yaml],
        output="screen",
    )

    storm_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_agimus_bridge"), "config", "storm_mppi_planner_params.yaml"]
    )
    storm_planner_node = Node(
        package="storm_agimus_bridge",
        executable="storm_mppi_planner_node",
        name="storm_mppi_planner",
        parameters=[{"use_sim_time": True}, storm_yaml, {"armed": armed_param}],
        output="screen",
        remappings=[("robot_description", "robot_description_with_collision")],
    )

    task_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "pick_place_task_manager_params.yaml"]
    )
    task_node = Node(
        package="storm_pick_place_demo",
        executable="pick_place_task_manager_node",
        name="pick_place_task_manager",
        parameters=[{"use_sim_time": True}, task_yaml, {"armed": armed_param, "auto_start": armed_param}],
        output="screen",
    )

    start_stack_on_ready = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=wait_for_non_zero_joints_node,
            on_exit=[
                object_manager_node,
                storm_planner_node,
                task_node,
            ],
        )
    )

    # Ensure STORM is importable from ROS nodes and from STORM's spawned MPPI optimization process.
    storm_root = str(Path("/workspace/storm"))
    set_storm_env = [
        SetEnvironmentVariable("STORM_ROOT", storm_root),
        SetEnvironmentVariable(
            "PYTHONPATH",
            [TextSubstitution(text=storm_root), TextSubstitution(text=":"), EnvironmentVariable("PYTHONPATH")],
        ),
    ]

    return [
        *set_storm_env,
        *set_sim_cfg,
        franka_robot_launch,
        environment_publisher_node,
        tf_env,
        wait_for_non_zero_joints_node,
        start_stack_on_ready,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "armed",
            default_value="true",
            description="Arm the demo (publish commands and operate gripper).",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "with_rviz",
            default_value="true",
            description="Launch RViz2 (disable for headless smoke tests).",
            choices=["true", "false"],
        ),
    ]
    return LaunchDescription(declared_arguments + generate_default_franka_args() + [OpaqueFunction(function=launch_setup)])
