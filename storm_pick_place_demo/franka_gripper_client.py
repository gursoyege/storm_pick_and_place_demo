from __future__ import annotations

from control_msgs.action import GripperCommand
from franka_msgs.action import Grasp
from rclpy.action import ActionClient
from rclpy.node import Node


class FrankaGripperClient:
    """Client for the Franka gripper actions used in existing AGIMUS demos."""

    def __init__(self, node: Node, arm_id: str = "fer") -> None:
        self._node = node
        prefix = f"/{arm_id}_gripper"
        self._gripper_cmd_client = ActionClient(self._node, GripperCommand, f"{prefix}/gripper_action")
        self._grasp_client = ActionClient(self._node, Grasp, f"{prefix}/grasp")

    def command(self, position: float, max_effort: float, timeout_sec: float = 2.0):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(position)
        goal_msg.command.max_effort = float(max_effort)

        self._node.get_logger().info("Waiting for gripper_action server...")
        if not self._gripper_cmd_client.wait_for_server(timeout_sec=timeout_sec):
            self._node.get_logger().error("Timeout waiting for gripper_action server.")
            return None
        return self._gripper_cmd_client.send_goal_async(goal_msg)

    def grasp(
        self,
        width: float = 0.0,
        speed: float = 0.04,
        force: float = 10.0,
        timeout_sec: float = 2.0,
    ):
        goal_msg = Grasp.Goal()
        goal_msg.width = float(width)
        goal_msg.speed = float(speed)
        goal_msg.force = float(force)
        goal_msg.epsilon.outer = 0.1

        self._node.get_logger().info("Waiting for grasp server...")
        if not self._grasp_client.wait_for_server(timeout_sec=timeout_sec):
            self._node.get_logger().warn("Timeout waiting for grasp server.")
            return None
        return self._grasp_client.send_goal_async(goal_msg)
