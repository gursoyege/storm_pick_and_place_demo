from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException
from tf2_ros.transform_listener import TransformListener

from .franka_gripper_client import FrankaGripperClient


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = float(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - s * (yy + zz), s * (xy - wz), s * (xz + wy)],
            [s * (xy + wz), 1.0 - s * (xx + zz), s * (yz - wx)],
            [s * (xz - wy), s * (yz + wx), 1.0 - s * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_msg_to_se3(tf) -> pin.SE3:
    R = _quat_to_matrix(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
    t = np.array([tf.translation.x, tf.translation.y, tf.translation.z], dtype=np.float64)
    return pin.SE3(R, t)


def se3_to_pose_msg(M: pin.SE3):
    from geometry_msgs.msg import Pose

    xyzquat = pin.SE3ToXYZQUAT(M)
    msg = Pose()
    msg.position.x = float(xyzquat[0])
    msg.position.y = float(xyzquat[1])
    msg.position.z = float(xyzquat[2])
    msg.orientation.x = float(xyzquat[3])
    msg.orientation.y = float(xyzquat[4])
    msg.orientation.z = float(xyzquat[5])
    msg.orientation.w = float(xyzquat[6])
    return msg


class Phase(str, Enum):
    IDLE = "idle"
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    CLOSE = "close_gripper"
    LIFT = "lift"
    PREPLACE = "preplace"
    PLACE = "place"
    OPEN = "open_gripper"
    RETREAT = "retreat"
    DONE = "done"


@dataclass
class TaskParams:
    goal_topic: str
    base_frame: str
    ee_frame: str
    pick_frame: str
    pick_xyzquat: np.ndarray
    place_frame: str
    place_xyzquat: np.ndarray
    pregrasp_offset: np.ndarray
    lift_offset: np.ndarray
    retreat_offset: np.ndarray
    pos_tolerance: float
    rot_tolerance_rad: float
    phase_timeout_sec: float
    loop_forever: bool
    auto_start: bool
    armed: bool
    gripper_open_width: float
    gripper_open_effort: float
    gripper_close_width: float
    gripper_close_effort: float


class PickPlaceTaskManagerNode(Node):
    """Sequences a pick-and-place task by publishing goals and operating the gripper."""

    def __init__(self) -> None:
        super().__init__("pick_place_task_manager")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._declare_parameters()
        self._params = self._get_params()

        self._arm_id = self._derive_arm_id(self._params.base_frame, self._params.ee_frame)
        self.get_logger().info(f"Using arm_id '{self._arm_id}'.")

        self._gripper = FrankaGripperClient(self, arm_id=self._arm_id)

        self._goal_pub = self.create_publisher(
            PoseStamped,
            self._params.goal_topic,
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

        self._phase: Phase = Phase.IDLE
        self._phase_start_wall: Optional[float] = None
        self._current_goal_M: Optional[pin.SE3] = None
        self._forward_pick_to_place: bool = True
        self._completed_cycles: int = 0

        self._loop_timer = self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"PickPlace task manager ready. armed={self._params.armed} auto_start={self._params.auto_start}"
        )

    def _read_bool_param(self, name: str) -> bool:
        v = self.get_parameter(name).value
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
        return bool(v)

    @staticmethod
    def _derive_arm_id(base_frame: str, ee_frame: str) -> str:
        for token in ("_hand", "_link"):
            if token in ee_frame:
                return ee_frame.split(token)[0]
        if "_link" in base_frame:
            return base_frame.split("_link")[0]
        raise RuntimeError(
            f"Unable to derive arm_id from base_frame='{base_frame}' ee_frame='{ee_frame}'."
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "goal_topic",
            "/goal_pose",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="PoseStamped topic used as goal input for the high-level planner.",
            ),
        )
        self.declare_parameter("base_frame", "fer_link0")
        self.declare_parameter("ee_frame", "fer_hand_tcp")

        self.declare_parameter(
            "auto_start",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, start the task automatically (pregrasp -> ...).",
            ),
        )
        self.declare_parameter(
            "armed",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Safety gate. When false, the node publishes goals but does not command the gripper."
            ),
        )

        self.declare_parameter("pick_frame", "fer_link0")
        self.declare_parameter(
            "pick_xyzquat",
            [0.60, -0.30, 0.20, 0.0, 1.0, 0.0, 0.0],
            ParameterDescriptor(description="Pick pose as [x,y,z,qx,qy,qz,qw]."),
        )
        self.declare_parameter("place_frame", "fer_link0")
        self.declare_parameter(
            "place_xyzquat",
            [0.40, 0.30, 0.20, 0.0, 1.0, 0.0, 0.0],
            ParameterDescriptor(description="Place pose as [x,y,z,qx,qy,qz,qw]."),
        )

        self.declare_parameter("pregrasp_offset", [0.0, 0.0, 0.10])
        self.declare_parameter("lift_offset", [0.0, 0.0, 0.15])
        self.declare_parameter("retreat_offset", [0.0, 0.0, 0.10])

        self.declare_parameter("pos_tolerance", 0.02)
        self.declare_parameter("rot_tolerance_rad", 0.6)
        self.declare_parameter("phase_timeout_sec", 60.0)
        self.declare_parameter("loop_forever", True)

        self.declare_parameter("gripper_open_width", 0.039)
        self.declare_parameter("gripper_open_effort", 10.0)
        self.declare_parameter("gripper_close_width", 0.01)
        self.declare_parameter("gripper_close_effort", 30.0)

    def _get_params(self) -> TaskParams:
        def b(name: str) -> bool:
            v = self.get_parameter(name).value
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
            return bool(v)

        def arr(name: str, n: int) -> np.ndarray:
            v = np.array(self.get_parameter(name).value, dtype=np.float64)
            if v.size != n:
                raise ValueError(f"{name} must have length {n}, got {v.size}")
            return v

        return TaskParams(
            goal_topic=str(self.get_parameter("goal_topic").value),
            base_frame=str(self.get_parameter("base_frame").value),
            ee_frame=str(self.get_parameter("ee_frame").value),
            pick_frame=str(self.get_parameter("pick_frame").value),
            pick_xyzquat=arr("pick_xyzquat", 7),
            place_frame=str(self.get_parameter("place_frame").value),
            place_xyzquat=arr("place_xyzquat", 7),
            pregrasp_offset=arr("pregrasp_offset", 3),
            lift_offset=arr("lift_offset", 3),
            retreat_offset=arr("retreat_offset", 3),
            pos_tolerance=float(self.get_parameter("pos_tolerance").value),
            rot_tolerance_rad=float(self.get_parameter("rot_tolerance_rad").value),
            phase_timeout_sec=float(self.get_parameter("phase_timeout_sec").value),
            loop_forever=b("loop_forever"),
            auto_start=b("auto_start"),
            armed=b("armed"),
            gripper_open_width=float(self.get_parameter("gripper_open_width").value),
            gripper_open_effort=float(self.get_parameter("gripper_open_effort").value),
            gripper_close_width=float(self.get_parameter("gripper_close_width").value),
            gripper_close_effort=float(self.get_parameter("gripper_close_effort").value),
        )

    def _tick(self) -> None:
        # Refresh params that users may toggle at runtime.
        self._params.armed = self._read_bool_param("armed")
        self._params.auto_start = self._read_bool_param("auto_start")
        self._params.loop_forever = self._read_bool_param("loop_forever")

        if self._phase == Phase.IDLE:
            if self._params.auto_start:
                self._enter_phase(Phase.PREGRASP)
            return

        if self._phase in (Phase.DONE,):
            return

        if self._phase_start_wall is None:
            self._phase_start_wall = time.time()

        if (time.time() - self._phase_start_wall) > self._params.phase_timeout_sec:
            self.get_logger().error(f"Phase timeout in {self._phase}. Aborting.")
            self._enter_phase(Phase.DONE)
            return

        if self._phase in (Phase.CLOSE, Phase.OPEN):
            # Wait a short time for gripper action to take effect.
            if (time.time() - self._phase_start_wall) > 1.0:
                next_phase = Phase.LIFT if self._phase == Phase.CLOSE else Phase.RETREAT
                self._enter_phase(next_phase)
            return

        # Publish current phase goal.
        goal_M = self._compute_phase_goal(self._phase)
        if goal_M is None:
            return
        self._current_goal_M = goal_M
        self._publish_goal(goal_M)

        # Check convergence.
        ee_M = self._lookup_ee_in_base()
        if ee_M is None:
            return

        pos_err = np.linalg.norm(ee_M.translation - goal_M.translation)
        rot_vec = pin.log3(goal_M.rotation.T @ ee_M.rotation)
        rot_err = float(np.linalg.norm(rot_vec))

        if pos_err < self._params.pos_tolerance and rot_err < self._params.rot_tolerance_rad:
            self._advance_from(self._phase)

    def _enter_phase(self, phase: Phase) -> None:
        self._phase = phase
        self._phase_start_wall = time.time()
        self.get_logger().info(f"Entering phase: {phase.value}")

        if phase == Phase.PREGRASP:
            # Always start with open gripper.
            if self._params.armed:
                self._gripper.command(
                    position=self._params.gripper_open_width,
                    max_effort=self._params.gripper_open_effort,
                )
        elif phase == Phase.CLOSE:
            if self._params.armed:
                # In simulation, only GripperCommand is typically available.
                # On hardware, prefer the Franka Grasp action when available.
                res = self._gripper.grasp(
                    width=self._params.gripper_close_width,
                    force=self._params.gripper_close_effort,
                    timeout_sec=0.2,
                )
                if res is None:
                    self._gripper.command(
                        position=self._params.gripper_close_width,
                        max_effort=self._params.gripper_close_effort,
                    )
        elif phase == Phase.OPEN:
            if self._params.armed:
                self._gripper.command(
                    position=self._params.gripper_open_width,
                    max_effort=self._params.gripper_open_effort,
                )

    def _advance_from(self, phase: Phase) -> None:
        if phase == Phase.PREGRASP:
            self._enter_phase(Phase.GRASP)
        elif phase == Phase.GRASP:
            self._enter_phase(Phase.CLOSE)
        elif phase == Phase.LIFT:
            self._enter_phase(Phase.PREPLACE)
        elif phase == Phase.PREPLACE:
            self._enter_phase(Phase.PLACE)
        elif phase == Phase.PLACE:
            self._enter_phase(Phase.OPEN)
        elif phase == Phase.RETREAT:
            self._completed_cycles += 1
            if self._params.loop_forever:
                self._forward_pick_to_place = not self._forward_pick_to_place
                if self._forward_pick_to_place:
                    direction = "cube1->cube2"
                else:
                    direction = "cube2->cube1"
                self.get_logger().info(
                    f"Completed cycle {self._completed_cycles}. Switching direction to {direction}."
                )
                self._enter_phase(Phase.PREGRASP)
            else:
                self._enter_phase(Phase.DONE)

    def _compute_phase_goal(self, phase: Phase) -> Optional[pin.SE3]:
        pick_frame, pick_xyzquat, place_frame, place_xyzquat = self._active_pick_place_targets()

        if phase == Phase.PREGRASP:
            return self._lookup_pose_in_base(pick_frame, pick_xyzquat, self._params.pregrasp_offset)
        if phase == Phase.GRASP:
            return self._lookup_pose_in_base(pick_frame, pick_xyzquat, np.zeros(3))
        if phase == Phase.LIFT:
            return self._lookup_pose_in_base(pick_frame, pick_xyzquat, self._params.lift_offset)
        if phase == Phase.PREPLACE:
            return self._lookup_pose_in_base(place_frame, place_xyzquat, self._params.pregrasp_offset)
        if phase == Phase.PLACE:
            return self._lookup_pose_in_base(place_frame, place_xyzquat, np.zeros(3))
        if phase == Phase.RETREAT:
            return self._lookup_pose_in_base(place_frame, place_xyzquat, self._params.retreat_offset)
        return None

    def _active_pick_place_targets(self) -> tuple[str, np.ndarray, str, np.ndarray]:
        if self._forward_pick_to_place:
            return (
                self._params.pick_frame,
                self._params.pick_xyzquat,
                self._params.place_frame,
                self._params.place_xyzquat,
            )
        return (
            self._params.place_frame,
            self._params.place_xyzquat,
            self._params.pick_frame,
            self._params.pick_xyzquat,
        )

    def _lookup_pose_in_base(self, frame: str, xyzquat: np.ndarray, offset: np.ndarray) -> Optional[pin.SE3]:
        # Pose in the requested frame.
        M_local = pin.XYZQUATToSE3(xyzquat)
        M_local.translation = M_local.translation + offset

        if frame == self._params.base_frame:
            return M_local

        try:
            tf = self._tf_buffer.lookup_transform(self._params.base_frame, frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed {frame}->{self._params.base_frame}: {exc}")
            return None

        base_T_frame = transform_msg_to_se3(tf.transform)
        return base_T_frame * M_local

    def _lookup_ee_in_base(self) -> Optional[pin.SE3]:
        try:
            tf = self._tf_buffer.lookup_transform(self._params.base_frame, self._params.ee_frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed for ee: {exc}")
            return None
        return transform_msg_to_se3(tf.transform)

    def _publish_goal(self, goal_M: pin.SE3) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._params.base_frame
        msg.pose = se3_to_pose_msg(goal_M)
        self._goal_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickPlaceTaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
