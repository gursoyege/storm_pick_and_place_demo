from __future__ import annotations
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class ObjectManagerParams:
    base_frame: str
    objects_markers_topic: str

    pick_frame: str
    pick_xyzquat: list[float]
    pick_dims: list[float]

    place_frame: str
    place_xyzquat: list[float]
    place_dims: list[float]

    barrier_enabled: bool
    barrier_height: float

    publish_rate_hz: float


class ObjectManagerNode(Node):
    """Publishes pick/place objects as TF frames and MarkerArray (used by RViz + STORM planner)."""

    def __init__(self) -> None:
        super().__init__("object_manager")

        self._declare_parameters()
        self._params = self._get_params()

        self._tf_broadcaster = TransformBroadcaster(self)
        self._pub_markers = self.create_publisher(
            MarkerArray,
            self._params.objects_markers_topic,
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

        period = 1.0 / float(self._params.publish_rate_hz)
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"object_manager publishing TF + markers on {self._params.objects_markers_topic} @ {self._params.publish_rate_hz:.1f} Hz"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "base_frame",
            "fer_link0",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            "objects_markers_topic",
            "/storm_pick_place/objects_markers",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )

        self.declare_parameter("pick_frame", "fer_link0")
        self.declare_parameter(
            "pick_xyzquat",
            [0.60, -0.30, 0.20, 0.0, 0.0, 0.0, 1.0],
            ParameterDescriptor(description="Pick object pose as [x,y,z,qx,qy,qz,qw] in pick_frame."),
        )
        self.declare_parameter("pick_dims", [0.04, 0.04, 0.04], ParameterDescriptor(description="Pick cube dims [x,y,z]."))

        self.declare_parameter("place_frame", "fer_link0")
        self.declare_parameter(
            "place_xyzquat",
            [0.40, 0.30, 0.20, 0.0, 0.0, 0.0, 1.0],
            ParameterDescriptor(description="Place target pose as [x,y,z,qx,qy,qz,qw] in place_frame."),
        )
        self.declare_parameter("place_dims", [0.06, 0.06, 0.02], ParameterDescriptor(description="Place cube dims [x,y,z]."))

        self.declare_parameter(
            "barrier_enabled",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, publish an additional tall collision cube between pick and place.",
            ),
        )
        self.declare_parameter(
            "barrier_height",
            2.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Height (Z dimension) of the barrier cube in meters.",
            ),
        )

        self.declare_parameter("publish_rate_hz", 10.0)

    def _get_params(self) -> ObjectManagerParams:
        def arr(name: str, n: int) -> list[float]:
            v = list(self.get_parameter(name).value)
            if len(v) != n:
                raise ValueError(f"{name} must have length {n}, got {len(v)}")
            return [float(x) for x in v]

        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be > 0")

        barrier_height = float(self.get_parameter("barrier_height").value)
        if barrier_height <= 0.0:
            raise ValueError("barrier_height must be > 0")

        return ObjectManagerParams(
            base_frame=str(self.get_parameter("base_frame").value),
            objects_markers_topic=str(self.get_parameter("objects_markers_topic").value),
            pick_frame=str(self.get_parameter("pick_frame").value),
            pick_xyzquat=arr("pick_xyzquat", 7),
            pick_dims=arr("pick_dims", 3),
            place_frame=str(self.get_parameter("place_frame").value),
            place_xyzquat=arr("place_xyzquat", 7),
            place_dims=arr("place_dims", 3),
            barrier_enabled=bool(self.get_parameter("barrier_enabled").value),
            barrier_height=barrier_height,
            publish_rate_hz=rate_hz,
        )

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()

        # TF frames for downstream consumers.
        pick_tf = TransformStamped()
        pick_tf.header.stamp = now
        pick_tf.header.frame_id = self._params.pick_frame
        pick_tf.child_frame_id = "pp_pick_object"
        pick_tf.transform.translation.x = float(self._params.pick_xyzquat[0])
        pick_tf.transform.translation.y = float(self._params.pick_xyzquat[1])
        pick_tf.transform.translation.z = float(self._params.pick_xyzquat[2])
        pick_tf.transform.rotation.x = float(self._params.pick_xyzquat[3])
        pick_tf.transform.rotation.y = float(self._params.pick_xyzquat[4])
        pick_tf.transform.rotation.z = float(self._params.pick_xyzquat[5])
        pick_tf.transform.rotation.w = float(self._params.pick_xyzquat[6])

        place_tf = TransformStamped()
        place_tf.header.stamp = now
        place_tf.header.frame_id = self._params.place_frame
        place_tf.child_frame_id = "pp_place_target"
        place_tf.transform.translation.x = float(self._params.place_xyzquat[0])
        place_tf.transform.translation.y = float(self._params.place_xyzquat[1])
        place_tf.transform.translation.z = float(self._params.place_xyzquat[2])
        place_tf.transform.rotation.x = float(self._params.place_xyzquat[3])
        place_tf.transform.rotation.y = float(self._params.place_xyzquat[4])
        place_tf.transform.rotation.z = float(self._params.place_xyzquat[5])
        place_tf.transform.rotation.w = float(self._params.place_xyzquat[6])

        tfs = [pick_tf, place_tf]

        # MarkerArray for RViz and for STORM planner (collision primitives).
        markers = []

        pick_marker = Marker()
        pick_marker.header.stamp = now
        pick_marker.header.frame_id = self._params.base_frame
        pick_marker.ns = "pp_pick_object"
        pick_marker.id = 0
        pick_marker.type = Marker.CUBE
        pick_marker.action = Marker.ADD
        pick_marker.pose.position.x = float(self._params.pick_xyzquat[0])
        pick_marker.pose.position.y = float(self._params.pick_xyzquat[1])
        pick_marker.pose.position.z = float(self._params.pick_xyzquat[2])
        pick_marker.pose.orientation.x = float(self._params.pick_xyzquat[3])
        pick_marker.pose.orientation.y = float(self._params.pick_xyzquat[4])
        pick_marker.pose.orientation.z = float(self._params.pick_xyzquat[5])
        pick_marker.pose.orientation.w = float(self._params.pick_xyzquat[6])
        pick_marker.scale.x = float(self._params.pick_dims[0])
        pick_marker.scale.y = float(self._params.pick_dims[1])
        pick_marker.scale.z = float(self._params.pick_dims[2])
        pick_marker.color.r = 1.0
        pick_marker.color.g = 0.5
        pick_marker.color.b = 0.0
        pick_marker.color.a = 0.8
        pick_marker.lifetime.sec = 0
        markers.append(pick_marker)

        place_marker = Marker()
        place_marker.header.stamp = now
        place_marker.header.frame_id = self._params.base_frame
        place_marker.ns = "pp_place_target"
        place_marker.id = 0
        place_marker.type = Marker.CUBE
        place_marker.action = Marker.ADD
        place_marker.pose.position.x = float(self._params.place_xyzquat[0])
        place_marker.pose.position.y = float(self._params.place_xyzquat[1])
        place_marker.pose.position.z = float(self._params.place_xyzquat[2])
        place_marker.pose.orientation.x = float(self._params.place_xyzquat[3])
        place_marker.pose.orientation.y = float(self._params.place_xyzquat[4])
        place_marker.pose.orientation.z = float(self._params.place_xyzquat[5])
        place_marker.pose.orientation.w = float(self._params.place_xyzquat[6])
        place_marker.scale.x = float(self._params.place_dims[0])
        place_marker.scale.y = float(self._params.place_dims[1])
        place_marker.scale.z = float(self._params.place_dims[2])
        place_marker.color.r = 0.0
        place_marker.color.g = 0.8
        place_marker.color.b = 0.2
        place_marker.color.a = 0.4
        place_marker.lifetime.sec = 0
        markers.append(place_marker)

        if self._params.barrier_enabled:
            # A tall collision cube placed between pick and place.
            mid_x = 0.5 * (float(self._params.pick_xyzquat[0]) + float(self._params.place_xyzquat[0]))
            mid_y = 0.5 * (float(self._params.pick_xyzquat[1]) + float(self._params.place_xyzquat[1]))

            barrier_tf = TransformStamped()
            barrier_tf.header.stamp = now
            barrier_tf.header.frame_id = self._params.base_frame
            barrier_tf.child_frame_id = "pp_barrier"
            barrier_tf.transform.translation.x = float(mid_x)
            barrier_tf.transform.translation.y = float(mid_y)
            barrier_tf.transform.translation.z = float(0.5 * self._params.barrier_height)
            barrier_tf.transform.rotation.w = 1.0
            tfs.append(barrier_tf)

            barrier_marker = Marker()
            barrier_marker.header.stamp = now
            barrier_marker.header.frame_id = self._params.base_frame
            barrier_marker.ns = "pp_barrier"
            barrier_marker.id = 0
            barrier_marker.type = Marker.CUBE
            barrier_marker.action = Marker.ADD
            barrier_marker.pose.position.x = float(mid_x)
            barrier_marker.pose.position.y = float(mid_y)
            barrier_marker.pose.position.z = float(0.5 * self._params.barrier_height)
            barrier_marker.pose.orientation.w = 1.0
            barrier_marker.scale.x = float(max(self._params.pick_dims[0], self._params.place_dims[0]))
            barrier_marker.scale.y = float(max(self._params.pick_dims[1], self._params.place_dims[1]))
            barrier_marker.scale.z = float(self._params.barrier_height)
            barrier_marker.color.r = 0.2
            barrier_marker.color.g = 0.2
            barrier_marker.color.b = 1.0
            barrier_marker.color.a = 0.35
            barrier_marker.lifetime.sec = 0
            markers.append(barrier_marker)

        self._tf_broadcaster.sendTransform(tfs)
        self._pub_markers.publish(MarkerArray(markers=markers))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
