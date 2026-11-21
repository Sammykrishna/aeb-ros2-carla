import math
from dataclasses import asdict

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection3DArray

from .state_machine import AebParams, AebStateMachine, lead_range_m
from .tracking import RangeTracker

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
CMD_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
MAX_TARGET_DECEL_MPS2 = 10.0


class AebController(Node):
    def __init__(self):
        super().__init__("aeb_controller")
        for name, default in asdict(AebParams()).items():
            self.declare_parameter(name, default)
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("object_timeout_s", 0.3)
        self.declare_parameter("track_window_s", 1.0)
        self.ego_speed_mps = 0.0

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        params = AebParams(**{name: self.get_parameter(name).value for name in asdict(AebParams())})
        self.sm = AebStateMachine(params)
        self.tracker = RangeTracker(window_s=self.get_parameter("track_window_s").value,
                                    stale_s=self.get_parameter("object_timeout_s").value)
        self.pubs = [self.create_lifecycle_publisher(Float32, "/aeb/brake_cmd", CMD_QOS),
                     self.create_lifecycle_publisher(String, "/aeb/state", CMD_QOS),
                     self.create_lifecycle_publisher(Float32, "/aeb/ttc", CMD_QOS),
                     self.create_lifecycle_publisher(Float32, "/aeb/required_decel", CMD_QOS)]
        self.subs = [self.create_subscription(Detection3DArray, "/aeb/objects_3d", self._on_objects, SENSOR_QOS),
                     self.create_subscription(Odometry, "/ego/odom", self._on_odom, SENSOR_QOS)]
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.sm.reset()
        self.tracker.reset()
        self.timer = self.create_timer(1.0 / self.get_parameter("rate_hz").value, self._tick)
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.destroy_timer(self.timer)
        self.pubs[0].publish(Float32(data=0.0))
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        for pub in self.pubs:
            self.destroy_publisher(pub)
        for sub in self.subs:
            self.destroy_subscription(sub)
        return TransitionCallbackReturn.SUCCESS

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry):
        self.ego_speed_mps = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def _on_objects(self, msg: Detection3DArray):
        pts_xy_m = np.array([[d.bbox.center.position.x, d.bbox.center.position.y]
                             for d in msg.detections]).reshape(-1, 2)
        range_m = lead_range_m(pts_xy_m)
        if range_m is None:
            return
        # message time, not arrival time: queueing jitter at 14 m/s closing looks like metres of range noise
        self.tracker.update(Time.from_msg(msg.header.stamp).nanoseconds * 1e-9, range_m)

    def _tick(self):
        now_s = self._now_s()
        estimate = self.tracker.estimate(now_s)
        range_m, closing_mps, closing_accel_mps2 = estimate if estimate else (None, 0.0, 0.0)
        target_decel_mps2 = float(np.clip(closing_accel_mps2, 0.0, MAX_TARGET_DECEL_MPS2))
        state, brake, ttc_s, a_req_mps2 = self.sm.step(now_s, self.ego_speed_mps, range_m, closing_mps,
                                                       target_decel_mps2)
        brake_pub, state_pub, ttc_pub, decel_pub = self.pubs
        brake_pub.publish(Float32(data=float(brake)))
        state_pub.publish(String(data=state.value))
        ttc_pub.publish(Float32(data=float(min(ttc_s, 99.0))))
        decel_pub.publish(Float32(data=float(a_req_mps2)))


def main(args=None):
    rclpy.init(args=args)
    node = AebController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
