import json

import rclpy
from aeb_common.qos import declare_stream_qos
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from .regression_stats import latency_summary


class LatencyProbe(Node):
    # a node of its own because the scenario node only handles one callback per sim tick and would add its own queueing
    def __init__(self):
        super().__init__("latency_probe")
        self.latencies_ms: list[float] = []
        self.create_subscription(Detection3DArray, "/aeb/objects_3d", self._on_objects, declare_stream_qos(self))
        self.pub = self.create_publisher(String, "/probe/latency", 1)
        self.create_timer(0.5, self._publish)

    def _on_objects(self, msg: Detection3DArray):
        self.latencies_ms.append((self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-6)

    def _publish(self):
        self.pub.publish(String(data=json.dumps(latency_summary(self.latencies_ms))))


def main(args=None):
    rclpy.init(args=args)
    node = LatencyProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
