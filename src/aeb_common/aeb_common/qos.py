from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

RELIABILITY = {"best_effort": ReliabilityPolicy.BEST_EFFORT, "reliable": ReliabilityPolicy.RELIABLE}


def stream_qos(reliability: str, depth: int) -> QoSProfile:
    return QoSProfile(reliability=RELIABILITY[reliability], history=HistoryPolicy.KEEP_LAST, depth=depth)


def declare_stream_qos(node: Node) -> QoSProfile:
    node.declare_parameter("qos_reliability", "reliable")
    node.declare_parameter("qos_depth", 10)
    return stream_qos(node.get_parameter("qos_reliability").value, node.get_parameter("qos_depth").value)
