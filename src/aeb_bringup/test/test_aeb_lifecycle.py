import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection3D, Detection3DArray


@pytest.mark.launch_test
def generate_test_description():
    node = launch_ros.actions.Node(package="aeb_controller", executable="aeb_controller_node",
                                   name="aeb_controller", output="screen")
    return launch.LaunchDescription([node, launch_testing.actions.ReadyToTest()]), {}


class TestAeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("aeb_test")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _transition(self, tid):
        cli = self.node.create_client(ChangeState, "/aeb_controller/change_state")
        self.assertTrue(cli.wait_for_service(10.0))
        fut = cli.call_async(ChangeState.Request(transition=Transition(id=tid)))
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=5.0)
        self.assertTrue(fut.result().success)

    def test_full_brake_on_closing_target(self):
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self._transition(Transition.TRANSITION_CONFIGURE)
        self._transition(Transition.TRANSITION_ACTIVATE)

        brakes, states = [], []
        self.node.create_subscription(Float32, "/aeb/brake_cmd", lambda m: brakes.append(m.data), 10)
        self.node.create_subscription(String, "/aeb/state", lambda m: states.append(m.data), 10)
        pub_o = self.node.create_publisher(Detection3DArray, "/aeb/objects_3d", be)
        pub_e = self.node.create_publisher(Odometry, "/ego/odom", be)

        rng, speed, dt = 40.0, 13.9, 0.05          # 50 km/h at 40 m from a stationary target
        end = time.time() + 6.0
        while time.time() < end and rng > 0.5:
            odom = Odometry()
            odom.twist.twist.linear.x = speed
            pub_e.publish(odom)
            det = Detection3D()
            det.bbox.center.position.x = rng
            objects = Detection3DArray(detections=[det])
            objects.header.stamp = self.node.get_clock().now().to_msg()
            pub_o.publish(objects)
            rclpy.spin_once(self.node, timeout_sec=dt)
            rng -= speed * dt
            time.sleep(dt)
        self.assertIn("FULL_BRAKE", states)
        self.assertEqual(max(brakes), 1.0)

        self._transition(Transition.TRANSITION_DEACTIVATE)
