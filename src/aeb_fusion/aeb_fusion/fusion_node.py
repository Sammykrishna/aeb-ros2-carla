import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from aeb_common.qos import declare_stream_qos
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray

from .geometry import nearest_cluster_xyz, project_to_image


class FusionNode(Node):
    def __init__(self):
        super().__init__("aeb_fusion")
        self.declare_parameter("cam_in_lidar_xyz", [1.5, 0.0, -0.2])
        self.declare_parameter("bumper_offset_m", 2.5)
        self.declare_parameter("box_shrink", 1.0)
        self.declare_parameter("min_points", 2)
        self.declare_parameter("depth_window_m", 0.5)
        self.declare_parameter("slop_s", 0.05)
        self.K: np.ndarray | None = None
        qos = declare_stream_qos(self)

        self.pub = self.create_publisher(Detection3DArray, "/aeb/objects_3d", qos)
        self.create_subscription(CameraInfo, "/ego/camera/camera_info", self._on_info, qos)
        dets_sub = message_filters.Subscriber(self, Detection2DArray, "/aeb/detections_2d",
                                              qos_profile=qos)
        cloud_sub = message_filters.Subscriber(self, PointCloud2, "/ego/lidar/points",
                                               qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [dets_sub, cloud_sub], 5, self.get_parameter("slop_s").value)
        self.sync.registerCallback(self._on_sync)

    def _on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def _on_sync(self, dets: Detection2DArray, cloud: PointCloud2):
        if self.K is None:
            return
        pts_lidar = point_cloud2.read_points_numpy(cloud, field_names=("x", "y", "z"), skip_nans=True)
        cam_in_lidar_m = np.array(self.get_parameter("cam_in_lidar_xyz").value)
        u_px, v_px, in_front = project_to_image(pts_lidar, cam_in_lidar_m, self.K)
        bumper_m = self.get_parameter("bumper_offset_m").value

        out = Detection3DArray(header=cloud.header)
        for det in dets.detections:
            box_px = (det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.size_x, det.bbox.size_y)
            xyz_m = nearest_cluster_xyz(pts_lidar, u_px, v_px, in_front, box_px,
                                        self.get_parameter("box_shrink").value,
                                        self.get_parameter("min_points").value,
                                        self.get_parameter("depth_window_m").value)
            if xyz_m is None:
                continue
            obj = Detection3D(header=cloud.header, results=det.results)
            obj.bbox.center.position.x = float(xyz_m[0] - bumper_m)
            obj.bbox.center.position.y = float(xyz_m[1])
            obj.bbox.center.position.z = float(xyz_m[2])
            out.detections.append(obj)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
