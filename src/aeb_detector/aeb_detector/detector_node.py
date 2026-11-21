import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

COCO_ROAD_USERS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")
        self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("conf", 0.4)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("image_topic", "/ego/camera/image")
        self.model = YOLO(self.get_parameter("model").value)
        # the first cuda inference takes ~3 s, so pay for it before subscribing to images
        self._predict(np.zeros((540, 960, 3), np.uint8))
        self.pub = self.create_publisher(Detection2DArray, "/aeb/detections_2d", 1)
        self.create_subscription(Image, self.get_parameter("image_topic").value,
                                 self._on_image, qos_profile_sensor_data)

    def _predict(self, img_bgr: np.ndarray):
        return self.model.predict(img_bgr, conf=self.get_parameter("conf").value,
                                  device=self.get_parameter("device").value,
                                  imgsz=self.get_parameter("imgsz").value,
                                  classes=list(COCO_ROAD_USERS), verbose=False)[0]

    def _on_image(self, msg: Image):
        # cv_bridge is built against numpy 1.x and breaks next to the venv's numpy 2
        img_bgr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        result = self._predict(img_bgr)
        out = Detection2DArray(header=msg.header)
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = Detection2D(header=msg.header)
            det.bbox.center.position.x, det.bbox.center.position.y = (x1 + x2) / 2, (y1 + y2) / 2
            det.bbox.size_x, det.bbox.size_y = x2 - x1, y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = COCO_ROAD_USERS[int(box.cls)]
            hyp.hypothesis.score = float(box.conf)
            det.results.append(hyp)
            out.detections.append(det)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
