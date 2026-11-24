import json
import math
import queue
import time

import carla
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection3DArray

from .scenario_cfg import jitter_cfg

DT_S = 0.05
MAX_START_MISALIGN_M = 0.3
MAX_DRIFT_M = 0.5
ZERO = carla.Vector3D(0.0, 0.0, 0.0)
NO_TARGET_GAP_M = 999.0
STATE_COLORS = {"IDLE": (0, 200, 0), "WARNING": (255, 200, 0), "PARTIAL_BRAKE": (255, 120, 0),
                "FULL_BRAKE": (255, 0, 0), "HOLD": (0, 120, 255)}
IMG_W_PX, IMG_H_PX, FOV_DEG = 960, 540, 90.0
LIDAR_ATTRS = dict(channels="32", range="100", points_per_second="600000",
                   rotation_frequency="20", upper_fov="10", lower_fov="-30")


def is_straight(wp: carla.Waypoint, length_m: float, max_yaw_deg: float = 0.5) -> bool:
    cur = wp
    for _ in range(int(length_m // 5)):
        nxt = cur.next(5.0)
        if not nxt:
            return False
        cur = nxt[0]
        dyaw_deg = abs((cur.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180)
        if cur.is_junction or dyaw_deg > max_yaw_deg:
            return False
    return True


def hold_velocity(actor: carla.Actor, direction: carla.Vector3D, speed_mps: float):
    actor.set_target_velocity(direction * speed_mps)
    actor.set_target_angular_velocity(ZERO)


def path_drift_m(loc: carla.Location, origin: carla.Location, direction: carla.Vector3D) -> float:
    return abs((loc.x - origin.x) * direction.y - (loc.y - origin.y) * direction.x)


def target_speed_mps(cfg: dict, t_s: float) -> float:
    speed_mps = cfg["target_kmh"] / 3.6
    if cfg["type"] != "braking":
        return speed_mps
    return max(0.0, speed_mps - cfg["target_decel_mps2"] * max(0.0, t_s - cfg["brake_after_s"]))


def bumper_gap_m(delta_xy_m: np.ndarray, fwd_xy: np.ndarray, ego_front_m: float,
                 tgt_rear_m: float) -> tuple[float, float]:
    fwd_xy = fwd_xy / np.linalg.norm(fwd_xy)
    right_xy = np.array([-fwd_xy[1], fwd_xy[0]])
    return float(delta_xy_m @ fwd_xy) - ego_front_m - tgt_rear_m, float(delta_xy_m @ right_xy)


def gap_to_target(ego: carla.Actor, tgt: carla.Actor) -> tuple[float, float]:
    ego_loc, tgt_loc = ego.get_location(), tgt.get_location()
    fwd = ego.get_transform().get_forward_vector()
    return bumper_gap_m(np.array([tgt_loc.x - ego_loc.x, tgt_loc.y - ego_loc.y]), np.array([fwd.x, fwd.y]),
                        ego.bounding_box.location.x + ego.bounding_box.extent.x,
                        tgt.bounding_box.extent.x - tgt.bounding_box.location.x)


def place_target_ahead(ego: carla.Actor, tgt: carla.Actor, gap_m: float, lateral_m: float = 0.0):
    ego_tf, ego_loc, tgt_loc = ego.get_transform(), ego.get_location(), tgt.get_location()
    fwd, right = ego_tf.get_forward_vector(), ego_tf.get_right_vector()
    dist_m = gap_m + ego.bounding_box.location.x + ego.bounding_box.extent.x \
        + tgt.bounding_box.extent.x - tgt.bounding_box.location.x
    tgt.set_transform(carla.Transform(
        carla.Location(ego_loc.x + fwd.x * dist_m + right.x * lateral_m,
                       ego_loc.y + fwd.y * dist_m + right.y * lateral_m, tgt_loc.z),
        carla.Rotation(pitch=ego_tf.rotation.pitch, yaw=ego_tf.rotation.yaw)))


def spawn_target(world: carla.World, ego: carla.Actor, wp_ego: carla.Waypoint, cfg: dict,
                 actors: list[carla.Actor]) -> carla.Actor | None:
    if cfg.get("no_target"):
        world.tick()
        return None
    lateral_m = cfg.get("lateral_offset_m", 0.0)
    tgt = spawn_on_lane(world, cfg.get("target_model", "vehicle.audi.a2"), wp_ego.next(cfg["gap_m"] + 10.0)[0])
    actors.append(tgt)
    world.tick()  # in sync mode a spawned actor reports a zero transform until the first tick
    cmd_gap_m, cmd_lat_m = cfg["gap_m"], lateral_m
    for _ in range(3):  # some bodies settle a few tenths off their placement, so correct by the measured error
        place_target_ahead(ego, tgt, cmd_gap_m, cmd_lat_m)
        world.tick()
        gap_m, lat_m = gap_to_target(ego, tgt)
        if abs(gap_m - cfg["gap_m"]) <= MAX_START_MISALIGN_M and abs(lat_m - lateral_m) <= MAX_START_MISALIGN_M:
            break
        cmd_gap_m, cmd_lat_m = cmd_gap_m + cfg["gap_m"] - gap_m, cmd_lat_m + lateral_m - lat_m
    if abs(gap_m - cfg["gap_m"]) > MAX_START_MISALIGN_M or abs(lat_m - lateral_m) > MAX_START_MISALIGN_M:
        raise RuntimeError(f"bad start geometry: gap {gap_m:.2f} m, lateral offset {lat_m:.2f} m")
    hold_velocity(tgt, ego.get_transform().get_forward_vector(), cfg["target_kmh"] / 3.6)
    return tgt


def draw_view(world: carla.World, ego: carla.Actor, node: "ScenarioNode", gap_m: float):
    tf = ego.get_transform()
    loc, fwd, right = tf.location, tf.get_forward_vector(), tf.get_right_vector()
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(loc.x - fwd.x * 11.0 + right.x * 2.5, loc.y - fwd.y * 11.0 + right.y * 2.5, loc.z + 4.5),
        carla.Rotation(pitch=-14.0, yaw=tf.rotation.yaw - 8.0)))

    color = carla.Color(*STATE_COLORS.get(node.aeb_state, (255, 255, 255)))
    text = (f"AEB {node.aeb_state}   brake {node.brake:.2f}   need {node.a_req_mps2:.1f} m/s2   "
            f"{ego.get_velocity().length() * 3.6:.0f} km/h   gap {gap_m:.1f} m")
    life_s = 2 * DT_S
    world.debug.draw_string(carla.Location(loc.x, loc.y, loc.z + 3.0), text, color=color, life_time=life_s)

    front_m = ego.bounding_box.location.x + ego.bounding_box.extent.x

    def draw_range(dist_m: float, height_m: float, line_color: carla.Color):
        start = carla.Location(loc.x + fwd.x * front_m, loc.y + fwd.y * front_m, loc.z + height_m)
        end = carla.Location(start.x + fwd.x * dist_m, start.y + fwd.y * dist_m, start.z)
        world.debug.draw_line(start, end, thickness=0.08, color=line_color, life_time=life_s)
        world.debug.draw_point(end, size=0.15, color=line_color, life_time=life_s)

    draw_range(gap_m, 1.3, carla.Color(0, 255, 0))
    if node.perceived_range_m is not None and time.monotonic() - node.perceived_at_s < 0.3:
        draw_range(node.perceived_range_m, 0.9, carla.Color(255, 0, 0))


def lidar_to_pts_ros(raw: bytes) -> np.ndarray:
    pts = np.frombuffer(raw, np.float32).reshape(-1, 4)[:, :3].copy()
    pts[:, 1] *= -1.0  # carla is left-handed (y right), ros is y left
    return pts


def spawn_on_lane(world: carla.World, model: str, wp: carla.Waypoint) -> carla.Actor:
    tf = wp.transform
    tf.location.z += 0.3  # spawning exactly on the road surface collides with it
    return world.spawn_actor(world.get_blueprint_library().find(model), tf)


def attach_sensor(world: carla.World, sensor_id: str, attrs: dict[str, str], tf: carla.Transform,
                  parent: carla.Actor) -> carla.Sensor:
    bp = world.get_blueprint_library().find(sensor_id)
    for key, value in attrs.items():
        bp.set_attribute(key, value)
    return world.spawn_actor(bp, tf, attach_to=parent)


class ScenarioNode(Node):
    def __init__(self):
        super().__init__("carla_scenario")
        self.declare_parameter("scenario", "CCRs_50")
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("town", "Town04")
        self.declare_parameter("timeout_s", 30.0)
        self.declare_parameter("wait_for_pipeline", True)
        self.declare_parameter("spectator", False)
        self.declare_parameter("end_pause_s", 0.0)
        self.declare_parameter("seed", -1)
        self.declare_parameter("summary_path", "")
        self.name = self.get_parameter("scenario").value
        with open(get_package_share_directory("aeb_bringup") + "/config/scenarios.yaml") as f:
            self.cfg = yaml.safe_load(f)[self.name]
        if self.get_parameter("seed").value >= 0:
            self.cfg = jitter_cfg(self.cfg, np.random.default_rng(self.get_parameter("seed").value))

        self.brake = 0.0
        self.aeb_state: str | None = None
        self.a_req_mps2 = 0.0
        self.perceived_range_m: float | None = None
        self.perceived_at_s = 0.0
        self.pub_img = self.create_publisher(Image, "/ego/camera/image", qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, "/ego/camera/camera_info", qos_profile_sensor_data)
        self.pub_pcl = self.create_publisher(PointCloud2, "/ego/lidar/points", qos_profile_sensor_data)
        self.pub_odom = self.create_publisher(Odometry, "/ego/odom", qos_profile_sensor_data)
        self.pub_gap = self.create_publisher(Float32, "/ground_truth/gap", 10)
        self.pub_lat = self.create_publisher(Float32, "/ground_truth/lateral_offset", 10)
        self.pub_col = self.create_publisher(Float32, "/ground_truth/collision", 10)
        self.pub_res = self.create_publisher(String, "/scenario/result", 10)
        self.create_subscription(Float32, "/aeb/brake_cmd", self._on_brake, 1)
        self.create_subscription(String, "/aeb/state", self._on_state, 1)
        self.create_subscription(Float32, "/aeb/required_decel", self._on_decel, 1)
        self.create_subscription(Detection3DArray, "/aeb/objects_3d", self._on_objects, qos_profile_sensor_data)

    def _on_brake(self, msg: Float32):
        self.brake = msg.data

    def _on_state(self, msg: String):
        self.aeb_state = msg.data

    def _on_decel(self, msg: Float32):
        self.a_req_mps2 = msg.data

    def _on_objects(self, msg: Detection3DArray):
        ahead_m = [d.bbox.center.position.x for d in msg.detections if d.bbox.center.position.x > 0.0]
        if ahead_m:
            self.perceived_range_m, self.perceived_at_s = min(ahead_m), time.monotonic()

    def wait_for_pipeline(self, timeout_s: float = 90.0):
        consumers = {"/ego/camera/image": "yolo_detector", "/ego/lidar/points": "aeb_fusion",
                     "/ego/odom": "aeb_controller"}
        t_end_s = time.monotonic() + timeout_s
        # the controller only publishes /aeb/state once its lifecycle is active
        while not (self.aeb_state and all(
                any(info.node_name == name for info in self.get_subscriptions_info_by_topic(topic))
                for topic, name in consumers.items())):
            if time.monotonic() > t_end_s:
                raise TimeoutError("detector, fusion and active controller not up")
            rclpy.spin_once(self, timeout_sec=0.1)

    def publish_frame(self, img: carla.Image, lidar: carla.LidarMeasurement, ego: carla.Actor,
                      gap_m: float, lat_m: float):
        stamp = self.get_clock().now().to_msg()
        img_bgr = np.frombuffer(img.raw_data, np.uint8).reshape(IMG_H_PX, IMG_W_PX, 4)[:, :, :3]
        img_msg = Image(height=IMG_H_PX, width=IMG_W_PX, encoding="bgr8", step=IMG_W_PX * 3, data=img_bgr.tobytes())
        img_msg.header.stamp, img_msg.header.frame_id = stamp, "camera"

        fx_px = IMG_W_PX / (2 * math.tan(math.radians(FOV_DEG) / 2))
        info = CameraInfo(width=IMG_W_PX, height=IMG_H_PX,
                          k=[fx_px, 0.0, IMG_W_PX / 2, 0.0, fx_px, IMG_H_PX / 2, 0.0, 0.0, 1.0])
        info.header.stamp, info.header.frame_id = stamp, "camera"

        pts_lidar = lidar_to_pts_ros(lidar.raw_data)
        cloud = PointCloud2(height=1, width=len(pts_lidar), is_dense=True, point_step=12,
                            row_step=12 * len(pts_lidar), data=pts_lidar.tobytes(),
                            fields=[PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                                    for i, n in enumerate("xyz")])
        cloud.header.stamp, cloud.header.frame_id = stamp, "lidar"

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "odom"
        odom.twist.twist.linear.x = float(ego.get_velocity().length())

        self.pub_img.publish(img_msg)
        self.pub_info.publish(info)
        self.pub_pcl.publish(cloud)
        self.pub_odom.publish(odom)
        self.pub_gap.publish(Float32(data=gap_m))
        self.pub_lat.publish(Float32(data=lat_m))


def run(node: ScenarioNode):
    cfg = node.cfg
    if node.get_parameter("wait_for_pipeline").value:
        node.wait_for_pipeline()
    client = carla.Client(node.get_parameter("host").value, node.get_parameter("port").value)
    client.set_timeout(60.0)
    town = node.get_parameter("town").value
    world = client.get_world()
    if not world.get_map().name.endswith(town):  # reloading the already loaded map has crashed the server
        world = client.load_world(town)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, DT_S
    world.apply_settings(settings)

    actors: list[carla.Actor] = []
    try:
        road_m = cfg.get("road_m", cfg["gap_m"] + 60.0)
        wp_ego = next(wp for wp in world.get_map().generate_waypoints(20.0)
                      if not wp.is_junction and is_straight(wp, road_m))
        ego = spawn_on_lane(world, "vehicle.tesla.model3", wp_ego)
        actors.append(ego)
        tgt = spawn_target(world, ego, wp_ego, cfg, actors)
        ref_dir, ref_origin = ego.get_transform().get_forward_vector(), ego.get_location()
        hold_velocity(ego, ref_dir, cfg["ego_kmh"] / 3.6)

        cam = attach_sensor(world, "sensor.camera.rgb",
                            dict(image_size_x=str(IMG_W_PX), image_size_y=str(IMG_H_PX), fov=str(FOV_DEG)),
                            carla.Transform(carla.Location(x=1.5, z=1.6)), ego)
        lidar = attach_sensor(world, "sensor.lidar.ray_cast", LIDAR_ATTRS,
                              carla.Transform(carla.Location(x=0.0, z=1.8)), ego)
        collision = attach_sensor(world, "sensor.other.collision", {}, carla.Transform(), ego)
        actors += [cam, lidar, collision]
        cam_q, lidar_q, col_q = queue.Queue(), queue.Queue(), queue.Queue()
        cam.listen(cam_q.put)
        lidar.listen(lidar_q.put)
        collision.listen(col_q.put)

        duration_s = cfg.get("duration_s", node.get_parameter("timeout_s").value)
        t_s, hold_s, min_gap_m, max_brake, collided, braking_began = 0.0, 0.0, float("inf"), 0.0, False, False
        first_brake, impact_kmh, prev_speed_kmh, max_drift_m = None, 0.0, 0.0, 0.0
        tick_wall_s = time.monotonic()
        while rclpy.ok() and t_s < duration_s and not collided:
            rclpy.spin_once(node, timeout_sec=0.0)
            braking_began = braking_began or node.brake > 0.0
            if braking_began:  # after a release the car coasts, re-imposing the scripted speed would undo the braking
                ego.apply_control(carla.VehicleControl(brake=float(node.brake)))
                ego.set_target_angular_velocity(ZERO)  # the ego has no steering and braking alone has yawed it
            else:
                hold_velocity(ego, ref_dir, cfg["ego_kmh"] / 3.6)

            if tgt is not None and cfg["target_kmh"] == 0:
                tgt.apply_control(carla.VehicleControl(hand_brake=True))
            elif tgt is not None:
                hold_velocity(tgt, ref_dir, target_speed_mps(cfg, t_s))

            # pace to wall clock so the controller's node-clock ttc matches sim time
            time.sleep(max(0.0, DT_S - (time.monotonic() - tick_wall_s)))
            tick_wall_s = time.monotonic()
            world.tick()
            t_s += DT_S

            gap_m, lat_m = gap_to_target(ego, tgt) if tgt is not None else (NO_TARGET_GAP_M, 0.0)
            min_gap_m = min(min_gap_m, gap_m)
            max_brake = max(max_brake, node.brake)
            speed_kmh = ego.get_velocity().length() * 3.6
            max_drift_m = max(max_drift_m, path_drift_m(ego.get_location(), ref_origin, ref_dir))
            if node.brake > 0.0 and first_brake is None:
                first_brake = (gap_m, speed_kmh)
            node.publish_frame(cam_q.get(timeout=2), lidar_q.get(timeout=2), ego, gap_m, lat_m)
            if node.get_parameter("spectator").value:
                draw_view(world, ego, node, gap_m)
            while not col_q.empty():
                event = col_q.get()
                if tgt is None or event.other_actor.id != tgt.id:
                    raise RuntimeError(f"ego hit {event.other_actor.type_id}, not the target")
                impact_kmh = prev_speed_kmh  # the speed on the collision tick is already after the impact
                imp = event.normal_impulse
                node.pub_col.publish(Float32(data=float(math.hypot(imp.x, imp.y, imp.z))))
                collided = True

            prev_speed_kmh = speed_kmh
            if ego.get_velocity().length() < 0.1 and braking_began:
                hold_s += DT_S
                if hold_s > 1.0:
                    break

        # overlapping the target without a collision event means the geometry is wrong, not that the AEB worked
        if max_drift_m > MAX_DRIFT_M:  # the ego left its line, so the run no longer tests what it says it tests
            result = "INVALID"
        elif cfg.get("false_positive"):
            result = "FALSE_BRAKE" if max_brake > 0.0 else "COLLISION" if collided else "PASS"
        else:
            result = "COLLISION" if collided else "INVALID" if min_gap_m < -0.3 else "PASS"
        node.get_logger().info(f"{node.name}: {result} (t={t_s:.1f}s, min gap {min_gap_m:.2f} m, drift {max_drift_m:.2f} m)")
        node.pub_res.publish(String(data=f"{node.name}:{result}:min_gap={min_gap_m:.2f}"))
        summary_path = node.get_parameter("summary_path").value
        if summary_path:
            with open(summary_path, "w") as f:
                json.dump(dict(scenario=node.name, seed=node.get_parameter("seed").value, result=result, min_gap_m=min_gap_m,
                               max_brake=max_brake, max_drift_m=max_drift_m, duration_s=t_s, impact_kmh=impact_kmh if collided else 0.0,
                               first_brake_gap_m=first_brake[0] if first_brake else None,
                               first_brake_kmh=first_brake[1] if first_brake else None,
                               **{k: cfg[k] for k in ("ego_kmh", "target_kmh", "gap_m", "lateral_offset_m", "target_model", "brake_after_s", "false_positive") if k in cfg}), f)
        time.sleep(node.get_parameter("end_pause_s").value)  # the sync-mode window holds the last frame
    finally:
        for actor in reversed(actors):
            if isinstance(actor, carla.Sensor):
                actor.stop()
            actor.destroy()
        world.apply_settings(original_settings)


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioNode()
    run(node)
    node.destroy_node()
    rclpy.try_shutdown()
