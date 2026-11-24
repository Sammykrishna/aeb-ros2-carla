import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler, Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

BAG_TOPICS = ["/ego/camera/image", "/ego/camera/camera_info", "/ego/lidar/points", "/ego/odom",
              "/aeb/detections_2d", "/aeb/objects_3d", "/aeb/brake_cmd", "/aeb/state", "/aeb/ttc",
              "/aeb/required_decel",
              "/ground_truth/gap", "/ground_truth/lateral_offset", "/ground_truth/collision", "/scenario/result"]


def generate_launch_description():
    params = os.path.join(get_package_share_directory("aeb_bringup"), "config", "aeb_params.yaml")
    scenario = LaunchConfiguration("scenario")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    controller = LifecycleNode(package="aeb_controller", executable="aeb_controller_node",
                               name="aeb_controller", namespace="", parameters=[params], output="screen")
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(controller),
        transition_id=Transition.TRANSITION_CONFIGURE))
    activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=controller, goal_state="inactive",
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(controller),
            transition_id=Transition.TRANSITION_ACTIVATE))]))

    detector = Node(package="aeb_detector", executable="yolo_detector_node", parameters=[params], output="screen")
    fusion = Node(package="aeb_fusion", executable="fusion_node", parameters=[params], output="screen")
    scenario_node = Node(package="aeb_bringup", executable="carla_scenario_node", output="screen",
                         parameters=[{"scenario": scenario,
                                      "spectator": ParameterValue(LaunchConfiguration("spectator"), value_type=bool),
                                      "end_pause_s": ParameterValue(LaunchConfiguration("end_pause_s"), value_type=float),
                                      "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
                                      "summary_path": ParameterValue(LaunchConfiguration("summary_path"), value_type=str)}])
    bag = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-s", "mcap", "-o",
             ["bags/", scenario, f"_{stamp}"], *BAG_TOPICS],
        condition=IfCondition(LaunchConfiguration("record")), output="screen")
    stop_all = RegisterEventHandler(OnProcessExit(target_action=scenario_node, on_exit=[Shutdown()]))

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="CCRs_50"),
        DeclareLaunchArgument("record", default_value="false"),
        DeclareLaunchArgument("spectator", default_value="false"),
        DeclareLaunchArgument("end_pause_s", default_value="0.0"),
        DeclareLaunchArgument("seed", default_value="-1"),
        DeclareLaunchArgument("summary_path", default_value=""),
        bag, controller, activate, detector, fusion, scenario_node, stop_all, configure,
    ])
