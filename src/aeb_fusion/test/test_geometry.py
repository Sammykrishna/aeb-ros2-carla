import numpy as np

from aeb_fusion.geometry import nearest_cluster_xyz, project_to_image

K = np.array([[480.0, 0.0, 480.0], [0.0, 480.0, 270.0], [0.0, 0.0, 1.0]])
CAM_IN_LIDAR_M = np.array([1.5, 0.0, -0.2])


def test_point_ahead_projects_to_image_centre():
    pts_lidar = np.array([[21.5, 0.0, -0.2]])
    u_px, v_px, in_front = project_to_image(pts_lidar, CAM_IN_LIDAR_M, K)
    assert in_front[0] and np.allclose([u_px[0], v_px[0]], [480.0, 270.0])


def test_point_to_the_left_lands_left_of_centre():
    u_px, _, _ = project_to_image(np.array([[21.5, 2.0, -0.2]]), CAM_IN_LIDAR_M, K)
    assert np.isclose(u_px[0], 432.0)


def test_point_behind_camera_is_not_in_front():
    _, _, in_front = project_to_image(np.array([[-5.0, 0.0, 0.0]]), CAM_IN_LIDAR_M, K)
    assert not in_front[0]


BOX_PX = (480.0, 270.0, 200.0, 200.0)


def fuse(pts_lidar: np.ndarray, shrink: float = 1.0, min_points: int = 2) -> np.ndarray | None:
    u_px, v_px, in_front = project_to_image(pts_lidar, CAM_IN_LIDAR_M, K)
    return nearest_cluster_xyz(pts_lidar, u_px, v_px, in_front, BOX_PX, shrink, min_points)


def test_range_is_median_of_points_within_depth_window_of_the_nearest_and_ignores_points_outside_box():
    pts_lidar = np.array([[20.0, 0.0, 0.0], [22.0, 0.0, 0.0], [21.0, 0.0, 0.0], [60.0, 8.0, 0.0]])
    assert np.isclose(fuse(pts_lidar)[0], 20.5)


def test_background_behind_the_target_does_not_skew_the_range():
    pts_lidar = np.array([[20.0, 0.0, 0.0], [20.3, 0.1, 0.0], [45.0, 0.0, 0.0], [46.0, 0.0, 0.0], [47.0, 0.0, 0.0]])
    assert fuse(pts_lidar)[0] < 20.5


def test_too_few_points_returns_none():
    assert fuse(np.array([[20.0, 0.0, 0.0]])) is None


def test_shrink_excludes_points_near_the_box_edge():
    edge_px = 480.0 + 90.0
    x_lidar_m = 20.0
    y_lidar_m = -(edge_px - K[0, 2]) * x_lidar_m / K[0, 0]
    pts_lidar = np.array([[x_lidar_m + 1.5, y_lidar_m, -0.2]] * 3)
    assert fuse(pts_lidar, shrink=1.0) is not None
    assert fuse(pts_lidar, shrink=0.5) is None
