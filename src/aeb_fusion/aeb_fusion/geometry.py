import numpy as np


def project_to_image(pts_lidar: np.ndarray, cam_in_lidar_m: np.ndarray,
                     K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_cam = pts_lidar - cam_in_lidar_m
    # ros body frame (x fwd, y left, z up) -> camera optical frame (x right, y down, z fwd)
    x_opt, y_opt, z_opt = -pts_cam[:, 1], -pts_cam[:, 2], pts_cam[:, 0]
    in_front = z_opt > 0.5
    z_safe = np.where(in_front, z_opt, 1.0)
    u_px = K[0, 0] * x_opt / z_safe + K[0, 2]
    v_px = K[1, 1] * y_opt / z_safe + K[1, 2]
    return u_px, v_px, in_front


def nearest_cluster_xyz(pts_lidar: np.ndarray, u_px: np.ndarray, v_px: np.ndarray, in_front: np.ndarray,
                        box_px: tuple[float, float, float, float], shrink: float, min_points: int,
                        depth_window_m: float = 1.5) -> np.ndarray | None:
    cx_px, cy_px, w_px, h_px = box_px
    in_box = in_front & (np.abs(u_px - cx_px) < w_px * shrink / 2) & (np.abs(v_px - cy_px) < h_px * shrink / 2)
    if in_box.sum() < min_points:
        return None
    pts_box = pts_lidar[in_box]
    # a distant box is only a few points wide, and road or background hits behind the car would skew a plain median
    return np.median(pts_box[pts_box[:, 0] < pts_box[:, 0].min() + depth_window_m], axis=0)
