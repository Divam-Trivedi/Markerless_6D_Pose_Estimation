import cv2, numpy as np, os, pathlib, sys, json
import logging
from collections import defaultdict

# persistence threshold (change to 3 or 5)
PERSISTENCE_THRESHOLD = 10

# IoU and depth thresholds for matching
IOU_MATCH_THRESHOLD = 0.35
DEPTH_MATCH_THRESHOLD_M = 0.06  # 6 cm tolerance

# random color for debug if needed:
YELLOW = (0, 255, 255)
GREEN  = (0, 255, 0)

# sys.path.append(f'/home/hirolab/divam/FoundationPose/FoundationPose')
from estimater import *
from datareader import *
from FP_Utils import *

script_directory = pathlib.Path(__file__).parent.resolve()

sys.path.insert(0, os.path.abspath('/home/hirolab/divam/FoundationPose/detectron2'))

import logging
logging.getLogger("detectron2").setLevel(logging.ERROR)

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

def parse_model_info(model_name):
    f = open(f"{script_directory}/detectron2_models/model_info.txt", 'r')
    lines = f.readlines()
    for line in lines:
        if line.startswith(model_name):
            data = line.split(';')
    dataset_name = str(data[0].replace(" ", ""))
    all_classes = list(data[1].split('[')[1].split(']')[0].replace('"', '').replace(" ", "").split(','))
    mesh_name = list(data[2].split('[')[1].split(']')[0].replace('"', '').replace(" ", "").split(','))
    return all_classes, dataset_name, mesh_name

def build_detector(dataset_name, num_classes, class_names, model_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    MetadataCatalog.get(dataset_name).thing_classes = class_names
    predictor = DefaultPredictor(cfg)
    return predictor

def setup_estimator_for_object(obj_name, mesh_names, all_classes, K, ROOT):
    # Lookup mesh name
    mesh_idx = all_classes.index(obj_name)
    mesh_name = mesh_names[mesh_idx]

    # Use original mesh folder directly, do NOT copy or modify anything
    mesh_file = pathlib.Path(f"{script_directory}/Meshes") / mesh_name / "textured_simple.obj"
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file does not exist: {mesh_file}")

    mesh = trimesh.load(mesh_file)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    # Create debug folder inside output folder (only for logging)
    debug_dir = ROOT / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir)
    )
    return est, mesh

def estimate_pose(frames_rgb, frames_depth, masks_per_frame, detected_objects,
                  intrinsics, mesh_names, all_classes, ROOT, instance_to_class):
    """
    Fixed version:
    - Only objects present in frame 0 are used.
    - register() is called exactly once.
    - track_one() is called only after register() succeeded.
    """

    K = intrinsics["K"]
    MM_PER_UNIT = intrinsics.get("MM_PER_UNIT", 1.0)
    MM_PER_UNIT = 1.0000000474974513

    poses_per_frame = defaultdict(dict)
    meshes_in_use = {}
    estimators = {}

    # ---------- 1. First: determine which objects exist in frame 0 ----------
    first_frame_masks = masks_per_frame[0]
    valid_objects = set()

    for instance_id in detected_objects:
        if instance_id in first_frame_masks:
            valid_objects.add(instance_id)
        else:
            print(f"[estimate_pose] Skipping {instance_id} — not present in frame 0 (required for registration).")

    # ---------- 2. Initialize estimators only for valid objects ----------
    for instance_id in valid_objects:
        class_name = instance_to_class[instance_id]
        est, mesh = setup_estimator_for_object(
            class_name, mesh_names, all_classes, K, ROOT
        )
        estimators[instance_id] = est
        meshes_in_use[instance_id] = mesh

    # ---------- 3. Process each frame ----------
    for frame_idx, (rgb_frame, depth_frame, mask_dict) in enumerate(
        zip(frames_rgb, frames_depth, masks_per_frame)
    ):
        depth_m = depth_frame.astype(np.float32) * MM_PER_UNIT / 1000.0

        for instance_id in valid_objects:
            est = estimators[instance_id]

            # If object not detected in this frame → skip tracking for this frame
            if instance_id not in mask_dict:
                continue

            mask = mask_dict[instance_id].astype(bool)

            # ---------- Frame 0: REGISTER ----------
            if frame_idx == 0:
                try:
                    pose = est.register(
                        K=K,
                        rgb=rgb_frame,
                        depth=depth_m,
                        ob_mask=mask,
                        iteration=5
                    )
                    ## POSE CORRECTION
                    # theta = np.deg2rad(-90)
                    # Rz = np.array([
                    #     [np.cos(theta), -np.sin(theta), 0],
                    #     [np.sin(theta),  np.cos(theta), 0],
                    #     [0, 0, 1]
                    # ])
                    # R_old = pose[:3, :3]
                    # t_old = pose[:3, 3]
                    # # Rotate around LOCAL Z axis → post-multiply
                    # R_new = R_old @ Rz
                    # T_rotated = np.eye(4)
                    # T_rotated[:3, :3] = R_new
                    # T_rotated[:3, 3]  = t_old
                    # pose = T_rotated.copy()
                    poses_per_frame[0][instance_id] = np.array(pose, dtype=np.float64)
                    print(f"[estimate_pose] Registered {instance_id} successfully.")
                except Exception as e:
                    print(f"[estimate_pose] Registration failed for {instance_id}: {e}")
                continue

            # ---------- Frame >0: TRACK ----------
            try:
                pose = est.track_one(
                    rgb=rgb_frame,
                    depth=depth_m,
                    K=K,
                    iteration=2
                )
                ## POSE CORRECTION
                # theta = np.deg2rad(-90)
                # Rz = np.array([
                #     [np.cos(theta), -np.sin(theta), 0],
                #     [np.sin(theta),  np.cos(theta), 0],
                #     [0, 0, 1]
                # ])
                # R_old = pose[:3, :3]
                # t_old = pose[:3, 3]
                # # Rotate around LOCAL Z axis → post-multiply
                # R_new = R_old @ Rz
                # T_rotated = np.eye(4)
                # T_rotated[:3, :3] = R_new
                # T_rotated[:3, 3]  = t_old
                # pose = T_rotated.copy()

                poses_per_frame[frame_idx][instance_id] = np.array(pose, dtype=np.float64)
            except Exception as e:
                print(f"[estimate_pose] Tracking failed for {instance_id} on frame {frame_idx}: {e}")
                continue

    return poses_per_frame, meshes_in_use


def save_combined_results_ros(frames_rgb, masks_per_frame, poses_per_frame, meshes_in_use, K, ROOT):
    """
    ROS2 version: visualize in memory only (publish via topic), no GUI or image saving.
    Still writes output_results.txt for poses.
    Returns list of annotated images for potential publishing.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / f"output_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    num_frames = len(frames_rgb)
    annotated_frames = []  # will store annotated RGB for ROS publishing

    for i in range(num_frames):
        frame = frames_rgb[i].copy()
        mask_dict = masks_per_frame[i]
        vis = frame.copy()

        # Overlay masks (light)
        for obj_name, mask in mask_dict.items():
            color = tuple(np.random.randint(0, 255, 3).tolist())
            mask_3ch = np.zeros_like(vis)
            mask_3ch[mask > 0] = color
            vis = cv2.addWeighted(vis, 1.0, mask_3ch, 0.35, 0)

        # Draw meshes + axes (optional)
        for obj_name, pose in poses_per_frame.get(i, {}).items():
            mesh = meshes_in_use.get(obj_name, None)
            if mesh is None:
                continue

            color = tuple(np.random.randint(0, 255, 3).tolist())

            # draw XYZ axes at object pose
            vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.1, K=K, thickness=3)

            # project mesh with object color
            m = mesh.copy()
            m.apply_transform(pose)
            projected = (K @ m.vertices.T).T
            z = projected[:, 2:3]
            z[z == 0] = 1e-6
            projected[:, :2] /= z
            pts_2d = projected[:, :2].astype(np.int32)

            for tri in m.faces:
                pts = pts_2d[tri]
                cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            # Label object
            mask = mask_dict.get(obj_name, None)
            if mask is not None:
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    cv2.putText(vis, obj_name, (xs[0], ys[0]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(vis, obj_name, (10, 40 + 25 * list(poses_per_frame[i]).index(obj_name)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Add frame index
        cv2.putText(vis, f"Frame {i}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        
        cv2.imwrite(f"{out_root}/frame_{i}.png", vis)
        annotated_frames.append(vis)
        print(f"Saved frame {i}")

    # Write output_results.txt (pose info)
    results_path = out_root / "output_results.txt"
    lines = []
    lines.append(f"Saved on {datetime.datetime.now().isoformat()}\n\n")

    poses_by_object = defaultdict(list)

    for fi in range(num_frames):
        lines.append(f"Frame {fi}:\n")
        pose_dict = poses_per_frame.get(fi, {})
        if len(pose_dict) == 0:
            lines.append("  (no detected objects)\n\n")
            continue

        for obj_name, pose in pose_dict.items():
            pose = np.array(pose, dtype=np.float64)
            R = pose[:3, :3]
            t = pose[:3, 3]
            q = rotmat_to_quat(R)
            rpy = rotmat_to_rpy(R)

            lines.append(f"  {obj_name}:\n")
            lines.append(f"    translation: {t.tolist()}\n")
            lines.append(f"    rotation_matrix:\n")
            for row in R.tolist():
                lines.append(f"      {row}\n")
            lines.append(f"    rpy_degrees (roll,pitch,yaw): {rpy.tolist()}\n")
            lines.append(f"    quaternion (x,y,z,w): {q.tolist()}\n")

            poses_by_object[obj_name].append((fi, pose))
        lines.append("\n")

    # Averaged poses
    lines.append("Final Averaged Poses:\n")
    for obj_name, plist in poses_by_object.items():
        translations = np.array([p[1][:3,3] for p in plist])
        t_mean = np.mean(translations, axis=0)
        quats = np.array([rotmat_to_quat(p[1][:3,:3]) for p in plist])
        q_avg = average_quaternions(quats)
        R_avg = quat_to_rotmat(q_avg)
        rpy_avg = rotmat_to_rpy(R_avg)

        pose_avg = np.eye(4)
        pose_avg[:3,:3] = R_avg
        pose_avg[:3,3] = t_mean

        lines.append(f"  {obj_name}:\n")
        lines.append(f"    mean_translation: {t_mean.tolist()}\n")
        lines.append(f"    avg_rotation_matrix:\n")
        for row in R_avg.tolist():
            lines.append(f"      {row}\n")
        lines.append(f"    avg_rpy_degrees (roll,pitch,yaw): {rpy_avg.tolist()}\n")
        lines.append(f"    avg_quaternion (x,y,z,w): {q_avg.tolist()}\n")
        lines.append(f"    avg_pose_4x4:\n")
        for row in pose_avg.tolist():
            lines.append(f"      {row}\n")
        lines.append("\n")

    with open(results_path, 'w') as fh:
        fh.writelines(lines)

    print(f"Wrote pose results summary to {results_path}")

    return annotated_frames  # return images for ROS publishing

# -------------------------------------------------------------
# Mask utility: centroid
# -------------------------------------------------------------
def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    u = float(np.mean(xs))
    v = float(np.mean(ys))
    return (u, v)


# -------------------------------------------------------------
# Mask utility: median depth in meters
# -------------------------------------------------------------
def mask_median_depth(mask, depth_meters):
    m = mask.astype(bool)
    if m.sum() == 0:
        return None

    d = depth_meters[m]
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return None

    return float(np.median(d[valid]))


# -------------------------------------------------------------
# Deprojection from pixel to 3D (camera frame)
# -------------------------------------------------------------
def deproject(u, v, z, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return np.array([X, Y, Z], dtype=float)


# -------------------------------------------------------------
# Color generation per class name
# -------------------------------------------------------------
def random_color(name):
    np.random.seed(abs(hash(name)) % (2**32))
    return tuple(int(x) for x in np.random.randint(50, 255, size=3))

def compute_pca_axis(mask):
    """
    Compute principal axis of a binary mask using PCA.
    Returns (cen_x, cen_y, dir_x, dir_y) which defines a 2D line:
        centroid = (cen_x, cen_y)
        direction = (dir_x, dir_y) normalized
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    # Subtract mean
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean

    # PCA (covariance eigenvectors)
    cov = np.cov(pts_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)

    # Principal axis = eigenvector with largest eigenvalue
    idx = np.argmax(eigvals)
    axis = eigvecs[:, idx]
    axis = axis / np.linalg.norm(axis)

    return float(mean[0]), float(mean[1]), float(axis[0]), float(axis[1])

def compute_yaw_from_pca(dx, dy):
    """
    Computes yaw angle (rotation in image plane) from the PCA direction vector.
    Returns angle in degrees, where 0 deg is vertical axis.
    """
    angle_rad = np.arctan2(dx, -dy)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


### MATHS ####
def rotmat_to_quat(R):
    # Returns quaternion [x, y, z, w] from 3x3 rotation matrix
    # Uses stable method
    m = R
    t = m[0,0] + m[1,1] + m[2,2]
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    # normalize
    q = q / np.linalg.norm(q)
    return q

def quat_to_rotmat(q):
    # q = [x, y, z, w]
    x, y, z, w = q
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R

def rotmat_to_rpy(R):
    # returns roll, pitch, yaw in degrees (ZYX convention -> yaw-pitch-roll)
    sy = -R[2,0]
    cy = np.sqrt(1 - sy*sy)
    singular = cy < 1e-6
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arcsin(sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        # Gimbal lock
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arcsin(sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])

def average_quaternions(quaternions, weights=None):
    # Markley et al. method: build symmetric accumulator and take principal eigenvector
    # quaternions: Nx4 (x,y,z,w)
    Q = np.array(quaternions, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q[None, :]
    if weights is None:
        weights = np.ones((Q.shape[0],), dtype=np.float64)
    W = np.array(weights, dtype=np.float64).reshape(-1)
    # normalize weights
    W = W / np.sum(W)
    A = np.zeros((4,4), dtype=np.float64)
    for q, w in zip(Q, W):
        q = q.reshape(4,1)
        A += w * (q @ q.T)
    # compute principal eigenvector
    vals, vecs = np.linalg.eigh(A)
    q_avg = vecs[:, np.argmax(vals)]
    # ensure scalar (w) positive for consistency
    if q_avg[3] < 0:
        q_avg = -q_avg
    return q_avg  # 4-vector [x,y,z,w]
