import cv2, numpy as np, os, pathlib, sys, time
import logging
from collections import defaultdict
import mediapipe
import pyrender
import trimesh
import datetime
from scipy.spatial.transform import Rotation as R

# persistence threshold (change to 3 or 5)
PERSISTENCE_THRESHOLD = 10

# IoU and depth thresholds for matching
IOU_MATCH_THRESHOLD = 0.35
DEPTH_MATCH_THRESHOLD_M = 0.06  # 6 cm tolerance

# random color for debug if needed:
YELLOW = (0, 255, 255)
GREEN  = (0, 255, 0)

# sys.path.append(f'/home/hirolab/divam/ros2_ws/src/pose_pkg/Markerless_6D_Pose_Estimation')
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
                    iteration=5
                )
                poses_per_frame[frame_idx][instance_id] = np.array(pose, dtype=np.float64)
            except Exception as e:
                print(f"[estimate_pose] Tracking failed for {instance_id} on frame {frame_idx}: {e}")
                continue

    return poses_per_frame, meshes_in_use

def save_combined_results_ros(frames_rgb, frames_depth, masks_per_frame,
                              poses_per_frame, meshes_in_use, K, MM_PER_UNIT, ROOT):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / f"output_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    annotated_frames = []
    results_lines = [f"Saved on {datetime.datetime.now().isoformat()}\n\n"]

    num_frames = len(frames_rgb)
    all_poses = {}  # obj_name -> list of poses
    for fi in range(num_frames):
        frame_rgb = frames_rgb[fi]
        frame_depth = frames_depth[fi].astype(np.float32) * MM_PER_UNIT / 1000.0
        masks = masks_per_frame[fi]
        poses = poses_per_frame.get(fi, {})

        results_lines.append(f"Frame {fi}:\n")

        mesh_img = frame_rgb.copy()
        axes_img = frame_rgb.copy()
        mask_img = frame_rgb.copy()

        for obj_name, pose in poses.items():
            mesh = meshes_in_use.get(obj_name)
            mask = masks.get(obj_name)
            if obj_name not in all_poses:
                all_poses[obj_name] = []


            if mesh is None or mask is None:
                continue

            h, w = frame_rgb.shape[:2]
            color = tuple(np.random.randint(0, 255, 3).tolist())

            mask_3ch = np.zeros_like(mask_img)
            mask_3ch[mask > 0] = color
            mask_img = cv2.addWeighted(mask_img, 1.0, mask_3ch, 0.4, 0)

            m = mesh.copy()
            m.apply_transform(pose)
            proj = (K @ m.vertices.T).T
            z = proj[:, 2:3]
            z[z == 0] = 1e-6
            proj[:, :2] /= z
            pts_2d = proj[:, :2].astype(np.int32)

            for tri in m.faces:
                cv2.polylines(mesh_img, [pts_2d[tri]], True, color, 2)

            # -90 deg rotation about X (object frame)
            T_rx = np.eye(4, dtype=np.float32)
            T_rx[:3, :3] = np.array([
                [1,  0,  0],
                [0,  0,  1],
                [0, -1,  0]
            ], dtype=np.float32)
            pose_new = pose @ T_rx
            axes_img = draw_xyz_axis(axes_img, ob_in_cam=pose_new, scale=0.1, K=K, thickness=3)

            depth_rendered = render_mesh_depth(mesh, pose, K, h, w)
            mesh_mask = depth_rendered > 0
            obj_mask = mask.astype(bool)

            valid = mesh_mask & obj_mask & (frame_depth > 0)
            valid &= np.abs(depth_rendered - frame_depth) < 0.02

            depth_error = np.zeros_like(depth_rendered)
            depth_error[valid] = np.abs(depth_rendered[valid] - frame_depth[valid])

            mean_err = depth_error[valid].mean() if np.any(valid) else np.inf
            median_err = np.median(depth_error[valid]) if np.any(valid) else np.inf
            inlier_ratio = np.mean(depth_error[valid] < 0.01) if np.any(valid) else 0.0
            iou = np.sum(mesh_mask & obj_mask) / np.sum(mesh_mask | obj_mask)

            pose_true = (
                mean_err * 1000 < 10 and
                median_err * 1000 < 5 and
                inlier_ratio > 0.7 and
                iou > 0.7
            )
            if pose_true:
                all_poses[obj_name].append(pose)

            depth_vis = np.clip(depth_error / 0.02 * 255, 0, 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            cv2.imwrite(str(out_root / f"{obj_name}_frame_{fi}_depth.png"), depth_vis)
            R_mat = pose[:3, :3]
            t_vec = pose[:3, 3]
            results_lines.append(
                f"  {obj_name}:\n"
                f"    mean_depth_error_mm: {mean_err*1000:.2f}\n"
                f"    median_depth_error_mm: {median_err*1000:.2f}\n"
                f"    inlier_ratio: {inlier_ratio:.3f}\n"
                f"    silhouette_iou: {iou:.3f}\n"
                f"    pose_true: {pose_true}\n"
            )
            results_lines.append("    pose:\n")
            results_lines.append("      R:\n")
            for row in R_mat:
                results_lines.append(f"        [{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}]\n")
            results_lines.append(
                f"      t: [{t_vec[0]:.6f} {t_vec[1]:.6f} {t_vec[2]:.6f}]\n"
            )
        
        cv2.imwrite(str(out_root / f"frame_{fi}_mesh.png"), mesh_img)
        cv2.imwrite(str(out_root / f"frame_{fi}_axes.png"), axes_img)
        cv2.imwrite(str(out_root / f"frame_{fi}_mask.png"), mask_img)
        list_images = [mesh_img, axes_img, mask_img]
        output_float = np.zeros_like(list_images[0], dtype=np.float32)
        for img in list_images:
            output_float = output_float + img.astype(np.float32) * (1.0 / len(list_images))

        final_average_blend = output_float.astype(np.uint8)
        annotated_frames.append(final_average_blend)
        results_lines.append("\n")

    results_lines.append("\nAveraged poses across all frames:\n")
    for obj_name, pose_list in all_poses.items():
        results_lines.append(f"  {obj_name}:\n")
        if len(pose_list) == 0:
            results_lines.append("    averaged_pose: None (no valid poses)\n")
            continue
        translations = np.stack([p[:3, 3] for p in pose_list], axis=0)
        t_avg = translations.mean(axis=0)
        rotations = R.from_matrix([p[:3, :3] for p in pose_list])
        R_avg = rotations.mean().as_matrix()

        results_lines.append("    averaged_pose:\n")
        results_lines.append("      R:\n")
        for row in R_avg:
            results_lines.append(f"        [{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}]\n")
        results_lines.append(f"      t: [{t_avg[0]:.6f} {t_avg[1]:.6f} {t_avg[2]:.6f}]\n")

    with open(out_root / "output_results.txt", "w") as f:
        f.writelines(results_lines)

    return annotated_frames


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


###
### Human Movement Detection
###
class HumanMovementDetector:
    def __init__(self, movement_threshold=0.5):
        self.mp_pose = mediapipe.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_torso_pos = None
        self.last_time = None
        self.moving = False
        self.velocity = 0.0
        self.human_present = False
        self.movement_threshold = movement_threshold

    def detect(self, rgb_frame, depth_frame):
        if rgb_frame is None or depth_frame is None:
            self.human_present = False
            self.moving = False
            self.velocity = 0.0
            return self.human_present, self.moving, self.velocity

        frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(frame_rgb)

        if not results.pose_landmarks:
            self.human_present = False
            self.moving = False
            self.velocity = 0.0
            self.last_torso_pos = None
            self.last_time = None
            return self.human_present, self.moving, self.velocity

        self.human_present = True
        lm = results.pose_landmarks.landmark

        torso_x = (
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x +
            lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x +
            lm[self.mp_pose.PoseLandmark.LEFT_HIP].x +
            lm[self.mp_pose.PoseLandmark.RIGHT_HIP].x
        ) / 4

        torso_y = (
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y +
            lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y +
            lm[self.mp_pose.PoseLandmark.LEFT_HIP].y +
            lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y
        ) / 4

        h, w = rgb_frame.shape[:2]
        cx = np.clip(int(torso_x * w), 0, w - 1)
        cy = np.clip(int(torso_y * h), 0, h - 1)

        depth_value = float(depth_frame[cy, cx])

        now = time.time()

        if self.last_torso_pos is not None and self.last_time is not None and depth_value > 0:
            dx = cx - self.last_torso_pos[0]
            dy = cy - self.last_torso_pos[1]
            pixel_dist = np.sqrt(dx * dx + dy * dy)
            metric_dist = pixel_dist * (depth_value / max(w, h))
            dt = now - self.last_time
            self.velocity = metric_dist / dt if dt > 0 else 0.0
            self.moving = self.velocity > self.movement_threshold
        else:
            self.moving = False
            self.velocity = 0.0

        self.last_torso_pos = (cx, cy)
        self.last_time = now

        return self.human_present, self.moving, self.velocity

def render_mesh_depth(mesh, pose_cv, K, H, W):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(cam, pose=np.eye(4))
    cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    pose_gl = cv_to_gl @ pose_cv
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_node, pose=pose_gl)
    r = pyrender.OffscreenRenderer(W, H)
    depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    r.delete()
    print("Rendered depth stats:",
      np.min(depth),
      np.max(depth),
      np.count_nonzero(depth))
    return depth


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
