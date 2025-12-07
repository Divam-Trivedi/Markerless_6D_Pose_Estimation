import cv2, numpy as np, os, pathlib, pyrealsense2 as rs, time, sys
import logging
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
sys.path.append("/home/hirolab/divam/FoundationPose/detectron2")
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

setup_logger()
logging.getLogger("detectron2").setLevel(logging.ERROR)

script_directory = pathlib.Path(__file__).parent.resolve()

# -----------------------------
# USER CONFIG
# -----------------------------
MODEL_NAME = "medical_objects_fruits"
TARGET_CLASSES = ["apple", "banana", "yogurt", "sb_cup", "ketchup", "tray", "orange"]
MODEL_PATH = f"/home/hirolab/divam/exprimental_FP/detectron2_models/{MODEL_NAME}.pth"
MODEL_INFO_FILE = f"/home/hirolab/divam/exprimental_FP/detectron2_models/model_info.txt"

MASK_ALPHA = 0.35
FONT = cv2.FONT_HERSHEY_SIMPLEX

class PosePublisherNode(Node):
    def __init__(self):
        super().__init__("pose_publisher")
        self.pub = self.create_publisher(String, "/object_poses", 10)
        self.img_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()

    def publish_results(self, results):
        clean_results = []
        for name, pose in results.items():
            clean_results = {
                "name": name,
                "xyz": pose["xyz"].tolist() if isinstance(pose["xyz"], np.ndarray) else pose["xyz"],
                "yaw_deg": float(pose["yaw_deg"])
            }
            msg = String()
            msg.data = json.dumps(clean_results)
            self.pub.publish(msg)
            self.get_logger().info(f"Published: {msg.data}")

    def publish_image(self, cv_image):
        """
        cv_image: numpy array, BGR
        """
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.img_pub.publish(img_msg)



# -------------------------------------------------------------
# Load model info (class names, dataset name, mesh names, etc.)
# -------------------------------------------------------------
def parse_model_info(model_name):
    with open(MODEL_INFO_FILE, "r") as f:
        lines = f.readlines()

    data = None
    for line in lines:
        if line.startswith(model_name):
            data = line.split(";")
            break

    dataset_name = data[0].strip()
    class_list = data[1].split("[")[1].split("]")[0]
    class_list = class_list.replace('"', "").replace(" ", "")
    class_list = class_list.split(",")

    meshes = data[2].split("[")[1].split("]")[0]
    meshes = meshes.replace('"', "").replace(" ", "").split(",")

    return class_list, dataset_name, meshes


# -------------------------------------------------------------
# Build Detectron2 Predictior
# -------------------------------------------------------------
def build_detector(dataset_name, num_classes, class_names, model_path):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    MetadataCatalog.get(dataset_name).thing_classes = class_names

    predictor = DefaultPredictor(cfg)
    return predictor


# -------------------------------------------------------------
# Setup Intel RealSense Camera
# -------------------------------------------------------------
def setup_intelrealsense_camera():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    MM_PER_UNIT = 1000.0 * depth_scale  # mm per unit (usually 1 mm)
    print(MM_PER_UNIT)

    return {
        "K": K,
        "MM_PER_UNIT": MM_PER_UNIT,
        "pipe": pipe,
        "align": align
    }


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

def capture_and_average_pose(predictor, intrinsics, class_names, target_classes, 
                             K, MM_PER_UNIT, align, pipe, num_frames=5):
    """
    Captures N frames and returns averaged XYZ + yaw for each object.
    """

    accumulated = {}   # object_class -> {"xyz": [], "yaw": []}

    for _ in range(num_frames):

        frames = align.process(pipe.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        depth_m = depth.astype(np.float32) * MM_PER_UNIT / 1000.0

        outputs = predictor(rgb)
        instances = outputs["instances"].to("cpu")

        if not instances.has("pred_masks"):
            continue

        masks = instances.pred_masks.numpy()
        pred_classes = instances.pred_classes.numpy()

        for i in range(len(pred_classes)):
            cname = class_names[pred_classes[i]]
            # if cname not in target_classes:
            #     continue

            mask = (masks[i].astype(np.uint8) * 255)

            # PCA → axis
            pca = compute_pca_axis(mask)
            if pca is None:
                continue

            cx, cy, dx, dy = pca
            yaw_deg = compute_yaw_from_pca(dx, dy)

            # Depth
            z = mask_median_depth(mask, depth_m)
            if z is None:
                continue

            cen = (cx, cy)
            xyz = deproject(cx, cy, z, K)

            if cname not in accumulated:
                accumulated[cname] = {"xyz": [], "yaw": []}

            accumulated[cname]["xyz"].append(xyz)
            accumulated[cname]["yaw"].append(yaw_deg)

        time.sleep(0.02)

    # ---- Compute averages ----
    results = {}

    for cname, info in accumulated.items():
        if len(info["xyz"]) == 0:
            continue

        mean_xyz = np.mean(np.vstack(info["xyz"]), axis=0)
        mean_yaw = float(np.mean(info["yaw"]))

        results[cname] = {
            "xyz": mean_xyz,
            "yaw_deg": mean_yaw
        }

    return results


# -------------------------------------------------------------
# MAIN REAL-TIME LOOP
# -------------------------------------------------------------
def main():
    rclpy.init()

    # Create the ROS node
    pose_node = PosePublisherNode()

    # Load detectron2 model info
    class_names, dataset_name, mesh_names = parse_model_info(MODEL_NAME)
    predictor = build_detector(dataset_name, len(class_names), class_names, MODEL_PATH)

    # Setup RealSense
    intrinsics = setup_intelrealsense_camera()
    pipe = intrinsics["pipe"]
    align = intrinsics["align"]
    K = intrinsics["K"]
    MM_PER_UNIT = intrinsics["MM_PER_UNIT"]

    cv2.namedWindow("3D Detection", cv2.WINDOW_NORMAL)
    print("Running real-time 3D detection. Press 'q' to quit.")

    while rclpy.ok():
        frames = align.process(pipe.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())  # raw units
        depth_m = depth.astype(np.float32) * MM_PER_UNIT / 1000.0  # convert to meters

        outputs = predictor(rgb)
        instances = outputs["instances"].to("cpu")

        vis = rgb.copy()
        summary_lines = []

        if instances.has("pred_masks"):
            masks = instances.pred_masks.numpy()
            pred_classes = instances.pred_classes.numpy()

            for i in range(len(pred_classes)):
                cname = class_names[pred_classes[i]]
                # if cname not in TARGET_CLASSES:
                #     continue

                mask = (masks[i].astype(np.uint8) * 255)

                # ---- Compute PCA axis ----
                pca = compute_pca_axis(mask)
                if pca is not None:
                    cx, cy, dx, dy = pca

                    # Scale axis length based on bounding box diagonal
                    axis_length = 80  # adjust as you like

                    x1 = int(cx - dx * axis_length)
                    y1 = int(cy - dy * axis_length)
                    x2 = int(cx + dx * axis_length)
                    y2 = int(cy + dy * axis_length)

                    cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)


                # depth estimate
                z = mask_median_depth(mask, depth_m)

                if z is not None:
                    cen = mask_centroid(mask)
                    if cen is not None:
                        u, v = cen
                        xyz = deproject(u, v, z, K)
                        xyz_text = f"({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:+.2f}) m"

                        # draw text near centroid
                        cv2.putText(vis, f"{cname} {xyz_text}",
                                    (int(u), int(v) - 10),
                                    FONT, 0.5, (255, 255, 255), 2)

                        summary_lines.append(f"{cname}: {xyz_text}")
                    else:
                        summary_lines.append(f"{cname}: no centroid")
                else:
                    summary_lines.append(f"{cname}: NO_DEPTH")

                # overlay mask
                color = random_color(cname)
                mask_rgb = np.zeros_like(vis)
                mask_rgb[mask > 0] = color
                vis = cv2.addWeighted(vis, 1.0, mask_rgb, MASK_ALPHA, 0)

                # outline for visibility
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, color, 2)

        # summary box
        x0, y0 = 10, 10
        for i, line in enumerate(summary_lines):
            cv2.putText(vis, line, (x0, y0 + 20 * i), FONT, 0.6, (0, 255, 255), 2)

        # ---- Draw global crosshair ----
        h, w = vis.shape[:2]
        cv2.line(vis, (w//2, 0), (w//2, h), (255, 255, 0), 1)  # vertical
        cv2.line(vis, (0, h//2), (w, h//2), (255, 255, 0), 1)  # horizontal

        pose_node.publish_image(vis)
        cv2.imshow("3D Detection", vis)
        

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('k'):
            print("\n--- Capturing 5 frames for averaged pose ---")
            results = capture_and_average_pose(
                        predictor, intrinsics, class_names, TARGET_CLASSES,
                        K, MM_PER_UNIT, align, pipe, num_frames=5
                    )
            print("Averaged Pose Results:")
            for cname, res in results.items():  
                xyz = res["xyz"]
                yaw = res["yaw_deg"]
                print(f"{cname}: XYZ = ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m, Yaw = {yaw:.1f} deg")

            pose_node.publish_results(results)

        rclpy.spin_once(pose_node, timeout_sec=0)

    pipe.stop()
    cv2.destroyAllWindows()
    pose_node.destroy_node()
    rclpy.shutdown()

def run_realtime_pose_estimator():
    main()

if __name__ == "__main__":
    main()


'''
import numpy as np

# --- INPUT ---
object_degree_rot = 30.0     # angle from your algorithm (deg)
R_cr = np.eye(3)             # camera->robot rotation (replace with your 3x3)

# --- HELPERS ---
def deg2rad(d): 
    return d * np.pi / 180.0

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def Rz(phi):
    c = np.cos(phi); s = np.sin(phi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def rotmat_to_quat(R):
    # quaternion (w,x,y,z)
    m = R
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        w = 0.25*S
        x = (m[2,1] - m[1,2]) / S
        y = (m[0,2] - m[2,0]) / S
        z = (m[1,0] - m[0,1]) / S
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])*2
            w = (m[2,1] - m[1,2]) / S
            x = 0.25*S
            y = (m[0,1] + m[1,0]) / S
            z = (m[0,2] + m[2,0]) / S
        elif m[1,1] > m[2,2]:
            S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])*2
            w = (m[0,2] - m[2,0]) / S
            x = (m[0,1] + m[1,0]) / S
            y = 0.25*S
            z = (m[1,2] + m[2,1]) / S
        else:
            S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])*2
            w = (m[1,0] - m[0,1]) / S
            x = (m[0,2] + m[2,0]) / S
            y = (m[1,2] + m[2,1]) / S
            z = 0.25*S
    return np.array([w,x,y,z])

# --- MAIN LOGIC ---

# 1) Convert object angle to camera-frame gripper rotation (add 90°, gripper perpendicular)
phi = deg2rad(object_degree_rot + 90.0)

# 2) Rotation in camera frame
R_cam = Rz(phi)

# 3) Convert to robot frame using your camera->robot rotation matrix
R_robot = R_cr @ R_cam

# 4A) Convert to quaternion (full orientation)
quat = rotmat_to_quat(R_robot)

# 4B) OR: extract yaw only (if robot uses a single rotation DOF)
yaw = wrap_to_pi(np.arctan2(R_robot[1,0], R_robot[0,0]))   # radians
yaw_deg = yaw * 180.0/np.pi

print("Quaternion (w,x,y,z):", quat)
print("Yaw (rad):", yaw, "Yaw (deg):", yaw_deg)
'''
