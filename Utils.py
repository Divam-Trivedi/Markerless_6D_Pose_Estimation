import cv2, numpy as np, os, pathlib, pyrealsense2 as rs, time, sys
import os, sys, time, trimesh, logging
from ultralytics import YOLO

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

sys.path.insert(0, os.path.abspath('./detectron2'))

import logging
logging.getLogger("detectron2").setLevel(logging.ERROR)

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

def setup_intelrealsense_camera():
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile   = pipe.start(cfg)
    align     = rs.align(rs.stream.color)

    intr = profile.get_stream(rs.stream.color)\
                .as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0,        intr.ppx],
                [0,       intr.fy,  intr.ppy],
                [0,       0,        1      ]], dtype=np.float32)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() # e.g. 0.001
    MM_PER_UNIT = 1000.0 * depth_scale
    intrinsics = {
        "K": K,
        "MM_PER_UNIT": MM_PER_UNIT,
        "align": align,
        "pipe": pipe,
    }
    return intrinsics

def get_mask(image, model):
    res = model.predict(source=image, save=False, save_txt=False, conf=0.5, iou=0.45)
    mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
    isolated_image = np.zeros_like(image)
    for r in res:
        for c in r:
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(mask_image, [contour], -1, 255, cv2.FILLED)
            mask3ch = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
            isolated_image = cv2.bitwise_and(image, mask3ch)
    return mask_image, isolated_image

    
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

def capture_frames_in_memory(align, pipe, num_frames, predictor, dataset_name, target_classes, intrinsics):
    frames_rgb, frames_depth, masks_per_frame = [], [], []

    # keep persistence state across frames
    persistent_objects = {}
    _next_instance_idx = defaultdict(int)
    frame_idx = 0
    captured = 0

    # explicit intrinsics unpacking (safer & clearer)
    K = intrinsics["K"]
    MM_PER_UNIT = intrinsics["MM_PER_UNIT"]
    # align and pipe are already passed separately to this function, but keep local copies if needed:
    # align = intrinsics["align"]
    # pipe = intrinsics["pipe"]

    while captured < num_frames:
        frames = align.process(pipe.wait_for_frames())
        rgb = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())

        # Detect objects
        outputs = predictor(rgb)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_masks = instances.pred_masks.cpu().numpy()
        metadata = MetadataCatalog.get(dataset_name)
        class_names = metadata.thing_classes

        # build current detections list
        curr_instances = []
        for i in range(len(pred_classes)):
            cname = class_names[pred_classes[i]]
            if cname not in target_classes:
                continue
            mask = (pred_masks[i].astype(np.uint8) * 255)
            cen = mask_centroid(mask)
            try:
                # convert depth to meters for mask_median_depth
                median_depth = mask_median_depth(mask, depth.astype(np.float32) * MM_PER_UNIT / 1000.0)
            except Exception:
                median_depth = None
            curr_instances.append({'class': cname, 'mask': mask, 'centroid': cen, 'depth': median_depth})

        # build prev_instances list from persistent_objects (fixed order)
        prev_instances = []
        prev_ids = []
        for pid, info in persistent_objects.items():
            prev_instances.append({'id': pid, 'class': info['class'], 'mask': info['last_mask'], 'depth': info.get('last_depth')})
            prev_ids.append(pid)

        # match
        assigns, unmatched_prev, unmatched_curr = match_detections(prev_instances, curr_instances)

        # prepare mapping from curr index -> instance_id
        curr_to_id = {}

        # apply assignments
        for pi, ci in assigns:
            pid = prev_ids[pi]
            curr_to_id[ci] = pid
            # update persistent_objects and increment consec_count
            persistent_objects[pid]['last_mask'] = curr_instances[ci]['mask']
            persistent_objects[pid]['last_depth'] = curr_instances[ci]['depth']
            persistent_objects[pid]['consec_count'] = persistent_objects[pid].get('consec_count', 0) + 1
            persistent_objects[pid]['last_seen_frame'] = frame_idx

        # handle unmatched current detections -> create new instance ids
        for ci in unmatched_curr:
            cname = curr_instances[ci]['class']
            idx = _next_instance_idx[cname]
            pid = f"{cname}_{idx}"
            _next_instance_idx[cname] += 1
            curr_to_id[ci] = pid
            persistent_objects[pid] = {
                'class': cname,
                'last_mask': curr_instances[ci]['mask'],
                'last_depth': curr_instances[ci]['depth'],
                'consec_count': 1,
                'last_seen_frame': frame_idx,
                'estimator': None
            }

        # handle unmatched_prev -> reduce consec_count or mark not-seen (avoid strict reset)
        for pi in unmatched_prev:
            pid = prev_ids[pi]
            # Option A (gentle): decrement consec_count but keep object alive for a few frames
            persistent_objects[pid]['consec_count'] = max(0, persistent_objects[pid].get('consec_count', 0) - 1)
            # leave last_mask/last_depth as-is; optionally do not delete immediately
            # Option B (stricter): uncomment to enforce immediate reset:
            # persistent_objects[pid]['consec_count'] = 0

        # build frame_instance_masks: dict instance_id -> mask
        frame_instance_masks = {}
        for ci, det in enumerate(curr_instances):
            iid = curr_to_id.get(ci)
            if iid is None:
                continue
            frame_instance_masks[iid] = det['mask']

        # Visualize overlays using persistence threshold
        vis = rgb.copy()
        for iid, mask in frame_instance_masks.items():
            info = persistent_objects[iid]
            color = GREEN if info['consec_count'] >= PERSISTENCE_THRESHOLD else YELLOW
            mask_3ch = np.zeros_like(vis)
            mask_3ch[mask > 0] = color
            vis = cv2.addWeighted(vis, 1.0, mask_3ch, 0.35, 0)
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                cv2.putText(vis, f"{iid} ({info['consec_count']})", (xs[0], ys[0]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # store frames / masks / increment counters
        frames_rgb.append(rgb)
        frames_depth.append(depth)
        masks_per_frame.append(frame_instance_masks)
        captured += 1
        frame_idx += 1

        print(f"Captured frame {captured}/{num_frames}")

    return frames_rgb, frames_depth, masks_per_frame, persistent_objects

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

def capture_images(intrinsics, predictor, dataset_name, target_classes, MAX_FRAMES=5):
    K, MM_PER_UNIT, align, pipe = list(intrinsics.values())
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    print('Press "K" to capture frames for pose estimation. Press "Q" to quit.')
    while True:
        frames = align.process(pipe.wait_for_frames())
        rgb = np.asanyarray(frames.get_color_frame().get_data())

        # detect all target objects
        outputs = predictor(rgb)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        metadata = MetadataCatalog.get(dataset_name)
        class_names = metadata.thing_classes

        vis = rgb.copy()
        for i, class_id in enumerate(pred_classes):
            cname = class_names[class_id]
            if cname in target_classes:
                mask = instances.pred_masks[i].cpu().numpy().astype(np.uint8)*255
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                vis = cv2.addWeighted(vis, 1.0, mask_3ch, 0.3, 0)
                y, x = np.where(mask>0)
                if len(y)>0:
                    cv2.putText(vis, cname, (x[0], y[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("RGB", vis)
        key = cv2.waitKey(1)
        if key == ord('k'):
            num_frames = MAX_FRAMES
            frame_locked = cv2.GaussianBlur(rgb.copy(), (21,21), 0)
            cv2.putText(frame_locked, "Camera is Locked in Pose", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            cv2.imshow("RGB", frame_locked)
            cv2.waitKey(1)  # short pause to display
            break
        elif key == ord('q'):
            pipe.stop()
            cv2.destroyAllWindows()
            exit()
    
    start_time = time.time()
    frames_rgb, frames_depth, masks_per_frame, persistent_objects = capture_frames_in_memory(align, pipe, num_frames, predictor, dataset_name, target_classes, intrinsics)
    pipe.stop()

    # collect all detected objects across frames
    detected_objects = set()
    for mask_dict in masks_per_frame:
        detected_objects.update(mask_dict.keys())
    print("Detected objects to process:", detected_objects)

    return frames_rgb, frames_depth, masks_per_frame, detected_objects, start_time, persistent_objects

def estimate_pose(frames_rgb, frames_depth, masks_per_frame, detected_objects, intrinsics, mesh_names, all_classes, ROOT, instance_to_class):
    K, MM_PER_UNIT, align, pipe = list(intrinsics.values())
    poses_per_frame = defaultdict(dict)   # poses_per_frame[frame_idx][obj_name] = 4x4 np.array
    meshes_in_use = {}  # meshes_in_use[obj_name] = trimesh object (loaded once)
    ests = {}  # optional cache of estimators per object
    # Run pose estimation per object
    # inside estimate_pose: iterate instance ids
    for instance_id in detected_objects:
        class_name = instance_to_class[instance_id]  # e.g. 'apple'
        est, mesh = setup_estimator_for_object(class_name, mesh_names, all_classes, K, ROOT)
        # store by instance id:
        meshes_in_use[instance_id] = mesh
        ests[instance_id] = est
        # then iterate frames and look for mask under this instance_id:
        for i, (rgb_frame, depth_frame, mask_dict) in enumerate(zip(frames_rgb, frames_depth, masks_per_frame)):
            if instance_id not in mask_dict:
                continue
            mask = mask_dict[instance_id]
            depth_m = depth_frame.astype(np.float32) * MM_PER_UNIT / 1000.0
            pose = est.register(K=K, rgb=rgb_frame, depth=depth_m, ob_mask=mask.astype(bool), iteration=5) \
                    if i==0 else est.track_one(rgb=rgb_frame, depth=depth_m, K=K, iteration=2)
            poses_per_frame[i][instance_id] = np.array(pose, dtype=np.float64)
            print("The pose of", class_name, "is", poses_per_frame[i][instance_id])

    end_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_combined_results(frames_rgb, masks_per_frame, poses_per_frame, meshes_in_use, K, ROOT, timestamp)
    first_frame_path = ROOT / f"output_{timestamp}" / "000.png"
    first_frame_annotated = cv2.imread(str(first_frame_path))
    cv2.imshow("RGB", first_frame_annotated)
    cv2.waitKey(0)  # wait until user presses a key

    cv2.destroyAllWindows()
    return end_time

def save_combined_results(frames_rgb, masks_per_frame, poses_per_frame, meshes_in_use, K, ROOT, timestamp):
    """
    Create ROOT/output_<timestamp>/ with:
      - N annotated images (all objects per frame: masks + colored meshes + axes)
      - output_results.txt with per-frame poses + final averaged pose
    """
    out_root = ROOT / f"output_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    num_frames = len(frames_rgb)

    for i in range(num_frames):
        frame = frames_rgb[i].copy()
        mask_dict = masks_per_frame[i]
        vis = frame.copy()

        # 1️⃣ Overlay masks (light)
        for obj_name, mask in mask_dict.items():
            color = tuple(np.random.randint(0, 255, 3).tolist())
            mask_3ch = np.zeros_like(vis)
            mask_3ch[mask > 0] = color  # color mask
            vis = cv2.addWeighted(vis, 1.0, mask_3ch, 0.35, 0)

        # 2️⃣ Draw meshes + axes for each object with a pose in this frame
        for obj_name, pose in poses_per_frame.get(i, {}).items():
            mesh = meshes_in_use.get(obj_name, None)
            if mesh is None:
                continue
            # assign color if not yet assigned
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # draw XYZ axes at object pose (standard RGB)
            theta = np.deg2rad(90)
            T_blender = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta),  np.cos(theta)]
            ])
            T_corr = T_blender
            R_corr = pose[:3,:3] @ T_corr
            pose_corr = np.eye(4)
            pose_corr[:3,:3] = R_corr
            pose_corr[:3,3] = pose[:3,3]
            vis = draw_xyz_axis(vis, ob_in_cam=pose_corr, scale=0.1, K=K, thickness=3)

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

            # label object
            mask = mask_dict.get(obj_name, None)
            if mask is not None:
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    cv2.putText(vis, obj_name, (xs[0], ys[0]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(vis, obj_name, (10, 40 + 25 * list(poses_per_frame[i]).index(obj_name)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 3️⃣ Add frame index (last so always visible)
        cv2.putText(vis, f"Frame {i}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # 4️⃣ Save final annotated image
        save_path = out_root / f"{i:03}.png"
        cv2.imwrite(str(save_path), vis)  # Save in BGR
        print(f"Saved combined visualization {save_path}")

    # 📄 ---- Write output_results.txt (same as before but with earlier fixes applied) ---- #
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
            for row in np.array(R).tolist():
                lines.append(f"      {row}\n")
            lines.append(f"    rpy_degrees (roll,pitch,yaw): {rpy.tolist()}\n")
            lines.append(f"    quaternion (x,y,z,w): {q.tolist()}\n")

            poses_by_object[obj_name].append((fi, pose))
        lines.append("\n")

    # Averaged pose section (unchanged logic — already correct)
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
        for row in np.array(R_avg).tolist():
            lines.append(f"      {row}\n")
        lines.append(f"    avg_rpy_degrees (roll,pitch,yaw): {rpy_avg.tolist()}\n")
        lines.append(f"    avg_quaternion (x,y,z,w): {q_avg.tolist()}\n")
        lines.append(f"    avg_pose_4x4:\n")
        for row in np.array(pose_avg).tolist():
            lines.append(f"      {row}\n")
        lines.append("\n")

    with open(results_path, 'w') as fh:
        fh.writelines(lines)

    print(f"Wrote results summary to {results_path}")

def mask_iou(maskA, maskB):
    # maskA, maskB: uint8 masks with non-zero = object
    a = (maskA > 0)
    b = (maskB > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / float(union)

def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))

def mask_median_depth(mask, depth_frame):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    vals = depth_frame[ys, xs].astype(np.float32)
    # depth_frame expected in same units as your MM_PER_UNIT usage (we will compare in meters)
    return float(np.median(vals))

def match_detections(prev_instances, curr_instances):
    """
    prev_instances: list of dicts each {'id': 'apple_0', 'class': 'apple', 'mask': mask, 'depth': z_m}
    curr_instances: list of dicts each {'class':..., 'mask':..., 'depth': z_m}
    Returns: list of assignments [(prev_idx, curr_idx), ...], unmatched_prev_idxs, unmatched_curr_idxs
    """
    assignments = []
    prev_assigned = set()
    curr_assigned = set()

    # compute IoU matrix only for same-class pairs
    scores = []
    for pi, p in enumerate(prev_instances):
        for ci, c in enumerate(curr_instances):
            if p['class'] != c['class']:
                continue
            iou = mask_iou(p['mask'], c['mask'])
            depth_ok = True
            if p.get('depth') is not None and c.get('depth') is not None:
                depth_ok = abs(p['depth'] - c['depth']) <= DEPTH_MATCH_THRESHOLD_M
            score = iou if depth_ok else iou * 0.5
            scores.append((score, pi, ci))

    # sort descending and greedily match if above IOU threshold
    scores.sort(reverse=True, key=lambda x: x[0])
    for score, pi, ci in scores:
        if score < IOU_MATCH_THRESHOLD:
            break
        if pi in prev_assigned or ci in curr_assigned:
            continue
        assignments.append((pi, ci))
        prev_assigned.add(pi)
        curr_assigned.add(ci)

    unmatched_prev = [i for i in range(len(prev_instances)) if i not in prev_assigned]
    unmatched_curr = [i for i in range(len(curr_instances)) if i not in curr_assigned]
    return assignments, unmatched_prev, unmatched_curr

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
