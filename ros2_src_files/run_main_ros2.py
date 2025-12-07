#!/usr/bin/env python3
import sys
sys.path.append("/home/hirolab/divam/Markerless_6D_Pose_Estimation")
import rclpy, os
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from threading import Thread
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
import cv2
from ROS2_Utils import *


class RGBDSubscriber(Node):
    def __init__(self):
        super().__init__('rgbd_subscriber')

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # -------------------- User Config --------------------
        self.Folder_Name = 'Meal_Tray_Scenario'
        self.model_name = 'medical_objects_fruits'
        self.target_classes = ["apple", "banana", "yogurt", "sb_cup", "ketchup", "tray", "orange"]
        self.script_directory = "/home/hirolab/divam/Markerless_6D_Pose_Estimation"
        self.ROOT = pathlib.Path(f"{self.script_directory}/{self.Folder_Name}")

        self.MAX_FRAMES_DEFAULT = 5
        # -----------------------------------------------------

        # Number of frames to capture per service call
        self.NUM_CAPTURE_FRAMES = 5

        # Subscribers
        self.rgb_sub = message_filters.Subscriber(self, ImageMsg, '/camera/color/image_raw', qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, ImageMsg, '/camera/depth/image_raw', qos_profile=qos)
        self.camera_intrinsics = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info', qos_profile=qos)
        self.intrinsics_dict = None

        # Publishers
        self.annotated_pub = self.create_publisher(ImageMsg, '/camera/annotated_rgb', 10)
        self.objects_pub = self.create_publisher(String, '/camera/detected_objects', 10)
        self.pose_annotated_pub = self.create_publisher(ImageMsg, '/pose_annotated', 10)
        self.pose_pub = self.create_publisher(String, '/object_pose', 10)
        self.annotated_translation_pub = self.create_publisher(ImageMsg, '/annotated_translation_color', 10)
        self.object_poses_translation_pub = self.create_publisher(String, '/object_poses_translation', 10)

        # predictor + model info (unchanged)
        self.all_classes, self.dataset_name, self.mesh_names = parse_model_info(self.model_name)
        self.predictor = build_detector(self.dataset_name, len(self.all_classes), self.all_classes,
                                        model_path=f"{self.script_directory}/detectron2_models/{self.model_name}.pth")

        # Service
        self.capture_service = self.create_service(
            Trigger,
            'capture_frames',
            self.capture_frames_callback
        )

        # Capture / processing state
        # self.capturing -> True means node is busy (collecting OR processing)
        # self.collecting -> True while we are collecting frames (subset of capturing)
        self.capturing = False
        self.collecting = False
        self.capture_target = self.NUM_CAPTURE_FRAMES
        self.frames_rgb = []
        self.frames_depth = []
        self.masks_per_frame = []
        self.next_instance_id = 0
        self.persistent_objects = {}

        # Worker thread handle (None when idle)
        self._processing_thread = None

        # sync approx
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.camera_intrinsics],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.callback)

        self.get_logger().info("RGB + Depth subscriber started.")

    def get_translation(self, rgb, depth):
        if self.intrinsics_dict is None:
            return

        K = self.intrinsics_dict["K"]
        MM_PER_UNIT = self.intrinsics_dict["MM_PER_UNIT"]
        depth_m = depth.astype(np.float32) * MM_PER_UNIT / 1000.0

        outputs = self.predictor(rgb)
        instances = outputs["instances"].to("cpu")

        if not instances.has("pred_masks"):
            return

        masks = instances.pred_masks.numpy()
        pred_classes = instances.pred_classes.numpy()

        vis = rgb.copy()
        results_dict = {}

        for i in range(len(pred_classes)):
            cname = self.class_names[pred_classes[i]]

            mask = (masks[i].astype(np.uint8) * 255)
            pca = compute_pca_axis(mask)
            if pca is None:
                continue

            cx, cy, dx, dy = pca
            yaw_deg = compute_yaw_from_pca(dx, dy)
            z = mask_median_depth(mask, depth_m)
            if z is None:
                continue

            cen = mask_centroid(mask)
            if cen is None:
                continue

            u, v = cen
            xyz = deproject(cx, cy, z, K)

            color = random_color(cname)
            mask_rgb = np.zeros_like(vis)
            mask_rgb[mask > 0] = color
            vis = cv2.addWeighted(vis, 1.0, mask_rgb, 0.3, 0)

            axis_len = 80
            x1 = int(cx - dx * axis_len)
            y1 = int(cy - dy * axis_len)
            x2 = int(cx + dx * axis_len)
            y2 = int(cy + dy * axis_len)
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            txt = f"{cname} ({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})m {yaw_deg:.1f}deg"
            cv2.putText(vis, txt, (int(u), int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            results_dict[cname] = {
                "translation": [float(xyz[0]/1000.0), float(xyz[1]/1000.0), float(xyz[2]/1000.0)],
                "yaw_deg": float(yaw_deg)
            }

        try:
            msg_img = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            self.annotated_translation_pub.publish(msg_img)
        except:
            pass

        try:
            msg = String()
            msg.data = json.dumps(results_dict)
            self.object_poses_translation_pub.publish(msg)
        except:
            pass

    def callback(self, rgb_msg, depth_msg, info_msg):
        """Receive synchronized frames and publish poses/images continuously."""
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        if self.intrinsics_dict is None:
            self.intrinsics_dict = {
                "K": np.array(info_msg.k).reshape(3, 3),
                "D": np.array(info_msg.d),
                "width": info_msg.width,
                "height": info_msg.height,
                "MM_PER_UNIT": 1000.0*1.0000000474974513
            }

        # ----- Detect objects (fast) -----
        outputs = self.predictor(rgb)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_masks = instances.pred_masks.cpu().numpy()
        metadata = MetadataCatalog.get(self.dataset_name)
        self.class_names = metadata.thing_classes

        # Annotated visualization for raw publishing (detection masks)
        vis = rgb.copy()
        detected_objects_names = []

        for i, class_id in enumerate(pred_classes):
            cname = self.class_names[class_id]
            if cname in self.target_classes:
                detected_objects_names.append(cname)
                mask = (pred_masks[i] * 255).astype(np.uint8)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                vis = cv2.addWeighted(vis, 1.0, mask_3ch, 0.3, 0)
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    cv2.putText(vis, cname, (xs[0], ys[0]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Publish raw annotated image (always)
        try:
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
        except Exception as e:
            # avoid crashing callback on publish errors
            self.get_logger().error(f"Failed to publish annotated_rgb: {e}")

        # Publish detected objects (always)
        objs_msg = String()
        objs_msg.data = ','.join(detected_objects_names)
        self.objects_pub.publish(objs_msg)
        # After publishing annotated images / detected object names
        self.get_translation(rgb, depth)


        # --- If we're in capture mode, collect frames but DO NOT run heavy pose estimation here ---
        if self.capturing and self.collecting:
            # Build frame_mask_dict for this frame and store
            frame_mask_dict = {}
            for i, class_id in enumerate(pred_classes):
                cname = self.class_names[class_id]
                if cname not in self.target_classes:
                    continue
                instance_id = f"{cname}_{self.next_instance_id}"
                # self.next_instance_id += 1
                mask = pred_masks[i].astype(np.uint8)
                frame_mask_dict[instance_id] = mask
                self.persistent_objects[instance_id] = {"class": cname}

            # Save copies
            self.frames_rgb.append(rgb.copy())
            self.frames_depth.append(depth.copy())
            self.masks_per_frame.append(frame_mask_dict)

            # If we've collected enough frames, spawn background worker to process them
            if len(self.frames_rgb) >= self.capture_target and self._processing_thread is None:
                self.get_logger().info(f"Collected {len(self.frames_rgb)} frames; starting processing thread.")
                # Make shallow deep copies of the lists to hand off to the worker
                rgb_list = [f.copy() for f in self.frames_rgb]
                depth_list = [d.copy() for d in self.frames_depth]
                masks_list = [dict(m) for m in self.masks_per_frame]
                persistent_copy = dict(self.persistent_objects)
                print(persistent_copy) 

                # Set flags: we are no longer collecting but still capturing (processing)
                self.collecting = False
                # self.capturing = False # debug
                # spawn worker
                t = Thread(target=self._process_captured_frames,
                           args=(rgb_list, depth_list, masks_list, persistent_copy, self.intrinsics_dict))
                t.daemon = True
                self._processing_thread = t
                t.start()
            else:
                # still collecting; nothing else to do on this frame
                pass

        # If not capturing, nothing extra to do in this callback (we already published raw annotated image)
        return

    def _process_captured_frames(self, rgb_list, depth_list, masks_list, persistent_objects_copy, intrinsics_dict):
        """Runs in a background thread: slow pose estimation, generate annotated images, publish poses + images."""
        try:
            self.get_logger().info("Background pose estimation started.")
            # Build set of detected object ids for pose estimator across frames (union of masks_list keys)
            detected_objects = set()
            for m in masks_list:
                detected_objects.update(m.keys())
            instance_to_class = {iid: persistent_objects_copy[iid]['class'] for iid in detected_objects}

            # Call your heavy pose estimation function on the captured batch
            poses_per_frame, meshes_in_use = estimate_pose(
                rgb_list, depth_list, masks_list,
                detected_objects, intrinsics_dict,
                self.mesh_names, self.all_classes,
                self.ROOT, instance_to_class
            )

            # K matrix for save_combined_results_ros
            K = intrinsics_dict["K"]

            # Generate annotated frames (pose overlays)
            annotated_frames = save_combined_results_ros(
                rgb_list, masks_list, poses_per_frame, meshes_in_use, K, self.ROOT
            )

            # Publish annotated frames (pose_annotated topic) — may be 1 or many images
            for vis in annotated_frames:
                try:
                    self.pose_annotated_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
                except Exception as e:
                    self.get_logger().error(f"Failed to publish pose_annotated image: {e}")

            # Publish pose messages for each pose found
            # poses_per_frame is expected to be a mapping frame_idx -> {instance_id: pose4x4}
            # for frame_idx, objects in poses_per_frame.items():
            #     for iid, pose4x4 in objects.items():
            #         ps = PoseStamped()
            #         ps.header.frame_id = "camera_frame"
            #         ps.header.stamp = self.get_clock().now().to_msg()
            #         ps.pose.position.x = float(pose4x4[0, 3])
            #         ps.pose.position.y = float(pose4x4[1, 3])
            #         ps.pose.position.z = float(pose4x4[2, 3])

            #         # rotation matrix -> quaternion (np.array [x,y,z,w])
            #         quat = rotmat_to_quat(pose4x4[:3, :3])
            #         ps.pose.orientation.x = float(quat[0])
            #         ps.pose.orientation.y = float(quat[1])
            #         ps.pose.orientation.z = float(quat[2])
            #         ps.pose.orientation.w = float(quat[3])

            #         try:
            #             self.pose_pub.publish(ps)
            #         except Exception as e:
            #             self.get_logger().error(f"Failed to publish PoseStamped: {e}")
            object_instances = {}

            for frame_idx, objects in poses_per_frame.items():
                for iid, pose4x4 in objects.items():
                    if iid not in object_instances:
                        object_instances[iid] = {"translations": [], "quaternions": [], "class": self.persistent_objects[iid]["class"]}
                    object_instances[iid]["translations"].append(pose4x4[:3, 3])
                    object_instances[iid]["quaternions"].append(rotmat_to_quat(pose4x4[:3, :3]))

            if len(object_instances) == 0:
                self.get_logger().warn("No poses found to average.")
                return

            # Compute averages and prepare final dict
            final_dict = {}
            for iid, data in object_instances.items():
                translations = np.array(data["translations"])
                quaternions = np.array(data["quaternions"])
                t_avg = translations.mean(axis=0).tolist()
                q_avg = average_quaternions(quaternions).tolist()
                final_dict[iid] = {
                    "class": data["class"],
                    "translation": t_avg,
                    "quaternion": q_avg
                }

            msg = String()
            msg.data = json.dumps(final_dict)
            self.pose_pub.publish(msg)
            self.get_logger().info("Published averaged poses as JSON string.")
            self.get_logger().info("Background pose estimation finished and published results.")

        except Exception as e:
            # ensure exceptions don't kill the thread silently
            self.get_logger().error(f"Exception in background processing thread: {e}")
        finally:
            # Clear capture buffers and mark node as ready again
            # NOTE: we intentionally keep self.persistent_objects growth minimal by removing the captured ones
            try:
                # Remove captured persistent entries (if present) to avoid unbounded growth
                for m in masks_list:
                    for iid in m.keys():
                        if iid in self.persistent_objects:
                            del self.persistent_objects[iid]
            except Exception:
                pass

            # Reset state so service can be called again
            self.frames_rgb = []
            self.frames_depth = []
            self.masks_per_frame = []
            self._processing_thread = None
            self.capturing = False
            self.collecting = False
            self.get_logger().info("Node ready for next capture call.")

    def capture_frames_callback(self, request, response):
        # If we're busy (collecting OR processing), reject new starts
        if self.capturing:
            response.success = False
            response.message = "Already capturing."
            return response

        # Start new capture cycle
        self.get_logger().info("Capture service triggered. Starting capture...")

        self.frames_rgb = []
        self.frames_depth = []
        self.masks_per_frame = []
        # optionally reset persistent_objects or keep it — we'll keep it but we clear entries after processing
        # self.persistent_objects = {}

        self.capture_target = self.NUM_CAPTURE_FRAMES
        self.capturing = True
        self.collecting = True

        response.success = True
        response.message = f"Capture started: collecting {self.capture_target} frames."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = RGBDSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
