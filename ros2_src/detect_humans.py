import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
import mediapipe as mp
import time

class HumanPoseMovement(Node):
    def __init__(self):
        super().__init__('human_pose_movement_detector')

        self.bridge = CvBridge()
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, qos_profile=sensor_qos)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, qos_profile=sensor_qos)

        self.alert_pub = self.create_publisher(String, '/human_movement_alert', 10)
        self.timer = self.create_timer(0.1, self.publish_state)  # 10Hz

        self.current_rgb_frame = None
        self.current_depth_frame = None
        self.last_torso_pos = None
        self.last_time = None
        self.moving = False
        self.velocity = 0.0
        self.human_present = False

    def depth_callback(self, msg):
        self.current_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.current_rgb_frame = frame.copy()
        if self.current_depth_frame is None:
            self.show_frame(frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(frame_rgb)

        if results.pose_landmarks:
            self.human_present = True
            lm = results.pose_landmarks.landmark

            # Torso centroid
            torso_x = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x +
                       lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x +
                       lm[self.mp_pose.PoseLandmark.LEFT_HIP].x +
                       lm[self.mp_pose.PoseLandmark.RIGHT_HIP].x) / 4
            torso_y = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                       lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y +
                       lm[self.mp_pose.PoseLandmark.LEFT_HIP].y +
                       lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 4

            cx = int(torso_x * frame.shape[1])
            cy = int(torso_y * frame.shape[0])

            # Clip coordinates to be within image bounds
            cx = np.clip(cx, 0, frame.shape[1] - 1)
            cy = np.clip(cy, 0, frame.shape[0] - 1)

            depth_value = float(self.current_depth_frame[cy, cx]) / 1000.0  # meters

            now = time.time()
            if self.last_torso_pos is not None and self.last_time is not None and depth_value > 0:
                dx = cx - self.last_torso_pos[0]
                dy = cy - self.last_torso_pos[1]
                pixel_dist = np.sqrt(dx*dx + dy*dy)
                metric_dist = pixel_dist * (depth_value / 600.0)
                dt = now - self.last_time
                self.velocity = metric_dist / dt if dt > 0 else 0.0

                # Movement detection
                if self.velocity > 0.5:  # threshold
                    self.moving = True
                else:
                    self.moving = False
            else:
                self.moving = False

            # Update torso state
            self.last_torso_pos = (cx, cy)
            self.last_time = now

            # Draw pose
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        else:
            # No human detected
            self.human_present = False
            self.moving = False
            self.last_torso_pos = None
            self.last_time = None
            self.velocity = 0.0

        self.show_frame(frame)

    def show_frame(self, frame):
        display = frame.copy()
        if self.moving:
            overlay = display.copy()
            red = (0, 0, 255)
            cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), red, -1)
            display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
        cv2.imshow("Human Pose Movement Monitor", display)
        cv2.waitKey(1)

    def publish_state(self):
        """Publish human movement state at 10Hz."""
        msg = String()
        if self.human_present:
            msg.data = "human movement detected" if self.moving else "human movement stopped"
        else:
            msg.data = "no human detected"
        self.alert_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanPoseMovement()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
