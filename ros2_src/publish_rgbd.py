#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class RealSensePublisher(Node):

    def __init__(self):
        super().__init__('realsense_publisher')

        self.bridge = CvBridge()

        qos = QoSProfile(
                            reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            history=QoSHistoryPolicy.KEEP_LAST,
                            depth=5
                        )
        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', qos)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', qos)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', qos)
        self.mm_per_unit_pub = self.create_publisher(Float32,'/camera/depth/mm_per_unit', qos)



        # Set up Intel RealSense streams
        TARGET_SERIAL = "244222076386"
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(TARGET_SERIAL)

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = self.pipeline.start(config)
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.MM_PER_UNIT = 1000.0 * depth_scale
        self.camera_info_msg = None  # Initialize the CameraInfo message

        # Timer for publishing images
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("RealSense RGB + Depth publisher started.")
        self.get_logger().info(f"Depth scale: {depth_scale} m/unit → MM_PER_UNIT = {self.MM_PER_UNIT}")

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return
        
        if self.camera_info_msg is None:
            self.camera_info_msg = self.create_camera_info(color_frame)

        # Convert to numpy arrays
        rgb_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        # Publish RGB
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8')
        self.rgb_pub.publish(rgb_msg)

        # Publish depth (16-bit)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding='16UC1')
        self.depth_pub.publish(depth_msg)

        # Publish CameraInfo
        self.info_pub.publish(self.camera_info_msg)

        # Publish MM_PER_UNIT
        mm_msg = Float32()
        mm_msg.data = float(self.MM_PER_UNIT)
        self.mm_per_unit_pub.publish(mm_msg)

    def create_camera_info(self, color_frame):
        """Extract intrinsics from RealSense and convert to CameraInfo."""
        camera_info = CameraInfo()

        profile = color_frame.get_profile()
        intr = profile.as_video_stream_profile().get_intrinsics()

        camera_info.width = intr.width
        camera_info.height = intr.height

        # Camera matrix (K)
        camera_info.k = [
            intr.fx, 0.0,     intr.ppx,
            0.0,     intr.fy, intr.ppy,
            0.0,     0.0,     1.0
        ]

        # Distortion parameters
        camera_info.d = list(intr.coeffs)

        camera_info.distortion_model = 'plumb_bob'

        # Projection matrix (P)
        camera_info.p = [
            intr.fx, 0.0,     intr.ppx, 0.0,
            0.0,     intr.fy, intr.ppy, 0.0,
            0.0,     0.0,     1.0,       0.0
        ]

        # depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # MM_PER_UNIT = 1000.0 * depth_scale  # mm per unit (usually 1 mm)

        # camera_info.MM_PER_UNIT = MM_PER_UNIT

        return camera_info
    
    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    ctx = rs.context()
    print("Connected devices:")
    for d in ctx.devices:
        print(f"  - {d.get_info(rs.camera_info.serial_number)}")
    
    rclpy.init(args=args)
    node = RealSensePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
