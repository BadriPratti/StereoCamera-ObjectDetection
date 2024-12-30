import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from yolo_model import load_yolo_model, detect_objects
from config import CONFIDENCE_THRESHOLD, BASELINE, FOCAL_LENGTH
from draw_utils import annotate_frame

class StereoProcessor(Node):
    def __init__(self):
        super().__init__('stereo_processor')
        self.bridge = CvBridge()
        self.yolo_model = load_yolo_model()

        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.left_sub = self.create_subscription(
            Image, '/multisense/right/image_rect', self.left_callback, qos_profile
        )
        self.disparity_sub = self.create_subscription(
            Image, '/multisense/right/disparity', self.disparity_callback, qos_profile
        )

        self.left_image = None
        self.disparity_map = None

    def left_callback(self, msg):
        self.get_logger().info("Received left image")
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_images()

    def disparity_callback(self, msg):
        self.get_logger().info("Received disparity map")
        self.disparity_map = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.process_images()

    def process_images(self):
        if self.left_image is None or self.disparity_map is None:
            return  

        boxes = detect_objects(self.yolo_model, self.left_image)

        for i, box in enumerate(boxes.xyxy):
            confidence = boxes.conf[i]
            if confidence < CONFIDENCE_THRESHOLD:
                continue  

            x1, y1, x2, y2 = map(int, box)  
            cls = int(boxes.cls[i])  
            label = self.yolo_model.names[cls]  

            disparity_region = self.disparity_map[y1:y2, x1:x2]
            valid_disparity = disparity_region[disparity_region > 0]  

            if valid_disparity.size > 0:
                disparity = np.median(valid_disparity)
                depth = BASELINE * FOCAL_LENGTH / disparity
            else:
                depth = float('inf') 

            pixel_width = x2 - x1
            pixel_height = y2 - y1

            real_width = (pixel_width * depth) / FOCAL_LENGTH if depth != float('inf') else float('inf')
            real_height = (pixel_height * depth) / FOCAL_LENGTH if depth != float('inf') else float('inf')

            annotate_frame(self.left_image, x1, y1, x2, y2, label, depth, real_width, real_height)

        cv2.imshow("YOLO with Depth", self.left_image)
        cv2.waitKey(1)
