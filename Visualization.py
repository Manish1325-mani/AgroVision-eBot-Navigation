import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Point, Quaternion
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import message_filters
from visualization_msgs.msg import Marker
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
import math
from collections import defaultdict

class FruitFertilizerDetector(Node):

    def __init__(self):
        super().__init__('fruit_fertilizer_detector')

        self.team_id = 0
        self.min_area = 550
        self.max_area = 5000

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(Marker, '/object_markers', 10)

        self.camera_matrix = None
        self.camera_frame_id = None
        self.base_frame = 'base_link'

        self.window_active = False
        self.window_title = "Detection Output"

        self.object_data = defaultdict(lambda: {
            'pos': None,
            'time': None,
            'obj_type': None,
            'obj_id': None,
            'marker_id': None
        })

        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.cam_info_cb, 10)

        self.img_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')

        self.sync = message_filters.TimeSynchronizer([self.img_sub, self.depth_sub], 10)
        self.sync.registerCallback(self.image_cb)

        self.get_logger().info("Detection system ready")

    def cam_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'cx': msg.k[2], 'cy': msg.k[5]
            }
            self.camera_frame_id = msg.header.frame_id
            self.get_logger().info("Camera setup complete")
            self.destroy_subscription(self.cam_info_sub)

    def find_bad_fruits(self, hsv_img):
        low_range = np.array([0, 0, 60])
        high_range = np.array([179, 150, 290])
        return cv2.inRange(hsv_img, low_range, high_range)

    def find_aruco_markers(self, color_img, depth_img):
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray_img)
        markers = []
        
        if ids is not None:
            for i, marker_id in enumerate(ids):
                center = np.mean(corners[i][0], axis=0)
                u, v = int(center[0]), int(center[1])
                
                if 0 <= v < depth_img.shape[0] and 0 <= u < depth_img.shape[1]:
                    d_val = depth_img[v, u]
                else:
                    continue
                    
                if np.isnan(d_val) or d_val == 0:
                    continue

                fx = self.camera_matrix['fx']
                fy = self.camera_matrix['fy']
                cx = self.camera_matrix['cx']
                cy = self.camera_matrix['cy']

                z = float(d_val)
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                cam_pos = [z, -x, -y]

                markers.append({
                    'id': marker_id[0],
                    'corners': corners[i],
                    'center': (u, v),
                    'cam_pos': cam_pos
                })
        
        return markers

    def to_base_frame(self, cam_point):
        try:
            point_msg = PointStamped()
            point_msg.header.frame_id = self.camera_frame_id
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.point.x = cam_point[0]
            point_msg.point.y = cam_point[1]
            point_msg.point.z = cam_point[2]

            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            base_point = do_transform_point(point_msg, transform)
            return [base_point.point.x, base_point.point.y, base_point.point.z]
            
        except Exception as e:
            self.get_logger().warn(f"Transform error: {str(e)}")
            return None

    def fertilizer_orientation(self):
        base_q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        roll_q = self.euler_to_quat(math.pi / 2.0, 0.0, 0.0)
        
        x1, y1, z1, w1 = base_q.x, base_q.y, base_q.z, base_q.w
        x2, y2, z2, w2 = roll_q.x, roll_q.y, roll_q.z, roll_q.w

        rx = w1*x2 + x1*w2 + y1*z2 - z1*y2
        ry = w1*y2 - x1*z2 + y1*w2 + z1*x2
        rz = w1*z2 + x1*y2 - y1*x2 + z1*w2
        rw = w1*w2 - x1*x2 - y1*y2 - z1*z2

        result = Quaternion()
        result.x = rx
        result.y = ry
        result.z = rz
        result.w = rw
        return result

    def publish_tf(self, position, obj_id=None, obj_type="fruit", aruco_id=None):
        try:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = self.base_frame
            
            if obj_type == "fruit":
                frame_name = f'{self.team_id}_bad_fruit_{obj_id}'
            elif aruco_id == 3:
                frame_name = f'{self.team_id}_fertiliser_can'
            else:
                frame_name = f'{self.team_id}_aruco_{aruco_id}'
            
            tf_msg.child_frame_id = frame_name
            tf_msg.transform.translation.x = position[0]
            tf_msg.transform.translation.y = position[1]
            tf_msg.transform.translation.z = position[2]
            
            if obj_type == "fruit":
                tf_msg.transform.rotation = self.euler_to_quat(math.pi, 0, math.pi)
            elif aruco_id == 3:
                tf_msg.transform.rotation = self.fertilizer_orientation()
            else:
                tf_msg.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            self.tf_broadcaster.sendTransform(tf_msg)
            self.create_marker(position, obj_id, obj_type, aruco_id)
            return True
            
        except Exception as e:
            self.get_logger().error(f"TF error: {str(e)}")
            return False

    def euler_to_quat(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q

    def create_marker(self, position, obj_id, obj_type, aruco_id=None):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.base_frame
        
        if obj_type == "fruit":
            marker.id = obj_id + 100
            marker.ns = "fruits"
            marker.type = Marker.SPHERE
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
        else:
            if aruco_id == 3:
                marker.id = 3
                marker.ns = "fertilizer"
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            else:
                marker.id = aruco_id + 200
                marker.ns = "markers"
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 1.0
            
            marker.type = Marker.CUBE
            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08
        
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        
        if obj_type == "fruit":
            marker.pose.orientation = self.euler_to_quat(math.pi, 0, math.pi)
        elif aruco_id == 3:
            marker.pose.orientation = self.fertilizer_orientation()
        else:
            marker.pose.orientation.w = 1.0
            
        marker.color.a = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
        
        self.marker_pub.publish(marker)

    def image_cb(self, rgb_msg, depth_msg):
        if self.camera_matrix is None:
            self.get_logger().warn("Waiting for camera data")
            return

        try:
            color_img = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except CvBridgeError as e:
            self.get_logger().error(f'Bridge error: {str(e)}')
            return

        display_img = color_img.copy()
        
        markers = self.find_aruco_markers(color_img, depth_img)
        
        fertilizer_found = False
        for marker in markers:
            marker_id = marker['id']
            corners = marker['corners']
            center = marker['center']
            cam_pos = marker['cam_pos']
            
            base_pos = self.to_base_frame(cam_pos)
            
            if base_pos is not None:
                if marker_id == 3:
                    fertilizer_found = True
                    
                    obj_name = f'{self.team_id}_fertiliser_can'
                    self.object_data[obj_name] = {
                        'pos': base_pos,
                        'time': self.get_clock().now(),
                        'obj_type': 'aruco',
                        'obj_id': None,
                        'marker_id': marker_id
                    }
                    
                    self.publish_tf(base_pos, obj_type="aruco", aruco_id=3)
                    
                    cv2.aruco.drawDetectedMarkers(display_img, [corners], np.array([marker_id]))
                    cv2.putText(display_img, "2266_fertiliser_can", 
                                (center[0]-60, center[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.circle(display_img, center, 5, (255, 0, 0), -1)
        
        hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        fruit_mask = self.find_bad_fruits(hsv_img)

        kernel = np.ones((20, 15), np.uint8)
        clean_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])

            if v >= fruit_mask.shape[0] or u >= fruit_mask.shape[1] or fruit_mask[v, u] == 0:
                continue

            if v < depth_img.shape[0] and u < depth_img.shape[1]:
                d_val = depth_img[v, u]
            else:
                continue
                
            if np.isnan(d_val) or d_val == 0:
                continue

            fx = self.camera_matrix['fx']
            fy = self.camera_matrix['fy']
            cx = self.camera_matrix['cx']
            cy = self.camera_matrix['cy']

            z = float(d_val)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            cam_pos = [z, -x, -y]
            
            base_pos = self.to_base_frame(cam_pos)
            
            if base_pos is not None:
                fruit_count += 1
                current_id = fruit_count

                obj_name = f'{self.team_id}_bad_fruit_{current_id}'
                self.object_data[obj_name] = {
                    'pos': base_pos,
                    'time': self.get_clock().now(),
                    'obj_type': 'fruit',
                    'obj_id': current_id,
                    'marker_id': None
                }

                self.publish_tf(base_pos, current_id, obj_type="fruit")

                x_pos, y_pos, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_img, (x_pos, y_pos), (x_pos + w, y_pos + h), (0, 255, 0), 2)
                cv2.putText(display_img, f"bad fruit {current_id}", (x_pos, y_pos - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(display_img, (u, v), 5, (0, 255, 0), -1)

        status_text = f"Fruits: {fruit_count} | Fertilizer: {'YES' if fertilizer_found else 'NO'}"
        cv2.putText(display_img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.show_image(display_img)

    def show_image(self, img):
        try:
            if not self.window_active:
                cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_title, 800, 600)
                self.window_active = True
            
            cv2.imshow(self.window_title, img)
            key = cv2.waitKey(1)
            
            if key == ord('x') or key == 27:
                self.get_logger().info("Shutdown initiated")
                raise KeyboardInterrupt
                
        except Exception as e:
            if not self.window_active:
                self.get_logger().warn(f"Display issue: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = FruitFertilizerDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown complete")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

