#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from enum import Enum
import time
import threading

from my_robot_interfaces.msg import ObjectPose 
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import Constraints, PositionConstraint, BoundingVolume, OrientationConstraint, PlanningScene, CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool

class RobotState(Enum):
    SETUP = 0
    IDLE = 1
    EXECUTING = 2

class VLABrainNodeV2(Node):
    def __init__(self):
        super().__init__('vla_brain_node_v2')
        self.current_state = RobotState.SETUP
        
        self.TCP_OFFSET = 0.15  
        # 🔥 BIẾN MỚI: Bù trừ độ dài ngón tay kẹp (3.5 cm) để mũi kẹp không đâm xuống bàn
        self.TIP_COMPENSATION = 0.015  
        
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._gripper_client = ActionClient(self, FollowJointTrajectory, '/ned_gripper_controller/follow_joint_trajectory')
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.marker_pub = self.create_publisher(Marker, '/vision_marker', 10)
        self.gripper_pub = self.create_publisher(Bool, '/physical_gripper_cmd', 10)
        
        self.get_logger().info('⏳ Đang kết nối Hệ thần kinh vận động ảo...')
        self._action_client.wait_for_server()
        self._gripper_client.wait_for_server()
        
        self.X_MIN, self.X_MAX = 0.0, 0.0
        self.Y_MIN, self.Y_MAX = 0.0, 0.0
        self.Z_SURFACE = 0.0

    def calibrate_workspace(self):
        print("\n" + "="*50)
        print("🛠️ CHẾ ĐỘ HIỆU CHUẨN WORKSPACE (ĐỌC PHẦN CỨNG THẬT)")
        print("="*50)
        print("\n👉 BƯỚC 1: Vào Niryo Studio, bật công tắc [Learning Mode] để tay máy lỏng ra.")
        
        try:
            from pyniryo import NiryoRobot
            self.get_logger().info("🔌 Đang kết nối trực tiếp tới IP 169.254.200.200...")
            calib_robot = NiryoRobot("169.254.200.200")
        except Exception as e:
            self.get_logger().error(f"🚨 Thất bại! Không kết nối được robot: {e}")
            return

        points = []
        corners = [
            "GÓC XA - TRÁI (Top-Left)", 
            "GÓC XA - PHẢI (Top-Right)", 
            "GÓC GẦN - TRÁI (Bottom-Left)", 
            "GÓC GẦN - PHẢI (Bottom-Right)"
        ]
        
        for corner in corners:
            input(f"\n👉 Kéo sát mũi kẹp chạm vào {corner} rồi bấm [ENTER]...")
            try:
                pose = calib_robot.get_pose()
                points.append((pose.x, pose.y, pose.z))
                self.get_logger().info(f"✅ Đã ghi nhận Encoder (Cổ tay): X={pose.x:.3f}, Y={pose.y:.3f}, Z={pose.z:.3f}")
            except Exception as e:
                self.get_logger().error(f"🚨 Lỗi đọc encoder: {e}")

        print("\n👉 BƯỚC 2: TẮT [Learning Mode] trên Niryo Studio.")
        input("👉 TẮT XONG CHƯA? Bấm [ENTER] để chốt tọa độ...")

        del calib_robot

        xs, ys, zs = [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]
        self.X_MIN, self.X_MAX = min(xs), max(xs)
        self.Y_MIN, self.Y_MAX = min(ys), max(ys)
        
        self.Z_SURFACE = (sum(zs) / len(zs)) - self.TCP_OFFSET  
        
        print("\n" + "="*50)
        self.get_logger().info('✅ HIỆU CHUẨN VẬT LÝ THÀNH CÔNG!')
        self.get_logger().info(f'📏 Trục X (Sâu): {self.X_MIN:.3f} m đến {self.X_MAX:.3f} m')
        self.get_logger().info(f'📏 Trục Y (Ngang): {self.Y_MIN:.3f} m đến {self.Y_MAX:.3f} m')
        self.get_logger().info(f'📏 Độ cao Z mặt bàn thực tế (mũi kẹp): {self.Z_SURFACE:.3f} m')
        print("="*50)
        
        self.subscription = self.create_subscription(ObjectPose, '/detected_object', self.object_callback, 10)
        self.get_logger().info('🧠 BỘ NÃO VLA ĐÃ SẴN SÀNG NHẬN LỆNH!')
        self.current_state = RobotState.IDLE

    def update_hologram(self, x, y, z, color_name, state="FLOOR"):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "locked_target"
        marker.id = 1
        marker.type = Marker.CUBE
        marker.frame_locked = True  
        
        if state == "HIDE":
            marker.action = Marker.DELETE
        else:
            marker.action = Marker.ADD
            marker.scale.x, marker.scale.y, marker.scale.z = 0.02, 0.02, 0.04
            marker.color.a = 0.9
            if "red" in color_name: marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            elif "blue" in color_name: marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
            elif "green" in color_name: marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            else: marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 1.0

            if state == "FLOOR":
                marker.header.frame_id = "base_link"
                marker.pose.position.x = float(x)
                marker.pose.position.y = float(y)
                marker.pose.position.z = float(z + 0.02)
                marker.pose.orientation.w = 1.0
            elif state == "HAND":
                marker.header.frame_id = "tool_link" 
                marker.pose.position.x = 0.0
                marker.pose.position.y = 0.0
                marker.pose.position.z = 0.02 
                marker.pose.orientation.w = 1.0
                
        self.marker_pub.publish(marker)

    def spawn_dynamic_box(self, x, y, z):
        co = CollisionObject()
        co.header.frame_id = "base_link"
        co.header.stamp = self.get_clock().now().to_msg() 
        co.id = "physical_box"
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.02, 0.02, 0.04] 
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z + 0.02)
        pose.orientation.w = 1.0 
        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self.scene_pub.publish(scene)
        time.sleep(0.5)

    def remove_dynamic_box(self):
        co = CollisionObject()
        co.header.frame_id = "base_link"
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = "physical_box"
        co.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self.scene_pub.publish(scene)
        time.sleep(0.2)

    def attach_detach_box(self, attach=True):
        aco = AttachedCollisionObject()
        aco.link_name = "hand_link"
        aco.object.id = "physical_box"
        aco.touch_links = ['mors_1', 'mors_2', 'base_gripper_1', 'hand_link', 'tool_link']
        if attach:
            aco.object.operation = CollisionObject.ADD
        else:
            aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.robot_state.attached_collision_objects.append(aco)
        scene.robot_state.is_diff = True
        scene.is_diff = True
        self.scene_pub.publish(scene)
        time.sleep(0.5)

    def control_gripper(self, open_gripper=True):
        cmd_msg = Bool()
        cmd_msg.data = open_gripper
        self.gripper_pub.publish(cmd_msg)

        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = ['joint_base_to_mors_1', 'joint_base_to_mors_2']
        point = JointTrajectoryPoint()
        point.positions = [0.01, 0.01] if open_gripper else [-0.01, -0.01]
        point.time_from_start.sec = 1
        trajectory.points.append(point)
        goal_msg.trajectory = trajectory
        send_goal_future = self._gripper_client.send_goal_async(goal_msg)
        while rclpy.ok() and not send_goal_future.done():
            time.sleep(0.05)
        goal_handle = send_goal_future.result()
        if goal_handle.accepted:
            result_future = goal_handle.get_result_async()
            while rclpy.ok() and not result_future.done():
                time.sleep(0.05)

    def go_to_xyz(self, x, y, z):
        actual_z = z + self.TCP_OFFSET
        goal_msg = MoveGroup.Goal()
        req = goal_msg.request
        req.group_name = 'ned_arm'
        req.start_state.is_diff = True
        req.max_velocity_scaling_factor = 0.2
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), float(actual_z)
        pose.pose.orientation.x, pose.pose.orientation.y = 0.0, 1.0
        pose.pose.orientation.z, pose.pose.orientation.w = 0.0, 0.0
        req.goal_constraints.append(Constraints(
            position_constraints=[PositionConstraint(header=pose.header, link_name="hand_link",
                constraint_region=BoundingVolume(primitives=[SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.01, 0.01, 0.01])],
                primitive_poses=[pose.pose]))],
            orientation_constraints=[OrientationConstraint(header=pose.header, link_name="hand_link", orientation=pose.pose.orientation,
                absolute_x_axis_tolerance=0.01, absolute_y_axis_tolerance=0.01, absolute_z_axis_tolerance=0.01, weight=1.0)]
        ))
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        while rclpy.ok() and not send_goal_future.done():
            time.sleep(0.05)
        if not send_goal_future.result().accepted:
            return False
        result_future = send_goal_future.result().get_result_async()
        while rclpy.ok() and not result_future.done():
            time.sleep(0.05)
        res = result_future.result().result
        return res.error_code.val == 1

    def object_callback(self, msg):
        if self.current_state == RobotState.EXECUTING:
            return
            
        self.current_state = RobotState.EXECUTING
        self.get_logger().info('====================================')
        self.get_logger().info(f'🎯 ĐÃ BẮT ĐƯỢC TÍN HIỆU TỪ MẮT V2 (TỶ LỆ)!')
        
        x_ratio = max(0.0, min(1.0, msg.x))
        y_ratio_norm = max(0.0, min(1.0, msg.y + 0.5))
        
        real_x = self.X_MIN + x_ratio * (self.X_MAX - self.X_MIN)
        real_y = self.Y_MIN + y_ratio_norm * (self.Y_MAX - self.Y_MIN)
        
        # 🔥 CỘNG THÊM BÙ TRỪ MŨI KẸP VÀO ĐỘ CAO GẮP
        real_z = self.Z_SURFACE + self.TIP_COMPENSATION 
        
        self.get_logger().info(f'📦 Vật thể: {msg.object_name} | X: {real_x:.3f} | Y: {real_y:.3f} | Z: {real_z:.3f}')
        
        threading.Thread(target=self.execute_mission, args=(real_x, real_y, real_z, msg.object_name)).start()

    def execute_mission(self, real_x, real_y, real_z, obj_name):
        try:
            hover_z = real_z + 0.10
            
            self.update_hologram(real_x, real_y, real_z, obj_name, "FLOOR")
            self.get_logger().info('🤖 Bắt đầu xuống gắp...')
            
            self.control_gripper(open_gripper=True)
            
            if self.go_to_xyz(real_x, real_y, hover_z):
                if self.go_to_xyz(real_x, real_y, real_z):
                    
                    self.spawn_dynamic_box(real_x, real_y, real_z)
                    self.control_gripper(open_gripper=False)
                    
                    self.attach_detach_box(attach=True)
                    self.update_hologram(0, 0, 0, obj_name, "HAND")
                    
                    if self.go_to_xyz(real_x, real_y, hover_z):
                        dest_x, dest_y = 0.0, 0.28
                        if self.go_to_xyz(dest_x, dest_y, hover_z):
                            # Nâng lên 1 chút khi thả để không đập hộp vào bàn
                            if self.go_to_xyz(dest_x, dest_y, real_z + 0.02):
                                
                                self.attach_detach_box(attach=False)
                                self.control_gripper(open_gripper=True)
                                self.update_hologram(0, 0, 0, "", "HIDE")
                                
                                time.sleep(0.5)
                                self.remove_dynamic_box()
                                
                                self.go_to_xyz(dest_x, dest_y, hover_z)
                                self.get_logger().info('🏁 HOÀN THÀNH NHIỆM VỤ!')

        except Exception as e:
            self.get_logger().error(f'🚨 Lỗi trong chu trình: {e}')
            self.update_hologram(0, 0, 0, "", "HIDE")
            self.remove_dynamic_box() 

        self.get_logger().info('✅ ĐÃ SẴN SÀNG NHẬN LỆNH MỚI TỪ VLA.')
        self.current_state = RobotState.IDLE

def main(args=None):
    rclpy.init(args=args)
    node = VLABrainNodeV2()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()
    
    node.calibrate_workspace()
    
    try:
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()