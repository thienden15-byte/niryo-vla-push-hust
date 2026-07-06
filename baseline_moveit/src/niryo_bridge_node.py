#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
import time
import threading
from std_msgs.msg import Bool 

# ==========================================
# CÔNG TẮC CHUYỂN ĐỔI MÔI TRƯỜNG 
# (Ở NHÀ ĐỂ FALSE. LÊN LAB ĐỔI THÀNH True)
IS_ON_LAB = True 
# ==========================================

if IS_ON_LAB:
    from pyniryo import NiryoRobot
else:
    class MockNiryoRobot:
        def __init__(self, ip): print(f"🤖 [MOCK] Kết nối IP: {ip}")
        def set_arm_max_velocity(self, vel): pass
        def execute_trajectory_from_poses_and_joints(self, list_pose_joints, list_type): time.sleep(2)
        def close_gripper(self): print("🤖 [MOCK] ĐÓNG kẹp!")
        def open_gripper(self): print("🤖 [MOCK] MỞ kẹp!")

class NiryoBridgeNode(Node):
    def __init__(self):
        super().__init__('niryo_bridge_node')
        
        self.get_logger().info('⏳ Khởi động Trạm Cầu Nối (Bridge Node)...')
        
        # 🔑 KHỞI TẠO KHÓA CHỐNG XUNG ĐỘT ĐA LUỒNG
        self.robot_lock = threading.Lock()
        
        if IS_ON_LAB:
            self.get_logger().info('🔌 [LAB] Đang kết nối TCP/IP tới phần cứng...')
            self.robot = NiryoRobot("169.254.200.200")
            self.robot.set_arm_max_velocity(10) 
        else:
            self.get_logger().info('🏠 [HOME] Chế độ Mocking kích hoạt.')
            self.robot = MockNiryoRobot("169.254.200.200")

        # Lắng nghe quỹ đạo di chuyển
        self.subscription = self.create_subscription(
            DisplayTrajectory, '/display_planned_path', self.trajectory_callback, 10)
            
        # Lắng nghe lệnh tay kẹp
        self.gripper_sub = self.create_subscription(
            Bool, '/physical_gripper_cmd', self.gripper_callback, 10)
            
        self.get_logger().info('🌉 ĐÃ SẴN SÀNG. Chờ tóm quỹ đạo và lệnh kẹp...')

    def gripper_callback(self, msg):
        # Chạy lệnh kẹp trong một luồng phụ để không làm đơ ROS
        threading.Thread(target=self.execute_gripper_hardware, args=(msg.data,)).start()

    def execute_gripper_hardware(self, should_open):
        # 🔑 XẾP HÀNG: Đợi cánh tay dừng hẳn/thả socket ra mới được điều khiển kẹp
        with self.robot_lock:
            if IS_ON_LAB:
                try:
                    if should_open:
                        self.get_logger().info("🗜️ [PHẦN CỨNG] Đang thực thi: MỞ KẸP THẬT!")
                        self.robot.open_gripper()
                    else:
                        self.get_logger().info("🗜️ [PHẦN CỨNG] Đang thực thi: ĐÓNG KẸP THẬT!")
                        self.robot.close_gripper()
                except Exception as e:
                    self.get_logger().error(f"🚨 Lỗi phần cứng kẹp: {e}")

    def trajectory_callback(self, msg):
        if not msg.trajectory or len(msg.trajectory[0].joint_trajectory.points) == 0:
            return
            
        points = msg.trajectory[0].joint_trajectory.points
        trajectory_array = []
        for pt in points:
            joints = [float(angle) for angle in pt.positions]
            trajectory_array.append(joints)
        
        type_array = ['joint'] * len(trajectory_array)
        
        # Đẩy việc di chuyển xuống luồng phụ
        threading.Thread(target=self.execute_robot, args=(trajectory_array, type_array)).start()

    def execute_robot(self, trajectory_array, type_array):
        # 🔑 XẾP HÀNG: Chiếm quyền điều khiển độc quyền đường truyền socket
        with self.robot_lock:
            self.get_logger().info('🚀 [PHẦN CỨNG] Cầu nối đang đẩy mảng quỹ đạo xuống motor...')
            try:
                self.robot.execute_trajectory_from_poses_and_joints(
                    list_pose_joints=trajectory_array,
                    list_type=type_array
                )
                self.get_logger().info('✅ [PHẦN CỨNG] Cánh tay di chuyển xong!')
            except Exception as e:
                self.get_logger().error(f"🚨 Lỗi truyền quỹ đạo: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = NiryoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()