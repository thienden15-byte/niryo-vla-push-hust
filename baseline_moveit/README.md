# Baseline AI Vision + MoveIt

Thư mục này ghi lại phần baseline AI vision kết hợp MoveIt được dùng trong đồ án Niryo VLA Push.

Đây là hướng tham chiếu trước khi chuyển sang các mô hình học hành động như Mini-VLA và Tiny-VLA.

---

## 1. Vai trò của baseline

Baseline này dùng cách tiếp cận truyền thống hơn:

1. Camera quan sát workspace.
2. Vision node xác định vật thể hoặc mục tiêu thao tác.
3. Brain node nhận tọa độ mục tiêu.
4. MoveIt xử lý IK, giới hạn khớp và lập kế hoạch chuyển động.
5. Robot thực hiện thao tác.

Baseline này không phải là Vision-Language-Action end-to-end.

Trong baseline này, phần vision hỗ trợ xác định mục tiêu, còn MoveIt đảm nhiệm phần lập kế hoạch chuyển động.

---

## 2. Workspace và package

Workspace ROS 2 / MoveIt:

`~/niryo_ws`

Package chính:

`my_robot_control`

Package interface:

`my_robot_interfaces`

Custom message:

`~/niryo_ws/src/my_robot_interfaces/msg/ObjectPose.msg`

Topic chính:

`/detected_object`

Topic này dùng để truyền thông tin vật thể từ vision node sang brain/control node.

---

## 3. Các node trong bản chạy cuối

Bản baseline cuối cùng sử dụng các node chính sau:

### Brain node

`vla_brain_node_v2`

Vai trò:

- nhận thông tin mục tiêu
- dạy hoặc hiệu chỉnh không gian thao tác
- chứa logic điều khiển MoveIt
- gửi lệnh robot thông qua hệ thống ROS 2 / MoveIt

File nguồn tương ứng:

`~/niryo_ws/src/my_robot_control/my_robot_control/vla_brain_node_v2.py`

### Vision node

`vla_vision_node_v2`

Vai trò:

- mở camera
- cho người dùng chọn 4 góc sa bàn
- xử lý live camera view
- xác định vật thể theo lệnh
- gửi tọa độ vật thể qua topic `/detected_object`

File nguồn tương ứng:

`~/niryo_ws/src/my_robot_control/my_robot_control/vla_vision_node_v2.py`

### Niryo bridge node

`niryo_bridge_node`

Vai trò:

- đóng vai trò cầu nối khi chạy với cánh tay robot thật
- hỗ trợ gửi lệnh từ hệ ROS 2 sang robot thật nếu cần

### Voice node

`voice_node`

Vai trò:

- nhận lệnh giọng nói
- chuyển giọng nói thành text command
- ví dụ câu lệnh nghiệm thu: `Can you pick up the red square please`

---

## 4. Build workspace

Build package sau khi sửa code:

`cd ~/niryo_ws && colcon build --symlink-install --packages-select my_robot_control`

Sau khi build xong, source workspace:

`source ~/niryo_ws/install/setup.bash`

Ghi chú:

Nếu setup lại từ máy sạch và custom message chưa được build, có thể cần build thêm package interface:

`cd ~/niryo_ws && colcon build --symlink-install --packages-select my_robot_interfaces my_robot_control`

---

## 5. Quy trình chạy bản cuối

Trước khi chạy lại, nên tắt các terminal ROS 2 cũ bằng `Ctrl + C`.

Sau đó chạy lần lượt từng terminal.

### Terminal 1: Gazebo

`source ~/niryo_ws/install/setup.bash && ros2 launch niryo_ned_moveit_config gazebo.launch.py`

### Terminal 2: MoveIt

`source ~/niryo_ws/install/setup.bash && ros2 launch niryo_ned_moveit_config move_group.launch.py use_sim_time:=true`

### Terminal 3: RViz

`source ~/niryo_ws/install/setup.bash && ros2 launch my_robot_control run_rviz.launch.py`

### Terminal 4: Gripper controller

`source ~/niryo_ws/install/setup.bash && ros2 run controller_manager spawner ned_gripper_controller`

### Terminal 5: Brain node

`source ~/niryo_ws/install/setup.bash && ros2 run my_robot_control vla_brain_node_v2`

Trong bước này, người vận hành dạy không gian thao tác:

1. Bật Learning Mode trong Niryo Studio.
2. Cầm mũi kẹp kéo tới 4 góc mặt giấy sa bàn.
3. Nhấn ENTER cho từng góc.
4. Tắt Learning Mode.
5. Nhấn ENTER để chốt không gian.

### Terminal 6: Niryo bridge node

`source ~/niryo_ws/install/setup.bash && ros2 run my_robot_control niryo_bridge_node`

### Terminal 7: Vision node

`source ~/niryo_ws/install/setup.bash && ros2 run my_robot_control vla_vision_node_v2`

Trong cửa sổ camera:

1. Chọn 4 góc sa bàn.
2. Chờ giao diện AI Live View.
3. Đặt vật thể vào workspace.
4. Không can thiệp chuột/bàn phím khi hệ thống chuẩn bị gửi lệnh.

### Terminal 8: Voice node

`source ~/niryo_ws/install/setup.bash && ros2 run my_robot_control voice_node`

Sau khi hệ thống báo đã lọc ồn và sẵn sàng nhận lệnh, nói câu lệnh ví dụ:

`Can you pick up the red square please`

Luồng nghiệm thu mong muốn:

1. Voice node nhận câu lệnh.
2. Vision node bắt từ khóa vật thể.
3. Vision node dời tâm ngắm tới vật thể phù hợp.
4. Vision node gửi tọa độ.
5. Brain node nhận tọa độ và điều khiển robot.

---

## 6. Không commit các thư mục này

Không đưa các thư mục sinh ra khi build ROS 2 lên GitHub:

- `build/`
- `install/`
- `log/`
- `__pycache__/`

Không commit dataset lớn, video lớn hoặc checkpoint lớn trực tiếp vào repo.

---

## 7. Ghi chú

Bản baseline này từng nằm trong workspace ROS 2 riêng, chưa được tách thành thư mục `baseline_moveit/` ngay từ đầu.

Khi bàn giao, nên copy các file nguồn liên quan từ `~/niryo_ws` vào thư mục này hoặc ghi rõ đường dẫn gốc để khóa sau truy vết.

Các file quan trọng cần copy hoặc tham chiếu:

- `vla_brain_node_v2.py`
- `vla_vision_node_v2.py`
- `niryo_bridge_node.py`
- `voice_node.py`
- `ObjectPose.msg`
- `run_rviz.launch.py`
