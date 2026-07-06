# 04. Baseline AI vision + MoveIt

File này hướng dẫn chạy lại nhánh Baseline AI vision + MoveIt của đồ án.

Baseline này là hướng tham chiếu trước khi chuyển sang các mô hình học hành động như Mini-VLA và Tiny-VLA.

---

## 1. Mục tiêu của baseline

Baseline dùng pipeline truyền thống:

1. Camera quan sát workspace.
2. Vision node xác định vật thể hoặc mục tiêu.
3. Brain node nhận tọa độ mục tiêu.
4. MoveIt xử lý IK, giới hạn khớp và lập kế hoạch chuyển động.
5. Robot hoặc mô phỏng thực hiện thao tác.

Baseline này không phải là Vision-Language-Action end-to-end.

Trong nhánh này:

- Vision node xử lý ảnh và phát hiện mục tiêu.
- Brain node xử lý logic điều khiển.
- MoveIt xử lý IK, giới hạn khớp và lập kế hoạch quỹ đạo.
- Voice node nhận lệnh giọng nói nếu dùng nghiệm thu voice command.

---

## 2. Workspace gốc

Workspace ROS 2 / MoveIt gốc:

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

## 3. Code baseline đã copy vào repo này

Các file baseline đã được copy vào repository bàn giao:

- `baseline_moveit/src/vla_brain_node_v2.py`
- `baseline_moveit/src/vla_vision_node_v2.py`
- `baseline_moveit/src/niryo_bridge_node.py`
- `baseline_moveit/src/voice_node.py`
- `baseline_moveit/msg/ObjectPose.msg`
- `baseline_moveit/launch/run_rviz.launch.py`

Lưu ý:

Các file trong `baseline_moveit/` là bản lưu để bàn giao và tham khảo.

Để chạy trực tiếp bằng `ros2 run`, code vẫn cần nằm trong đúng ROS 2 package `my_robot_control` trong workspace `~/niryo_ws`.

---

## 4. Build workspace

Mở terminal ROS 2, sau đó chạy:

`cd ~/niryo_ws`

`colcon build --symlink-install --packages-select my_robot_control`

`source install/setup.bash`

Nếu custom message chưa được build, build cả interface package:

`cd ~/niryo_ws`

`colcon build --symlink-install --packages-select my_robot_interfaces my_robot_control`

`source install/setup.bash`

---

## 5. Quy trình chạy bản cuối

Trước khi chạy lại, nên tắt các terminal ROS 2 cũ bằng `Ctrl + C`.

Chạy lần lượt từng terminal.

---

### Terminal 1: Gazebo

`source ~/niryo_ws/install/setup.bash && ros2 launch niryo_ned_moveit_config gazebo.launch.py`

---

### Terminal 2: MoveIt

`source ~/niryo_ws/install/setup.bash && ros2 launch niryo_ned_moveit_config move_group.launch.py use_sim_time:=true`

---

### Terminal 3: RViz

`source ~/niryo_ws/install/setup.bash && ros2 launch my_robot_control run_rviz.launch.py`

---

### Terminal 4: Gripper controller

`source ~/niryo_ws/install/setup.bash && ros2 run controller_manager spawner ned_gripper_controller`

---

### Terminal 5: Brain node

`source ~/niryo_ws/install/setup.bash && ros2 run my_robot_control vla_brain_node_v2`

Trong bước này, người vận hành dạy không gian thao tác:

1. Bật Learning Mode trong Niryo Studio.
2. Cầm mũi kẹp kéo tới 4 góc mặt giấy sa bàn.
3. Nhấn ENTER cho từng góc.
4. Tắt Learning Mode.
5. Nhấn ENTER để chốt không gian.

---

### Terminal 6: Niryo bridge node

`source ~/niryo_ws/install/setup.bash && ros2 run my_robot_control niryo_bridge_node`

Node này dùng khi chạy với robot thật hoặc cần cầu nối điều khiển robot.

---

### Terminal 7: Vision node

`source ~/niryo_ws/install/setup.bash && ros2 run my_robot_control vla_vision_node_v2`

Trong cửa sổ camera:

1. Click 4 góc của sa bàn.
2. Chờ giao diện AI Live View.
3. Đặt vật thể vào workspace.
4. Không can thiệp chuột hoặc bàn phím khi hệ thống chuẩn bị gửi lệnh.

---

### Terminal 8: Voice node

`source ~/niryo_ws/install/setup.bash && ros2 run my_robot_control voice_node`

Sau khi terminal báo đã lọc ồn và sẵn sàng nhận lệnh, nói câu lệnh ví dụ:

`Can you pick up the red square please`

---

## 6. Luồng nghiệm thu mong muốn

Luồng hoạt động mong muốn:

1. Voice node nhận câu lệnh.
2. Vision node bắt từ khóa vật thể, ví dụ `red square`.
3. Vision node dời tâm ngắm tới vật thể phù hợp.
4. Vision node gửi tọa độ qua `/detected_object`.
5. Brain node nhận tọa độ.
6. MoveIt lập kế hoạch chuyển động.
7. Robot thực hiện thao tác.

---

## 7. Checklist trước khi chạy

Trước khi chạy baseline, kiểm tra:

- đã source `~/niryo_ws/install/setup.bash`
- workspace đã build thành công
- Gazebo đã mở nếu chạy mô phỏng
- MoveIt `move_group` đã chạy
- RViz đã mở
- gripper controller đã spawn
- robot thật đã an toàn nếu chạy hardware
- camera nhìn rõ workspace
- microphone hoạt động nếu dùng voice node
- không có terminal ROS 2 cũ đang chạy lệnh bị treo

---

## 8. Không commit các thư mục sau

Không đưa các thư mục build sinh tự động vào GitHub:

- `build/`
- `install/`
- `log/`
- `__pycache__/`

Không commit dataset, checkpoint hoặc video lớn trực tiếp vào repository.
