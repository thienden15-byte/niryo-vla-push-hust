# 07. HSV-assisted waypoint

File này ghi lại phần HSV-assisted waypoint / Goal V2 trong đồ án Niryo VLA Push.

Đây là hướng hỗ trợ thực nghiệm, dùng thị giác đơn giản để xác định vật thể và sinh quỹ đạo khớp cho robot.

---

## 1. Vai trò của HSV-assisted waypoint

HSV-assisted waypoint không phải là Vision-Language-Action end-to-end.

Phương pháp này dùng:

- camera image
- click hoặc HSV color detection
- object feature `[cx, cy, area]`
- Goal V2 cluster/retrieval policy
- PyNiryo robot control

Mục tiêu là tạo một hướng chạy robot thật ổn định hơn so với Direct Mini-VLA rollout.

---

## 2. Root gốc

Code gốc nằm tại:

`~/mini-vla/mini-vla/goal_v1`

Tên thư mục là `goal_v1`, nhưng bên trong có cả Goal V1 và Goal V2.

---

## 3. Input và output

Input chính của Goal V2:

`object_feature = [cx, cy, area]`

Trong đó:

- `cx`: tọa độ tâm vật thể theo ảnh hoặc vùng chuẩn hóa
- `cy`: tọa độ tâm vật thể theo ảnh hoặc vùng chuẩn hóa
- `area`: diện tích vật thể sau khi detect

Luồng input thường là:

`camera image -> click / HSV detection -> object_feature [cx, cy, area]`

Output của Goal V2:

`15 joint waypoints`

Shape:

`(15, 6)`

Tức là 15 điểm quỹ đạo, mỗi điểm gồm 6 khớp robot.

Sau đó quỹ đạo được gửi qua PyNiryo để robot thực hiện.

---

## 4. Code đã copy vào repo này

Preprocessing:

- `scripts/phase_2_preprocessing/hsv_goal/convert_v5_to_goal_v1.py`

Build policy:

- `scripts/phase_3_training/hsv_goal/build_goal_v2_cluster_policy.py`

Evaluation / test feature:

- `scripts/phase_4_evaluation/hsv_goal/test_auto_object_feature.py`

Runtime robot thật:

- `scripts/phase_5_real_robot_test/hsv_goal/live_click_object_to_niryo_v2.py`
- `scripts/phase_5_real_robot_test/hsv_goal/live_click_object_to_niryo.py`
- `scripts/phase_5_real_robot_test/hsv_goal/run_goal_v2_cluster_on_niryo.py`
- `scripts/phase_5_real_robot_test/hsv_goal/run_policy_path_on_niryo.py`

---

## 5. Script chính

Script chính để chạy camera click + HSV + Goal V2:

`live_click_object_to_niryo_v2.py`

Đường dẫn gốc:

`~/mini-vla/mini-vla/goal_v1/scripts/live_click_object_to_niryo_v2.py`

Script chạy Goal V2 bằng object_feature nhập tay:

`run_goal_v2_cluster_on_niryo.py`

Đường dẫn gốc:

`~/mini-vla/mini-vla/goal_v1/scripts/run_goal_v2_cluster_on_niryo.py`

---

## 6. Chạy Goal V2 bằng object_feature nhập tay

Từ root gốc Mini-VLA:

`cd ~/mini-vla/mini-vla`

Command ví dụ:

`python3 goal_v1/scripts/run_goal_v2_cluster_on_niryo.py --ip 169.254.200.200 --object-feature 0.7673067 0.45228928 0.00358995 --velocity 3`

Dry-run:

`python3 goal_v1/scripts/run_goal_v2_cluster_on_niryo.py --ip 169.254.200.200 --object-feature 0.7673067 0.45228928 0.00358995 --dry-run`

---

## 7. Chạy camera click + HSV + Goal V2

Từ root gốc Mini-VLA:

`cd ~/mini-vla/mini-vla`

Command chính:

`python3 goal_v1/scripts/live_click_object_to_niryo_v2.py --ip 169.254.200.200 --camera-index 3 --velocity 3`

Dry-run:

`python3 goal_v1/scripts/live_click_object_to_niryo_v2.py --ip 169.254.200.200 --camera-index 3 --velocity 3 --dry-run`

Chỉ định màu:

`python3 goal_v1/scripts/live_click_object_to_niryo_v2.py --ip 169.254.200.200 --camera-index 3 --velocity 3 --color green`

---

## 8. Dataset và policy

Goal V2 data:

`~/mini-vla/mini-vla/goal_v1/data/push_goal_v2_cluster.npz`

Goal V2 policy:

`~/mini-vla/mini-vla/goal_v1/checkpoints/goal_v2_cluster_policy.npz`

Goal V1 data:

`~/mini-vla/mini-vla/goal_v1/data/push_goal_v1.npz`

Goal V1 policy:

`~/mini-vla/mini-vla/goal_v1/checkpoints/goal_v1_retrieval_policy.npz`

Ghi chú:

- `push_goal_v2_cluster.npz` và `goal_v2_cluster_policy.npz` khá nhỏ.
- Tuy nhiên `.gitignore` hiện chặn `*.npz`, nên repo hiện tại chưa commit các file `.npz`.
- Nếu muốn demo chạy ngay, có thể tạo ngoại lệ riêng cho policy nhỏ sau.

---

## 9. Kết quả robot thật

Ghi nhận hiện tại:

- Goal V1 3 điểm chạy được qua PyNiryo.
- Goal V2 sinh 15 joint waypoints.
- Goal V2 bằng object_feature nhập tay đã chạy được trên robot thật.
- Camera click + HSV + Goal V2 là hướng runtime chính.
- Robot đi ổn định hơn Direct Mini-VLA rollout thật.
- Trong các test chính không ghi nhận cạ bàn nghiêm trọng.

So với Mini-VLA Direct, hướng HSV/Goal V2 ổn định hơn trong robot thật vì không dự đoán delta joints từng bước, mà sinh quỹ đạo waypoint rõ ràng hơn.

---

## 10. Hạn chế

Hạn chế chính:

- perception vẫn dựa vào HSV hoặc click
- chưa phải VLA đầy đủ `image + language + state`
- instruction mới ở mức chọn màu hoặc chọn vật thể đơn giản
- vật màu xanh dương có thể bị nhầm với màu robot nếu ROI không siết tốt
- phụ thuộc ánh sáng và camera
- dataset/demo memory còn ít

---

## 11. Không commit file lớn

Không commit:

- dataset lớn
- raw frames
- video lớn
- zip dataset
- checkpoint lớn
- converted Mini-VLA NPZ
- file `.npz` lớn

Các policy `.npz` nhỏ có thể cân nhắc commit sau nếu cần demo chạy ngay, nhưng cần sửa `.gitignore` có kiểm soát.

---

## 12. Ghi chú cho báo cáo

Tên báo cáo có thể gọi hướng này là:

`HSV-assisted waypoint`

hoặc:

`Goal V2 cluster/retrieval waypoint policy`

Mô tả ngắn:

Phương pháp này dùng đặc trưng vật thể `[cx, cy, area]` từ camera/HSV để truy xuất hoặc sinh quỹ đạo khớp gồm 15 waypoint, sau đó gửi trực tiếp cho robot thật qua PyNiryo.
