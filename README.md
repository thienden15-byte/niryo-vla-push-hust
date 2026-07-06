# VLA Niryo Object Manipulation

Repository này là kho mã nguồn và tài liệu bàn giao cho đồ án thao tác vật thể trên robot Niryo Ned.

Mục tiêu của repository là giúp người nghiên cứu sau có thể hiểu lại hệ thống, chạy lại các bước chính và tiếp tục phát triển mà không phải xây dựng lại từ đầu.

## 1. Các hướng triển khai trong đồ án

Đồ án gồm bốn hướng triển khai chính:

1. Baseline AI vision + MoveIt
2. Mini-VLA
3. HSV-assisted waypoint
4. Tiny-VLA

### 1.1. Baseline AI vision + MoveIt

Đây là hướng tham chiếu theo cách tiếp cận truyền thống hơn.

Luồng xử lý:

- camera hoặc AI vision xác định mục tiêu
- MoveIt xử lý IK, giới hạn khớp và lập kế hoạch chuyển động
- robot thực hiện quỹ đạo

Nhánh này không phải Vision-Language-Action end-to-end. AI vision hỗ trợ xác định mục tiêu, còn MoveIt đảm nhiệm phần lập kế hoạch chuyển động.

### 1.2. Mini-VLA

Mini-VLA là hướng học hành động chính trong đồ án.

Đầu vào:

- ảnh RGB từ camera
- câu lệnh nhiệm vụ
- trạng thái khớp robot

Đầu ra:

- 6D delta joints

Trong đồ án có hai biến thể:

- Diffusion Mini-VLA
- Direct Mini-VLA

Direct Mini-VLA được ưu tiên thử nghiệm trên robot thật vì cho kết quả offline tốt hơn Diffusion Mini-VLA.

### 1.3. HSV-assisted waypoint

HSV-assisted waypoint là hướng hỗ trợ thực nghiệm.

Nhánh này dùng phân đoạn màu HSV để xác định vật thể, sinh waypoint và gửi lệnh cho robot. Đây là hướng giúp kiểm chứng luồng camera đến robot thật và so sánh với hướng học hành động.

### 1.4. Tiny-VLA

Tiny-VLA là hướng mở rộng theo mô hình VLA có sẵn.

Nhánh này sử dụng dữ liệu theo định dạng HDF5, qpos 7D và action 10D. Mục tiêu là kiểm tra khả năng thích nghi Tiny-VLA cho robot Niryo Ned.

## 2. Cấu trúc repository

Các thư mục chính:

- assets/: hình ảnh, sơ đồ và video minh họa
- baseline_moveit/: code và ghi chú cho baseline AI vision + MoveIt
- checkpoints/: mô tả checkpoint và link tải checkpoint lớn
- configs/: file cấu hình camera, robot, workspace và training
- datasets/: mô tả dataset, định dạng dữ liệu và link tải dữ liệu lớn
- docs/: tài liệu kỹ thuật tổng quan
- evidence/: log, biểu đồ và SHA256 manifest để đối chiếu kết quả
- experiments/: kết quả thử nghiệm, ảnh, video và bảng so sánh
- huong_dan/: hướng dẫn thao tác chi tiết bằng tiếng Việt
- models/: mô hình Mini-VLA, Tiny-VLA và HSV-assisted waypoint
- scripts/: script thu dữ liệu, xử lý dữ liệu, train, eval và chạy robot
- src/: mã nguồn tách theo module
- utils/: hàm phụ trợ cho camera, robot, dataset và xử lý ảnh

## 3. Luồng làm việc chính

Các bước chính của project:

1. Thiết lập robot Niryo Ned, camera và máy tính điều khiển
2. Thu thập dữ liệu trình diễn từ robot thật
3. Làm sạch dữ liệu và loại bỏ pha không cần thiết
4. Chuyển đổi dữ liệu sang định dạng train cho Mini-VLA hoặc Tiny-VLA
5. Huấn luyện mô hình
6. Đánh giá offline bằng L2 error, cosine similarity và action norm
7. Chạy rollout trên robot thật
8. Ghi lại log, ảnh, biểu đồ và kết quả thực nghiệm

## 4. Lưu ý về môi trường

Project có cả code ROS 2 / MoveIt và code PyTorch.

Không nên trộn hai môi trường này trong cùng một terminal nếu không cần thiết.

Khuyến nghị:

- Terminal chạy ROS 2 / MoveIt: source /opt/ros/humble/setup.bash
- Terminal chạy Mini-VLA / Tiny-VLA / PyTorch: dùng virtualenv hoặc conda riêng

## 5. Cấu hình robot và camera

Không nên hardcode IP robot hoặc camera index trong code.

Repo sử dụng file mẫu:

- .env.example

Người dùng copy thành file cấu hình local:

- cp .env.example .env

Các thông số thường cần sửa:

- NIRYO_IP
- CAMERA_INDEX
- ROBOT_VELOCITY
- MINIVLA_STEPS
- MINIVLA_ACTION_SCALE
- MINIVLA_MAX_DELTA

File .env thật không nên commit lên GitHub.

## 6. Dataset và checkpoint

Dataset và checkpoint lớn không nên lưu trực tiếp trong GitHub.

Thay vào đó, lưu ở Google Drive, GitHub Release hoặc nền tảng lưu trữ ngoài, sau đó ghi link tải và SHA256 trong:

- datasets/README.md
- checkpoints/README.md

Các dataset chính:

- Mini-VLA NPZ dataset: ảnh RGB, 6D joints, 6D delta joints
- Tiny-VLA HDF5 dataset: qpos 7D, action 10D

Các checkpoint chính:

- Diffusion Mini-VLA
- Direct Mini-VLA
- Tiny-VLA author-style

## 7. Evidence và kết quả thực nghiệm

Thư mục evidence/ dùng để lưu bằng chứng có thể đối chiếu lại.

Nội dung nên có:

- log đánh giá offline
- log rollout robot thật nếu có
- biểu đồ so sánh kết quả
- SHA256 manifest cho dataset, checkpoint, script và log

Mục đích là giúp người đọc kiểm tra lại số liệu trong báo cáo từ file thật, checkpoint thật và script thật.

## 8. Trạng thái hiện tại

Repository hiện đang trong giai đoạn chuẩn hóa để bàn giao.

Các phần cần có trong bản hoàn thiện:

- code thu thập dữ liệu
- code xử lý và chuyển đổi dataset
- code huấn luyện Mini-VLA
- code đánh giá offline
- code rollout robot thật
- code baseline MoveIt
- code HSV-assisted waypoint
- ghi chú Tiny-VLA runtime
- link dataset và checkpoint
- log, biểu đồ và file bằng chứng

## 9. Hướng phát triển

Các hướng phát triển tiếp theo:

- mở rộng dataset với nhiều vị trí vật thể hơn
- bổ sung dữ liệu correction khi robot đi lệch
- cải thiện visual grounding
- thử image encoder mạnh hơn như CLIP hoặc SigLIP
- cải thiện độ ổn định khi rollout robot thật
- mở rộng từ bài toán đẩy vật sang gắp và đặt vật
- so sánh định lượng giữa baseline MoveIt, Mini-VLA, Tiny-VLA và HSV waypoint

## 10. Ghi chú

Repository này được thiết kế như tài liệu bàn giao kỹ thuật cho khóa sau.

Người đọc nên bắt đầu từ README.md, sau đó đọc tiếp huong_dan/ và docs/ để biết cách cài đặt, thu dữ liệu, train, eval và chạy robot thật.

---

## Demo videos

Một số video demo ngắn được lưu tại:

`assets/videos/`

Các video hiện có:

- `01_baseline_moveit_demo.mp4`: Baseline AI vision + MoveIt demo
- `02_hsv_goal_v2_demo_1.mp4`: HSV / Goal V2 robot demo
- `03_hsv_goal_v2_demo_2.mp4`: HSV / Goal V2 robot demo
- `04_hsv_goal_v2_demo_3.mp4`: HSV / Goal V2 robot demo
- `05_hsv_goal_v2_demo_4.mp4`: HSV / Goal V2 robot demo

Chỉ các video demo ngắn được commit vào GitHub. Video raw dài, dataset và checkpoint lớn không được commit.
