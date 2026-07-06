# Hướng dẫn sử dụng repository

Thư mục này chứa các hướng dẫn thao tác chi tiết bằng tiếng Việt cho đồ án robot Niryo Ned.

Người đọc nên đi theo thứ tự dưới đây để hiểu và chạy lại hệ thống.

## Thứ tự đọc đề xuất

1. Cài đặt môi trường và kết nối robot/camera
2. Baseline AI vision + MoveIt
3. Dataset và thu thập dữ liệu
4. Mini-VLA
5. HSV-assisted waypoint
6. Tiny-VLA
7. Evidence, log và checkpoint
8. Commit/push và bàn giao repository

## Ý nghĩa từng phần

### 1. Cài đặt môi trường và kết nối robot/camera

Phần này hướng dẫn chuẩn bị máy tính, môi trường Python, ROS 2 / MoveIt, PyNiryo, robot Niryo Ned và camera.

### 2. Baseline AI vision + MoveIt

Phần này mô tả hướng baseline dùng AI vision hoặc xử lý ảnh để xác định mục tiêu, sau đó dùng MoveIt để lập kế hoạch chuyển động cho robot.

Đây là hướng tham chiếu trước khi chuyển sang các mô hình học hành động như Mini-VLA và Tiny-VLA.

### 3. Dataset và thu thập dữ liệu

Phần này hướng dẫn cách thu dữ liệu từ robot thật, tổ chức episode, lưu ảnh, trạng thái robot và action.

### 4. Mini-VLA

Phần này hướng dẫn chuyển đổi dataset, train Mini-VLA, đánh giá offline và rollout trên robot thật.

### 5. HSV-assisted waypoint

Phần này mô tả phương pháp hỗ trợ thực nghiệm dùng HSV để phát hiện vật thể và sinh waypoint cho robot.

### 6. Tiny-VLA

Phần này ghi lại hướng thích nghi Tiny-VLA cho robot Niryo Ned, bao gồm định dạng HDF5, qpos, action và checkpoint.

### 7. Evidence, log và checkpoint

Phần này hướng dẫn cách lưu log, biểu đồ, SHA256 manifest, dataset link và checkpoint link để đối chiếu kết quả trong báo cáo.

### 8. Commit/push và bàn giao repository

Phần này hướng dẫn cách kiểm tra file, commit và push repository lên GitHub.
