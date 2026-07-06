# 05. Dataset và thu dữ liệu

File này hướng dẫn phần dataset và thu dữ liệu cho đồ án Niryo VLA Push.

Phần này tập trung vào dataset robot thật dùng cho Mini-VLA.

---

## 1. Dataset sạch cuối cùng

Dataset sạch/trim cuối cùng nằm tại:

`~/mini-vla/mini-vla/dataset_push_real_v5_trim_manual`

Dataset này có:

- 73 episode
- file từ `ep_000.npz` đến `ep_072.npz`
- định dạng NPZ từng episode
- đã loại bỏ pha return-home thủ công

Dataset gốc V5 cũ là:

`~/mini-vla/mini-vla/dataset_push_real_v5`

Tuy nhiên bản này không còn là bản bàn giao chính. Bản dùng chính là:

`dataset_push_real_v5_trim_manual`

---

## 2. Format mỗi episode

Mỗi episode là một file `.npz`.

Các key chính:

- `images`: ảnh RGB, shape `(T, 224, 224, 3)`, dtype `uint8`
- `joints`: trạng thái khớp robot, shape `(T, 6)`, dtype `float32`
- `actions_delta_joints`: action dạng delta joints, shape `(T, 6)`
- `actions_next_joints`: joint kế tiếp, shape `(T, 6)`
- `instruction`: câu lệnh nhiệm vụ
- `valid_len`: số frame hợp lệ
- `timestamps`: thời gian từng bước
- `target_joints`: target joints
- `replay_target_joints`: replay target joints
- `start_object_feature`: đặc trưng vật thể ban đầu
- `success`: nhãn thành công
- `object_moved`: vật thể có di chuyển hay không
- `table_touch`: robot có chạm bàn hay không
- `manual_cut_return_start_frame`: mốc cắt pha return-home
- `robot_zone`: vùng đặt robot/object

Ví dụ một episode có thể có:

- `valid_len = 61`
- `instruction = "push the object"`
- `success = True`
- `object_moved = True`
- `table_touch = False`

---

## 3. Script thu dữ liệu

Script thu dữ liệu chính đã copy vào repo này:

`scripts/phase_1_data_collection/collect_data_real_v5.py`

Đường dẫn gốc trên máy:

`~/mini-vla/mini-vla/collect_data_real_v5.py`

Script cũ hơn để tham khảo:

`scripts/phase_1_data_collection/collect_data_real_v4.py`

Đường dẫn gốc:

`~/mini-vla/mini-vla/collect_data_real_v4.py`

Lưu ý:

- Ưu tiên dùng bản V5.
- Bản V4 giữ lại để truy vết lịch sử phát triển.
- Không nên xóa bản cũ nếu chưa chắc các ghi chú hoặc kết quả nào còn phụ thuộc vào nó.

---

## 4. Lệnh thu dữ liệu dự kiến

Lệnh chạy từ repo gốc Mini-VLA:

`cd ~/mini-vla/mini-vla`

`python3 collect_data_real_v5.py --robot-zone near_center`

Có thể đổi `near_center` thành zone khác nếu script hỗ trợ.

Lưu ý:

Lệnh trên là lệnh tái chạy theo script hiện có. Trước khi thu dữ liệu mới, cần mở script để kiểm tra lại các tham số hiện tại.

---

## 5. Camera và cấu hình khi thu dữ liệu

Thông tin đã dùng trong quá trình thu dữ liệu:

- camera USB IMX179
- camera name thường chứa `USB Camera2`
- camera index thường gặp `/dev/video3`
- ảnh resize về `224x224`
- instruction chính: `push the object`
- replay khoảng 100 waypoint
- no-padding trong dataset V5
- state là 6D joints
- action là 6D delta joints

Trước khi thu dữ liệu mới, cần kiểm tra lại camera bằng:

`v4l2-ctl --list-devices`

và kiểm tra `/dev/videoX` bằng OpenCV.

---

## 6. Script convert sang Mini-VLA format

Script convert đã copy vào repo này:

`scripts/phase_2_preprocessing/convert_trim_v5_to_minivla.py`

Đường dẫn gốc trên máy:

`~/mini-vla/mini-vla/convert_trim_v5_to_minivla.py`

File output converted chính:

`~/mini-vla/mini-vla/data/niryo_push_real_v5_trim_minivla.npz`

---

## 7. Converted dataset cho Mini-VLA

File converted có:

- `images`: `(3991, 224, 224, 3)`, dtype `uint8`
- `states`: `(3991, 6)`, dtype `float32`
- `actions`: `(3991, 6)`, dtype `float32`
- `text_ids`: `(3991, 16)`, dtype `int64`
- `episode_ids`: `(3991,)`, dtype `int32`
- `step_ids`: `(3991,)`, dtype `int32`

Vocabulary:

- `<pad>`: 0
- `<unk>`: 1
- `push`: 2
- `the`: 3
- `object`: 4

Instruction chính:

`push the object`

---

## 8. Lệnh kiểm tra dataset

Chạy từ thư mục gốc Mini-VLA:

`cd ~/mini-vla/mini-vla`

Kiểm tra nhanh:

`python3 - <<'PY'`
`from pathlib import Path`
`import numpy as np`
`D = Path("dataset_push_real_v5_trim_manual")`
`files = sorted(D.glob("ep_*.npz"))`
`print("episodes:", len(files))`
`print("first:", files[0] if files else None)`
`print("last :", files[-1] if files else None)`
`d = np.load(files[0], allow_pickle=True)`
`for k in d.files:`
`    v = d[k]`
`    print(f"{k:32s}", v.shape, v.dtype)`
`PY`

Audit command đã dùng:

`python3 report_evidence/dataset_audit/audit_dataset_repo.py | tee report_evidence/dataset_audit/dataset_audit_output.txt`

Output audit:

`~/mini-vla/mini-vla/report_evidence/dataset_audit/dataset_audit_output.txt`

---

## 9. Không commit dataset lớn

Không commit trực tiếp các file hoặc thư mục sau lên GitHub:

- `dataset_push_real_v5_trim_manual/`
- `dataset_push_real_v5/`
- `raw_frames/`
- `bad_episodes/`
- `bad_raw_frames/`
- `extra_episodes/`
- `ep_*.npz`
- `*.npz`
- `*.zip`
- `*.pt`
- `*.pth`
- `*.ckpt`

Dataset lớn nên lưu ở Google Drive, GitHub Release hoặc nền tảng lưu trữ ngoài.

Trong GitHub repo chỉ nên lưu:

- script thu dữ liệu
- script convert
- mô tả dataset
- checksum nếu có
- link tải dataset nếu có

---

## 10. Ghi chú quan trọng

Thông tin dataset sạch:

- episodes: 73
- converted samples: 3991
- image size: 224x224 RGB
- state dimension: 6D joints
- action dimension: 6D delta joints
- instruction duy nhất: `push the object`
- valid_len min/mean/max xấp xỉ `45 / 55.67 / 71`
- trim audit: `0/73` episode còn nghi return-home

Dataset này là dataset chính dùng để train và đánh giá Mini-VLA trong đồ án.
