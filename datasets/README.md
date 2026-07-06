# Datasets

Thư mục này mô tả các dataset dùng trong đồ án Niryo VLA Push.

Dataset thật có dung lượng lớn nên không commit trực tiếp lên GitHub. Chỉ lưu mô tả, cấu trúc dữ liệu, lệnh kiểm tra và link tải nếu có.

---

## 1. Dataset sạch cuối cùng

Dataset sạch/trim cuối cùng:

`~/mini-vla/mini-vla/dataset_push_real_v5_trim_manual`

Số episode:

`73`

Tên file:

`ep_000.npz` đến `ep_072.npz`

Định dạng:

NPZ từng episode.

Dataset gốc V5 cũ là `dataset_push_real_v5`, nhưng bản bàn giao chính là `dataset_push_real_v5_trim_manual`.

---

## 2. Format mỗi episode

Mỗi episode là một file `.npz`.

Các key chính:

- `images`: ảnh RGB, shape `(T, 224, 224, 3)`, dtype `uint8`
- `joints`: trạng thái khớp robot, shape `(T, 6)`, dtype `float32`
- `actions_delta_joints`: action dạng delta joints, shape `(T, 6)`, dtype `float32`
- `actions_next_joints`: joint kế tiếp, shape `(T, 6)`, dtype `float32`
- `instruction`: câu lệnh nhiệm vụ
- `valid_len`: số frame hợp lệ
- `timestamps`: thời gian từng bước
- `target_joints`: target joints
- `replay_target_joints`: replay target joints
- `start_object_feature`: đặc trưng vật thể ban đầu
- `success`: nhãn thành công
- `object_moved`: vật thể có di chuyển hay không
- `table_touch`: robot có chạm bàn hay không
- `manual_cut_return_start_frame`: mốc cắt pha return-home thủ công
- `robot_zone`: vùng đặt robot/object

Ví dụ một episode có:

- `valid_len = 61`
- `instruction = "push the object"`
- `success = True`
- `object_moved = True`
- `table_touch = False`

---

## 3. Converted dataset cho Mini-VLA

File converted chính:

`~/mini-vla/mini-vla/data/niryo_push_real_v5_trim_minivla.npz`

Bản copy trong package Colab:

`~/mini-vla/mini-vla/colab_minivla_push_trim_v5/data/niryo_push_real_v5_trim_minivla.npz`

Shape của file converted:

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

## 4. Script liên quan dataset

Script thu dữ liệu chính:

`~/mini-vla/mini-vla/collect_data_real_v5.py`

Script thu dữ liệu cũ hơn:

`~/mini-vla/mini-vla/collect_data_real_v4.py`

Script convert dataset sang Mini-VLA format:

`~/mini-vla/mini-vla/convert_trim_v5_to_minivla.py`

Script trim/làm sạch cuối cùng cần xác minh thêm nếu muốn tái chạy. Dataset hiện tại có metadata trim như:

- `trim_source_start`
- `trim_source_end`
- `trim_original_valid_len`
- `manual_cut_return_start_frame`

Không nên ghi chắc lệnh trim nếu chưa xác nhận script và command cuối cùng.

---

## 5. Lệnh kiểm tra dataset

Kiểm tra nhanh số episode và shape của một file:

`cd ~/mini-vla/mini-vla`

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

`cd ~/mini-vla/mini-vla`

`python3 report_evidence/dataset_audit/audit_dataset_repo.py | tee report_evidence/dataset_audit/dataset_audit_output.txt`

Output audit:

`~/mini-vla/mini-vla/report_evidence/dataset_audit/dataset_audit_output.txt`

---

## 6. Lệnh convert dataset

Script convert đã xác nhận:

`convert_trim_v5_to_minivla.py`

Lệnh tái tạo dự kiến:

`cd ~/mini-vla/mini-vla`

`python3 convert_trim_v5_to_minivla.py`

Lưu ý:

Lệnh trên là lệnh tái tạo dự kiến dựa trên script đã tìm thấy. Trước khi ghi là command chính thức, nên mở script để xác nhận có cần tham số đầu vào/đầu ra hay không.

---

## 7. Không commit dataset lớn

Không commit trực tiếp các file/thư mục sau lên GitHub:

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
- `*.bin`
- `*.safetensors`

Dataset và checkpoint lớn nên lưu bằng Google Drive, GitHub Release hoặc nền tảng lưu trữ ngoài.

---

## 8. Ghi chú quan trọng

Thông tin chính của dataset sạch:

- image size: `224x224 RGB`
- state dimension: `6D joints`
- action dimension: `6D delta joints`
- converted samples: `3991`
- episodes: `73`
- instruction duy nhất: `push the object`
- source dataset: `dataset_push_real_v5_trim_manual`
- valid_len biến thiên theo episode
- trim audit: `0/73` episode còn nghi return-home
- valid_len min/mean/max xấp xỉ `45 / 55.67 / 71`

Câu tóm tắt:

Final clean dataset is `dataset_push_real_v5_trim_manual` with 73 per-episode NPZ files. It was converted to `data/niryo_push_real_v5_trim_minivla.npz` for Mini-VLA training, containing 3991 samples with images, 6D joint states, 6D delta-joint actions, and tokenized instruction `push the object`.
