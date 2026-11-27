# BASELINE — Jetson AI Racer Challenge Fall 2026

Tài liệu gốc hiện hành:
- [Thể_lệ_JetRacer_FA2026.docx](docs/reference/Thể_lệ_JetRacer_FA2026.docx) — Thể lệ chính thức Fall 2026.
- [Cheemsity_JetRacer_FA2026_v1.pdf](docs/reference/Cheemsity_JetRacer_FA2026_v1.pdf) — Đề cương nghiên cứu (Observability-Aware Camera–LiDAR Fusion for Autonomous JetRacer).
- [HuongDanSuDungLidar.pdf](docs/reference/HuongDanSuDungLidar.pdf) — Quy trình khởi động JetRacer, LiDAR, SLAM (ROS Melodic) và RViz.
- [Jetson.ipynb](Jetson.ipynb) — Notebook camera live, thu dataset và điều khiển tay cầm từ tài liệu mới.

---

## 1. Cấu trúc cuộc thi & Chiến lược điểm FA2026

| Vòng / Hạng mục | Trọng số | Nội dung chính |
|---|---|---|
| **Vòng 2 – Speed Track** (29/11) | Lọc Top 15 + Thưởng (+10/+7/+5) | 2 lượt × 5p, sa bàn vòng kín, bám làn + vật cản tĩnh. Lấy lượt cao nhất. |
| **Vòng 3 – Bài 1: SEE** (27/12) | 35đ SC (**25%** tổng) | Camera: bám làn (12), biển báo (13), đèn giao thông (6, vượt đỏ hủy lượt), vạch dừng (4). Cần undistortion rìa ảnh 160°. |
| **Vòng 3 – Bài 2: SENSE** (27/12) | 35đ SC (**25%** tổng) | LiDAR + Camera: vật cản Cấp 1/2/3 (8/8/9), khu thiếu sáng/hành lang hẹp (5 – LiDAR), vật cản thấp dưới mặt phẳng LiDAR (5 – Camera). |
| **Vòng 3 – Bài 3: DECIDE** (27/12) | 30đ SC (**20%** tổng) | Bắt buộc Camera + LiDAR: nút giao xung đột (15) + đỗ xe vào ô (15: trong vạch, ≤ 10 cm, ≤ 15°, ≤ 3 lần chỉnh, dừng ≥ 3s). |
| **Technical Paper** (20/12) | **20%** | 6–8 trang IEEE, nộp kèm video, source code, logs. |
| **Oral Defense** (27/12) | **10%** | 6 phút trình bày + 6 phút Q&A. |

> **Lưu ý trọng yếu:**
> 1. Điểm mỗi bài Smart City = `0.6 × lượt cao + 0.4 × lượt thấp` → **độ ổn định lặp lại** quan trọng hơn 1 lượt ăn may.
> 2. **Bắt buộc nộp rosbag** chứng minh dữ liệu Camera & LiDAR thực sự đi vào pipeline ra quyết định (không ghi thụ động).
> 3. **Vượt đèn đỏ = hủy lượt**; không chắc trạng thái đèn thì luôn dừng.

---

## 2. Kiến trúc mục tiêu FA2026 (theo Technical Paper)

Hệ thống mục tiêu vận hành trên **ROS Melodic (`~/catkin_ws`)** trên Jetson Nano (JetPack 4.5.1, Python 3.6 / C++):

| ROS Module | Đầu ra chính | Trạng thái trong repo hiện tại |
|---|---|---|
| `visual_perception_node` | `/perception/lane`, `/perception/traffic`, `/perception/objects`, `/perception/parking_slot` | **Một phần** (đã có CV/CNN lane, stopline, shading ở Python thuần; **chưa có** ROS wrapper, fisheye undistortion, YOLOv5n TRT FP16, parking bay) |
| `lidar_perception_node` | `/lidar/obstacles`, `/lidar/corridor` | **Chưa triển khai** (mới có hướng dẫn bring-up `slam.launch` & `/scan`) |
| `local_state_node` | `/state/odom` (kèm covariance) | **Chưa triển khai** (cần scan registration + visual geometric supplement) |
| `adaptive_fusion_node` | `/world/local_map`, `/fusion/evidence` | **Chưa triển khai** (3 layers: LiDAR obstacles, camera low obstacles, observability) |
| `behavior_fsm_node` | `/behavior/events` | **Một phần** (`fsm.py` hiện mới có bản đơn giản cho đèn/biển; **chưa có** conflict assessment & parking states) |
| `local_planner_node` | `/planning/trajectory` | **Chưa triển khai** (cần Hybrid A\* cho parking với ngân sách ≤ 3 lần chỉnh hướng) |
| `trajectory_controller_node` | `/control/candidate` | **Một phần** (đã có PID + CornerController; **chưa có** Pure Pursuit & short-horizon parking controller) |
| `safety_supervisor_node` | `/control/safe_cmd` | **Chưa triển khai** (cần khối giám sát độc lập: stale data, braking corridor, trajectory validity) |
| `vehicle_interface_node` | Throttle & Servo / `/cmd_vel` | **Đã có nền tảng** (`control/driver.py` hỗ trợ cả `nvidia` PCA9685 `0x40` và `ros` `/cmd_vel`, có watchdog) |

---

## 3. Vai trò của codebase Python hiện tại (`src/jetracer_baseline/`)

Codebase hiện tại được kế thừa từ baseline ban đầu (chạy Python thuần, chỉ dùng camera) và hiện đóng vai trò:

1. **Hạ tầng bring-up & kiểm tra phần cứng** (`tools/check_hardware.py`, `tools/diagnose_shutdown.py`, `tools/check_py36.py`, `camera.py`, `perception/shading.py`) — tiếp tục sử dụng để kiểm tra camera CSI, nguồn 5W, servo/motor, lens shading.
2. **Baseline bám làn ban đầu cho Speed Track** (`perception/lane.py`, `perception/lane_cnn.py`, `control/pid.py`, `control/corner.py`, `tuning_ui.py`) — dùng để kiểm chứng bám làn nhanh trên sa bàn trong giai đoạn đầu Shared Lab đợt 1, trước khi tích hợp LiDAR tránh vật cản và chuyển sang kiến trúc ROS đầy đủ.
3. **Thu thập dữ liệu** (`manual_collection.py`, `collect_drive.ipynb`, `Jetson.ipynb`, `tools/collect_dataset.py`).

**Các hạng mục cũ đã lỗi thời (cần thay thế khi nâng cấp lên ROS):**
- Công thức tính điểm cũ trong `tools/analyze_log.py` (Speed Track FPS nhị phân / Smart City cũ).
- Pipeline đơn tiến trình chỉ dùng camera trong `pipeline.py` và FSM luật cứng theo thời gian trong `fsm.py`.
- Danh sách class biển báo cũ trong `configs/default.yaml` (thiếu `stop`, `speed_limit`, `no_entry`, `pedestrian`, `parking_slot`).

---

## 4. Phân công & Rủi ro chính (theo Technical Paper §VI)

- **Member 1:** Vision, dataset, camera calibration/undistortion, YOLOv5n TRT FP16, **RQ1 (SEE)**.
- **Member 2:** LiDAR, extrinsic calibration, adaptive fusion, ROS deployment & rosbag logging, **RQ2 (SENSE)**.
- **Member 3:** FSM, Pure Pursuit, Hybrid A\* parking, conflict resolution, safety supervisor, **RQ3 (DECIDE)**.

**Rủi ro ưu tiên:** Quá tải Jetson Nano (giảm tải bằng bounded queues + C++ nodes + TRT FP16), thay đổi ánh sáng (quality-dependent fusion + fallback LiDAR corridor), sai lệch calibration giữa 4 xe Shared Lab (kiểm tra theo `vehicle_id`), thiếu ca lab (sàng lọc offline trên rosbag/replay trước khi chạy closed-loop).
