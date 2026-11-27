# JetRacer — Jetson AI Racer Challenge Fall 2026

Repository mã nguồn và tài liệu triển khai cho **Jetson AI Racer Challenge – Fall 2026** (Vòng loại **Speed Track** & Vòng chung kết **Smart City: SEE – SENSE – DECIDE**).

> **Đọc [BASELINE.md](BASELINE.md) trước** để xem kiến trúc mục tiêu FA2026 (Camera IMX219 160° + 2D LiDAR + ROS Melodic) và mức độ tương thích của codebase baseline hiện tại.

## Tài liệu dự án

| Tài liệu | Nội dung |
|---|---|
| [BASELINE.md](BASELINE.md) | Tổng quan FA2026, kiến trúc ROS mục tiêu, trạng thái code hiện tại |
| [docs/de-bai-rut-gon.md](docs/de-bai-rut-gon.md) | Checklist yêu cầu rút từ Thể lệ FA2026 (Speed Track, SEE/SENSE/DECIDE, Rosbag, Paper) |
| [docs/ke-hoach-thuc-nghiem.md](docs/ke-hoach-thuc-nghiem.md) | Kế hoạch thực nghiệm RQ1–RQ3, E1–E4, metric và lịch trình Nov–Dec 2026 |
| [docs/quy-trinh-test-xe.md](docs/quy-trinh-test-xe.md) | Quy trình bring-up xe từng bước (an toàn → rủi ro) |
| [docs/quy-trinh-thu-data-sa-ban.md](docs/quy-trinh-thu-data-sa-ban.md) | Quy trình hiệu chuẩn shading và thu dữ liệu bám làn trên sa bàn |
| [docs/reference/Thể_lệ_JetRacer_FA2026.docx](docs/reference/Thể_lệ_JetRacer_FA2026.docx) | Thể lệ chính thức Fall 2026 |
| [docs/reference/Cheemsity_JetRacer_FA2026_v1.pdf](docs/reference/Cheemsity_JetRacer_FA2026_v1.pdf) | Đề cương nghiên cứu / Technical Paper v1 |
| [docs/reference/HuongDanSuDungLidar.pdf](docs/reference/HuongDanSuDungLidar.pdf) | Hướng dẫn khởi động JetRacer, LiDAR, SLAM (ROS Melodic) & RViz |
| [Jetson.ipynb](Jetson.ipynb) | Notebook camera live, ghi dataset & lái tay cầm mới |

---

## Cài đặt & Thiết lập trên Jetson Nano (sau khi `git pull`)

**Trên Jetson Nano** (JetPack 4.5.1 / Ubuntu 18.04 / Python 3.6 / TensorRT 7.1.3) — OpenCV/numpy/tensorrt đã có sẵn trong JetPack, **không** `pip install` đè:

1. Cài gói còn thiếu (chỉ cần chạy lần đầu trên xe mới):
   ```bash
   pip3 install --user PyYAML
   # Nếu chưa có pycuda (phải export biến CUDA, KHÔNG dùng sudo):
   export PATH=/usr/local/cuda/bin:$PATH
   export CPATH=/usr/local/cuda/include:$CPATH
   export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
   pip3 install --user --no-cache-dir pycuda
   ```
2. **Build 2 model Option B (`.onnx` → `.engine` FP16) trực tiếp trên xe** *(engine không chuyển giữa các máy được)*:
   ```bash
   python3 tools/setup_jetson_models.py
   ```
   *(Lệnh trên tự kiểm tra môi trường, build `models/lane_option_b_resse.engine` & `models/lane_option_b_fastscnn.engine`, rồi test nạp TensorRT + đo latency.)*
3. Kiểm tra dấu CTE trước khi hạ bánh (đẩy xe sang TRÁI → vạch trôi PHẢI → `cte` DƯƠNG):
   ```bash
   python3 tools/bench_lane_cnn.py --override configs/cnn_option_b.yaml --watch --source csi
   ```
4. Mở giao diện Speed Track trên JupyterLab ([tune_lane.ipynb](tune_lane.ipynb)):
   ```bash
   jupyter lab --ip=0.0.0.0 --no-browser
   ```
   Bấm **1. MO CAMERA**, đặt xe giữa làn và quan sát mặt đường xanh cùng tâm vàng. Khi hiện `MAT DUONG OK - BAM CHAY`, bấm **2. CHAY - BAM LINE**.

**Trên laptop** (dev + test + phân tích log):
```bash
pip install -r requirements.txt
```

---

## Khởi động LiDAR & SLAM (ROS Melodic trên Jetson)

Xem chi tiết tại [docs/reference/HuongDanSuDungLidar.pdf](docs/reference/HuongDanSuDungLidar.pdf):

```bash
cd ~/catkin_ws
source /opt/ros/melodic/setup.bash
source devel/setup.bash
roslaunch jetracer slam.launch
```

Kiểm tra các topic tối thiểu (`/scan`, `/tf`, `/map`, `/odom`, `/cmd_vel`):

```bash
rostopic list
```

---

## Chạy thử baseline offline (không cần xe)

```bash
python -m src.jetracer_baseline.cli replay --source synthetic --frames 200
python tests/test_smoke.py
python tools/check_py36.py
```

Chạy replay trên video đã quay ở sa bàn:

```bash
python -m src.jetracer_baseline.cli replay --source video --video data/lap1.mp4
```

---

## Bring-up & Chạy baseline bám làn trên xe

Quy trình đầy đủ ở [docs/quy-trinh-test-xe.md](docs/quy-trinh-test-xe.md).

1. Đặt Jetson ở chế độ nguồn 5W trước khi test actuator:
   ```bash
   sudo nvpmodel -m 1 && nvpmodel -q
   ```
2. Kiểm tra môi trường, camera CSI và driver:
   ```bash
   python3 tools/check_hardware.py --driver nvidia
   ```
3. Test servo khi **bánh xe đã kê khỏi mặt đất**:
   ```bash
   python3 tools/check_hardware.py --driver nvidia --actuator --wheels-are-lifted --steering-only
   ```
4. Hiệu chuẩn lens shading màu (trước khi tune ngưỡng màu):
   ```bash
   python3 tools/collect_dataset.py --mode video --source csi --session flatfield --out data/calib --seconds 15 --raw
   python tools/calib_shading.py --mode flatfield --source data/calib/flatfield_<ts>.avi --preview reports/shading.png
   ```
5. Tune bám vạch trên JupyterLab ([tune_lane.ipynb](tune_lane.ipynb)) hoặc thu data lái tay ([Jetson.ipynb](Jetson.ipynb) / [collect_drive.ipynb](collect_drive.ipynb)):
   ```bash
   jupyter lab --ip=0.0.0.0 --no-browser
   ```
6. Chạy baseline bám làn trên xe:
   ```bash
   python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia --override configs/tuned.yaml --record
   ```

> **Lưu ý kiểm tra phần cứng:** Nếu lỗi `Failed to create CaptureSession`, đóng mọi kernel dùng camera rồi chạy `sudo systemctl restart nvargus-daemon`. Nếu lỗi I²C, kiểm tra `sudo i2cdetect -y -r 1` (phải có địa chỉ `40`). Nếu bẻ lái làm tắt nguồn/reboot, chạy `python3 tools/diagnose_shutdown.py monitor` để xác minh sụt áp rail 5V / kẹt servo.

---

## Trạng thái phát triển theo FA2026

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| Bring-up phần cứng, camera CSI, shading, Python 3.6 check | **Sẵn sàng** | Dùng ngay cho Shared Lab đợt 1 |
| Bám làn cơ bản (CV / CNN TensorRT) + PID/Corner | **Baseline sẵn sàng** | Dùng cho Speed Track, chưa có LiDAR tránh vật cản |
| Camera calibration & Fisheye Undistortion (RQ1 / SEE) | **Cần triển khai** | Bắt buộc cho biển báo rìa ảnh IMX219 160° |
| Detector YOLOv5n TensorRT FP16 (SEE) | **Cần triển khai** | Hiện `signs.py` mới có `StubDetector` & `YoloDetector` (YOLOv8) |
| ROS nodes + LiDAR perception (`/scan`) + Local State | **Cần triển khai** | Xem hướng dẫn LiDAR + Table I trong Technical Paper |
| Observability-Aware Adaptive Fusion (RQ2 / SENSE) | **Cần triển khai** | 3-layer local map (LiDAR, camera low-obstacle, observability) |
| Conflict Resolution + Hybrid A\* Parking + Pure Pursuit (RQ3 / DECIDE) | **Cần triển khai** | `fsm.py` hiện chưa có xử lý xung đột & đỗ xe Ackermann |
| Rosbag logging & công cụ đánh giá FA2026 | **Cần triển khai** | `analyze_log.py` hiện đang dùng công thức chấm điểm cũ |
