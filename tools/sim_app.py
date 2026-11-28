# -*- coding: utf-8 -*-
"""Local ONNX Track Simulator + Web UI cho JetRacer FA2026 Speed Track.

Chay mo phong vong kin (closed-loop) ngay tren anh sa ban chinh thuc
`docs/reference/sa_ban_chinh_thuc.jpg` voi:
  - Camera goc nhin thu nhat (320x240) render tu sa ban theo dung hinh hoc BEV
    va dai mau thuc te cua camera CSI JetRacer.
  - Suy dien TRUC TIEP bang cac file `.onnx` trong thu muc `models/` (khong can Jetson).
  - Chi dung kenh duong chay (`Ch0`: drivable_road) — KHONG dung vien trang (`Ch1`), KHONG LiDAR.
  - So sanh truc tiep 3 bo dieu khien:
      1. `project_current`  : dung code hien tai cua project (`CnnLaneDetector` + PID + `CornerController`)
      2. `road_pure_pursuit`: chi dung `Ch0` (mat duong) -> truc giua do dai cung + Pure Pursuit + giam toc doc lap
      3. `road_ftg`         : chi dung `Ch0` (mat duong) -> 31 tia goc cuc + Follow-The-Gap (chong cat cua dao do)
  - Giao dien web hien thi dong thoi: sa ban top-down + khung tam nhin camera (FOV/BEV) +
    panel 2x2 giong het `tuning_ui.py` + bieu do + bang diem vong dua.

Cach chay:
  cd D:\\Competition\\Jetson_Race_SS2\\JetsonRacer
  .\\.venv_cuda\\Scripts\\python.exe tools\\sim_app.py --port 8766

Mo trinh duyet tai: http://localhost:8766
"""

from __future__ import print_function

import argparse
import base64
import json
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
    from urllib.parse import urlparse
except ImportError:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn
    from urlparse import urlparse

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TOOLS_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.jetracer_baseline.control.corner import CornerController  # noqa: E402
from src.jetracer_baseline.control.driver import servo_output_for  # noqa: E402
from src.jetracer_baseline.control.pid import PID  # noqa: E402
from src.jetracer_baseline.control.pure_pursuit import (  # noqa: E402
    RoadPurePursuitController,
    fill_small_holes,
)
from src.jetracer_baseline.perception.lane_cnn import (  # noqa: E402
    CnnLaneDetector,
    IN_H,
    IN_W,
    OnnxEngine,
    PROC_H,
    PROC_W,
    SEG_H,
    SEG_W,
    _build_warp,
)

DEFAULT_MAP = os.path.join(REPO_DIR, 'docs', 'reference', 'sa_ban_chinh_thuc.jpg')
DEFAULT_MODELS_DIR = os.path.join(REPO_DIR, 'models')
DEFAULT_HTML = os.path.join(TOOLS_DIR, 'sim_app.html')
DEFAULT_YAML_PATH = os.path.join(REPO_DIR, 'configs', 'default.yaml')


def load_fixed_default_yaml(yaml_path=DEFAULT_YAML_PATH):
    """Doc cac thong so co dinh tu configs/default.yaml (KHONG bao gio sua file nay)."""
    # Gia tri fallback khop chinh xac configs/default.yaml neu moi truong thieu PyYAML
    fixed = {
        'control_hz': 30,
        'proc_width': 320,
        'proc_height': 240,
        'steer_max': 0.60,
        'corner_steer_max': 0.60,
        'v_straight': 0.30,
        'v_corner': 0.12,
        'v_max': 0.35,
        'v_min': 0.10,
        'throttle_rise_rate': 0.80,
        'throttle_fall_rate': 3.00,
        'steering_gain': -0.65,
        'steering_offset': 0.0,
        'steering_output_min': -0.40,
        'steering_output_max': 0.40,
        'throttle_gain': 0.80,
        'pulse_width_range': [750, 2250],
    }
    if not os.path.isfile(yaml_path):
        return fixed
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        p = data.get('pipeline', {})
        c = data.get('control', {})
        d = c.get('driver', {})
        fixed['control_hz'] = int(p.get('control_hz', fixed['control_hz']))
        fixed['proc_width'] = int(p.get('proc_width', fixed['proc_width']))
        fixed['proc_height'] = int(p.get('proc_height', fixed['proc_height']))
        for k in ('steer_max', 'corner_steer_max', 'v_straight', 'v_corner',
                  'v_max', 'v_min', 'throttle_rise_rate', 'throttle_fall_rate'):
            if k in c and c[k] is not None:
                fixed[k] = float(c[k])
        for k in ('steering_gain', 'steering_offset', 'steering_output_min',
                  'steering_output_max', 'throttle_gain'):
            if k in d and d[k] is not None:
                fixed[k] = float(d[k])
        if 'pulse_width_range' in d and d['pulse_width_range']:
            fixed['pulse_width_range'] = [int(v) for v in d['pulse_width_range']]
    except Exception:
        # Parser nhe khong can PyYAML de doc cac khoa don trong default.yaml
        try:
            with open(yaml_path, 'r', encoding='utf-8') as fh:
                lines = fh.readlines()
            in_driver = False
            for raw in lines:
                line = raw.split('#', 1)[0].rstrip()
                if not line.strip():
                    continue
                if line.strip() == 'driver:':
                    in_driver = True
                    continue
                if not raw.startswith(' '):
                    in_driver = False
                if ':' not in line:
                    continue
                key, val = [s.strip() for s in line.split(':', 1)]
                if in_driver and key in ('steering_gain', 'steering_offset',
                                         'steering_output_min', 'steering_output_max',
                                         'throttle_gain'):
                    fixed[key] = float(val)
                elif not in_driver and key in ('v_straight', 'v_corner', 'v_max', 'v_min',
                                               'throttle_rise_rate', 'throttle_fall_rate',
                                               'steer_max', 'corner_steer_max'):
                    fixed[key] = float(val)
                elif key == 'control_hz':
                    fixed[key] = int(float(val))
        except Exception:
            pass
    return fixed


FIXED_DEFAULT = load_fixed_default_yaml()

ROI_TOP = 0.34
ROI_Y0 = int(PROC_H * ROI_TOP)
WARP_SRC = [[0.00, 0.92], [1.00, 0.92], [0.66, 0.42], [0.33, 0.42]]
WARP_DST_MARGIN = 0.32
M_BEV_STD = _build_warp(PROC_W, PROC_H, WARP_SRC, WARP_DST_MARGIN)

# Wide BEV rieng cho Road-Only Pure Pursuit & Follow-The-Gap: giu lai hai goc duoi
# (mep trong cua chu U ngay sat mui xe) thay vi bi hinh thang hep cat mat.
WIDE_BEV_W, WIDE_BEV_H = 320, 240
WIDE_MARGIN = 0.36
M_BEV_WIDE = _build_warp(
    WIDE_BEV_W, WIDE_BEV_H,
    [[0.00, 1.00], [1.00, 1.00], [0.74, 0.42], [0.26, 0.42]],
    WIDE_MARGIN,
)

# Thong tin uoc luong hieu nang Jetson Nano TRT FP16 cho cac model pho bien
MODEL_HINTS = {
    'lane_road_only_fast.onnx': {'label': 'LaneRoadOnly-Fast [PRUNED 1ch] (53K, 0 reg)', 'nano_fps': '30+ FPS'},
    'lane_road_only_resse.onnx': {'label': 'LaneRoadOnly-ResSE [PRUNED 1ch] (153K, 0 reg)', 'nano_fps': '25-30 FPS'},
    'lane_option_b_fast.onnx': {'label': 'LaneTiny-Fast 3ch (56K, 110 MFLOPs)', 'nano_fps': '30+ FPS'},
    'lane_option_b_resse.onnx': {'label': 'LaneTiny-ResSE 3ch (157K, 157 MFLOPs)', 'nano_fps': '25-30 FPS'},
    'lane_option_b_fastscnn.onnx': {'label': 'FastSCNN-Lane 3ch (331K, 368 MFLOPs)', 'nano_fps': '20-25 FPS'},
    'lane_tiny_fast.onnx': {'label': 'LaneTiny-Fast 1ch (legacy)', 'nano_fps': '30+ FPS'},
    'lane_tiny.onnx': {'label': 'LaneTiny 1ch (legacy)', 'nano_fps': '25-30 FPS'},
}


class DictConfig(object):
    """Drop-in thay Config de chay duoc ke ca khi moi truong khong cai PyYAML."""

    def __init__(self, data):
        self._data = data

    def get(self, dotted_key, default=None):
        node = self._data
        for part in dotted_key.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key, value):
        parts = dotted_key.split('.')
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def build_default_project_cfg(onnx_path, params, fixed=None):
    """Cau hinh trung khop 100% configs/default.yaml (CO DINH) + configs/cnn_option_b.yaml."""
    fx = FIXED_DEFAULT if fixed is None else fixed
    return DictConfig({
        'lane': {
            'mode': 'cnn',
            'smooth_alpha': float(params.get('smooth_alpha', 0.60)),
            'cnn': {
                'backend': 'onnx',
                'engine': onnx_path,
                'roi_top': ROI_TOP,
                'warp_src': WARP_SRC,
                'warp_dst_margin': WARP_DST_MARGIN,
                'min_bands': 2,
                'band_min_pixels': 8,
                'lookahead': float(params.get('lookahead', 0.60)),
                'check_reg': True,
                'reg_disagree': 0.35,
                'single_line_offset_px': 74.0,
                'center_edge_margin_px': 32.0,
                'edge_guard_px': 42.0,
                'edge_guard_gain': 0.45,
            },
        },
        'control': {
            'pid': {
                'kp': float(params.get('kp', 1.10)),
                'ki': 0.00,
                'kd': float(params.get('kd', 0.12)),
                'out_limit': 1.0,
            },
            'steer_max': 1.00,
            'corner_steer_max': 1.00,
            'steer_lookahead_weight': float(params.get('steer_lookahead_weight', 0.70)),
            # Giu CO DINH dung configs/default.yaml
            'v_straight': float(fx['v_straight']),
            'v_corner': float(fx['v_corner']),
            'v_max': float(fx['v_max']),
            'v_min': float(fx['v_min']),
            'curve_enter': float(params.get('curve_enter', 0.12)),
            'curve_exit': float(params.get('curve_exit', 0.08)),
            'curve_feedforward': float(params.get('curve_feedforward', 1.80)),
            'corner_steer_gain': float(params.get('corner_steer_gain', 2.40)),
            'road_turn_steer_gain': 1.15,
            'road_turn_release_s': 0.30,
            'pre_corner_decel': 0.45,
            'corner_enter_delay_s': float(params.get('corner_enter_delay_s', 0.18)),
            'curvy_zone_enabled': True,
            'curvy_steer_boost': 1.20,
            'curvy_lookahead_scale': 0.65,
            'curvy_speed_scale': 0.88,
            'curvy_window_s': 6.0,
            'curvy_turn_count': 2,
            'curvy_exit_straight_s': 1.8,
            'slowdown': 0.20,
            'curve_brake': 0.00,
            'corner_hold_s': 0.70,
            'throttle_rise_rate': float(fx['throttle_rise_rate']),
            'throttle_fall_rate': float(fx['throttle_fall_rate']),
            'driver': {
                'steering_gain': -1.00,  # theo configs/cnn_option_b.yaml
                'steering_offset': float(fx['steering_offset']),
                'steering_output_min': float(fx['steering_output_min']),
                'steering_output_max': float(fx['steering_output_max']),
                'throttle_gain': 1.00,   # theo configs/cnn_option_b.yaml
            },
        },
    })


# =============================================================================
# 1. TRACK WORLD & SYNTHETIC FIRST-PERSON CAMERA
# =============================================================================

class TrackWorld(object):
    """Chua ban do sa ban chinh thuc, mat na ground-truth va renderer camera gia lap."""

    # Map px tren 1 BEV px cua BEV 320x240 chuan:
    # - Be rong lan tren sa_ban_chinh_thuc.jpg = 60 px
    # - Be rong lan trong BEV 320x240 do tu 1039 frame that = 158 px
    SCALE_X = 60.0 / 158.0
    SCALE_Y = 60.0 / 115.0
    CAM_NEAR_OFFSET = 14.0     # khoang cach tu tam truc sau / xe den mep day camera (map px)
    HORIZON_Y = 86             # duong chan troi trong anh 320x240 (khop roi_top=0.34 -> y=81)

    # Vi tri xuat phat ngay chu START (huong sang trai theta = pi)
    START_X = 745.0
    START_Y = 179.0
    START_THETA = math.pi
    START_LINE_X = 795.0       # vach trang START

    def __init__(self, map_path):
        self.map_path = os.path.abspath(map_path)
        raw = cv2.imread(self.map_path)
        if raw is None:
            raise IOError('Khong doc duoc anh sa ban: %s' % self.map_path)
        self.raw_bgr = raw
        self.h, self.w = raw.shape[:2]

        # Tach vung ngu nghia chinh xac cua sa_ban_chinh_thuc.jpg
        in_box = np.zeros((self.h, self.w), dtype=bool)
        in_box[111:610, 22:1010] = True
        b, g, r = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
        is_red = in_box & (r > 130) & (g < 80) & (b < 80)
        is_black_raw = in_box & (r < 60) & (g < 60) & (b < 60)

        # Lap chu START / mui ten ben trong lan den de tao ground-truth drivable road sach
        road_closed = cv2.morphologyEx(
            is_black_raw.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        )
        _, lbl, stats, _ = cv2.connectedComponentsWithStats(road_closed, connectivity=8)
        best_cc = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        self.road_gt = (lbl == best_cc).astype(np.uint8) * 255
        self.road_dist = cv2.distanceTransform(self.road_gt, cv2.DIST_L2, 5)

        is_white = in_box & (r > 180) & (g > 180) & (b > 180) & (~(self.road_gt > 0))
        is_start_mark = (self.road_gt > 0) & (r > 150) & (g > 150) & (b > 150)
        self.white_gt = is_white.astype(np.uint8) * 255
        self.red_gt = is_red.astype(np.uint8) * 255

        # Render canvas phoi mau khop camera CSI JetRacer thuc te (da xac minh IoU ~0.945 tren ca 3 ONNX)
        rng = np.random.RandomState(42)
        grain = rng.normal(0.0, 3.8, (self.h, self.w, 1)).astype(np.float32)
        canvas = np.full((self.h, self.w, 3), (155.0, 150.0, 142.0), dtype=np.float32)
        canvas[is_red] = (72.0, 74.0, 152.0)
        canvas[is_white] = (228.0, 225.0, 218.0)
        canvas[self.road_gt > 0] = (62.0, 60.0, 58.0)
        canvas[is_start_mark] = (220.0, 216.0, 208.0)
        canvas = np.clip(canvas + grain, 0.0, 255.0).astype(np.uint8)
        self.cam_canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)

        # Quan ly vat can tinh tren sa ban (duoc ve truc tiep len camera gia lap -> ONNX tu phat hien lo khuyet Ch0)
        self.obstacles = []
        self.cam_canvas_obs = self.cam_canvas
        self.road_gt_obs = self.road_gt

        # Nen phong lap phia tren duong chan troi (gradient xam nhe giong anh dataset)
        self._bg_top = np.zeros((self.HORIZON_Y, PROC_W, 3), dtype=np.uint8)
        for y in range(self.HORIZON_Y):
            t = y / float(max(1, self.HORIZON_Y - 1))
            self._bg_top[y, :] = (int(115 + 22 * t), int(116 + 20 * t), int(104 + 18 * t))

        # Tao danh sach checkpoint doc tam vong dua de do % hoan thanh vong chinh xac
        self.checkpoints = [
            (600, 179), (380, 179), (170, 179),
            (93, 290), (93, 460), (165, 555),
            (250, 420), (355, 295), (455, 420),
            (528, 555), (602, 420), (685, 325),
            (780, 440), (855, 520), (938, 400),
            (938, 250), (840, 179),
        ]

    @staticmethod
    def sample_obstacles():
        """3 vat can tinh mau dat lech 1 ben lan (van con ~36px duong trong de xe 21px lach qua)."""
        return [
            {'x': 520.0, 'y': 163.0, 'r': 12.0},
            {'x': 108.0, 'y': 365.0, 'r': 12.0},
            {'x': 442.0, 'y': 420.0, 'r': 12.0},
        ]

    def set_obstacles(self, obs_list):
        cleaned = []
        for o in (obs_list or []):
            x = float(o.get('x', 0.0))
            y = float(o.get('y', 0.0))
            r = max(6.0, min(24.0, float(o.get('r', 12.0))))
            cleaned.append({'x': round(x, 1), 'y': round(y, 1), 'r': round(r, 1)})
        self.obstacles = cleaned
        if not cleaned:
            self.cam_canvas_obs = self.cam_canvas
            self.road_gt_obs = self.road_gt
            return
        canvas = self.cam_canvas.copy()
        rgt = self.road_gt.copy()
        for o in cleaned:
            cx, cy, rad = int(round(o['x'])), int(round(o['y'])), int(round(o['r']))
            # Ve khoi vat can mau cam-do + soc trang noi bat (khac han mat duong den -> CNN Ch0 se cat lo!)
            cv2.circle(canvas, (cx, cy), rad + 2, (228, 224, 218), -1)
            cv2.circle(canvas, (cx, cy), max(2, rad - 1), (55, 95, 215), -1)
            cv2.circle(canvas, (cx, cy), max(1, rad // 2), (235, 232, 225), -1)
            cv2.circle(rgt, (cx, cy), rad + 1, 0, -1)
        self.cam_canvas_obs = cv2.GaussianBlur(canvas, (3, 3), 0.4)
        self.road_gt_obs = rgt

    def _bev_to_map_matrix(self, car_x, car_y, car_theta, cam_near_offset=None):
        near = self.CAM_NEAR_OFFSET if cam_near_offset is None else float(cam_near_offset)
        c, s = math.cos(car_theta), math.sin(car_theta)
        sx, sy = self.SCALE_X, self.SCALE_Y
        return np.array([
            [-s * sx, -c * sy, car_x + c * (PROC_H * sy + near) + s * (PROC_W * 0.5 * sx)],
            [c * sx, -s * sy, car_y + s * (PROC_H * sy + near) - c * (PROC_W * 0.5 * sx)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def render_camera(self, car_x, car_y, car_theta, cam_near_offset=None,
                      glare=0.0, shadow=0.0):
        """Render frame camera 320x240 BGR + GT road mask + da giac FOV/BEV tren ban do."""
        A_bev2map = self._bev_to_map_matrix(car_x, car_y, car_theta, cam_near_offset)
        H_cam2map = A_bev2map.dot(M_BEV_STD)

        cam = cv2.warpPerspective(
            self.cam_canvas_obs, H_cam2map, (PROC_W, PROC_H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(155, 150, 142),
        )
        gt_cam = cv2.warpPerspective(
            self.road_gt_obs, H_cam2map, (PROC_W, PROC_H),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        cam[:self.HORIZON_Y, :] = self._bg_top
        gt_cam[:self.HORIZON_Y, :] = 0

        # Tuy chon gia lap bong do / chói den tran de stress-test ONNX
        if shadow > 0.01 or glare > 0.01:
            f = cam.astype(np.float32)
            if shadow > 0.01:
                xs = np.linspace(0.0, 1.0, PROC_W, dtype=np.float32)[None, :, None]
                # Toi dan ve mot phia
                grad = (1.0 - 0.55 * float(shadow)) + (0.55 * float(shadow)) * xs
                f[self.HORIZON_Y:] *= grad
            if glare > 0.01:
                yy, xx = np.mgrid[0:PROC_H, 0:PROC_W].astype(np.float32)
                blob = np.exp(-(((xx - PROC_W * 0.52) / 45.0) ** 2 + ((yy - PROC_H * 0.62) / 28.0) ** 2))
                f += (blob[:, :, None] * (120.0 * float(glare)))
            cam = np.clip(f, 0.0, 255.0).astype(np.uint8)

        # 4 goc hinh thang tam nhin camera (ROI duoi duong chan troi) tren ban do
        cam_roi_pts = np.float32([
            [0.0, PROC_H - 1.0],
            [PROC_W - 1.0, PROC_H - 1.0],
            [PROC_W - 1.0, PROC_H * 0.44],
            [0.0, PROC_H * 0.44],
        ]).reshape(-1, 1, 2)
        fov_poly = cv2.perspectiveTransform(cam_roi_pts, H_cam2map).reshape(-1, 2)

        # 4 goc hinh thang BEV chuan (vung model that su warp vao BEV)
        bev_pts = np.float32([
            [p[0] * PROC_W, p[1] * PROC_H] for p in WARP_SRC
        ]).reshape(-1, 1, 2)
        bev_poly = cv2.perspectiveTransform(bev_pts, H_cam2map).reshape(-1, 2)

        return cam, gt_cam, fov_poly, bev_poly, A_bev2map, H_cam2map

    def check_car_footprint(self, car_x, car_y, car_theta, car_len, car_wid):
        """Kiem tra 4 goc than xe + tam xe xem co cham vach trang / ra ngoai duong den khong."""
        c, s = math.cos(car_theta), math.sin(car_theta)
        hl, hw = car_len * 0.5, car_wid * 0.5
        corners = []
        min_clearance = 999.0
        on_road_all = True
        in_red_any = False
        for dx, dy in ((hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)):
            px = car_x + c * dx - s * dy
            py = car_y + s * dx + c * dy
            corners.append([round(px, 1), round(py, 1)])
            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= self.w or iy < 0 or iy >= self.h:
                on_road_all = False
                in_red_any = True
                min_clearance = min(min_clearance, -10.0)
            else:
                d = float(self.road_dist[iy, ix])
                if d <= 0.5:
                    on_road_all = False
                if self.red_gt[iy, ix] > 0:
                    in_red_any = True
                min_clearance = min(min_clearance, d)
        icx, icy = int(round(car_x)), int(round(car_y))
        center_on_road = (0 <= icx < self.w and 0 <= icy < self.h and self.road_gt[icy, icx] > 0)
        # Kiem tra va cham voi cac vat can tinh tren duong (4 goc + trung diem 4 canh + tam xe)
        collided_obs = False
        min_obs_clear = 999.0
        if self.obstacles:
            sample_pts = list(corners) + [[car_x, car_y]]
            for k in range(4):
                a, b = corners[k], corners[(k + 1) % 4]
                sample_pts.append([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5])
            for o in self.obstacles:
                ox, oy, orad = o['x'], o['y'], o['r']
                for px, py in sample_pts:
                    d_obs = math.hypot(px - ox, py - oy) - orad
                    if d_obs < min_obs_clear:
                        min_obs_clear = d_obs
                    if d_obs <= 0.0:
                        collided_obs = True
        return {
            'corners': corners,
            'on_road_all': bool(on_road_all),
            'center_on_road': bool(center_on_road),
            'in_red_any': bool(in_red_any),
            'min_clearance_px': round(min_clearance, 1),
            'collided_obs': bool(collided_obs),
            'min_obs_clear_px': round(min_obs_clear, 1) if min_obs_clear < 900 else None,
        }


# =============================================================================
# 2. ROAD-ONLY CONTROLLERS (CHI DUNG CH0 DRIVABLE ROAD - KHONG DUNG CH1 WHITE, KHONG LIDAR)
# =============================================================================

class RoadOnlyPurePursuit(object):
    """Dieu khien chi dung Ch0 (drivable_road) tu ONNX:
      1. Tach vung ego_road (giu tinh lien tuc theo frame truoc -> khong nhay sang nhanh ben kia dao do).
      2. Tinh distance transform D tu vung ngoai duong trong tam nhin (khong tinh bien camera la tuong).
      3. Lan theo truc tam do dai cung (medial ridge) -> om tron cua 180 do khong bi vo parabol x(y).
      4. Tinh goc lai Ackermann Pure Pursuit o khoang cach cung L_d ngan (khong cat day cung dao do).
      5. Dung do cong xa CHI de giam toc truoc cua, KHONG danh lai som!
    """

    def __init__(self):
        self._seg_to_proc = np.array([
            [PROC_W / float(SEG_W), 0.0, 0.0],
            [0.0, (PROC_H - ROI_Y0) / float(SEG_H), float(ROI_Y0)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.M_seg = M_BEV_STD.dot(self._seg_to_proc)
        valid = cv2.warpPerspective(
            np.ones((SEG_H, SEG_W), np.uint8), self.M_seg,
            (PROC_W, PROC_H), flags=cv2.INTER_NEAREST,
        )
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.bev_visible = cv2.erode(valid, ker, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.reset()

    def reset(self):
        self._prev_steer = 0.0
        self._throttle = 0.0
        self._last_good_side = 0.0
        self._last_steer_dir = 0.0
        self.debug_bev_pts = []
        self.debug_target = None
        self.debug_road_bev = None
        self.debug_mode = 'THANG'
        self.debug_far_angle = 0.0

    def _extract_ego_road(self, road_bev):
        n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(road_bev, connectivity=8)
        if n_cc <= 1:
            return None, False
        half = PROC_W // 2
        # Uu tien thanh phan nam ngay truoc mui xe (day anh BEV)
        near = lbl[PROC_H - 34:PROC_H, half - 28:half + 28].ravel()
        near = near[near > 0]
        if len(near) >= 40:
            cid = int(np.bincount(near).argmax())
            ego = (lbl == cid).astype(np.uint8)
            # Lap lo ben trong (vd chu START)
            cnts = cv2.findContours(ego, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if cnts:
                cv2.drawContours(ego, cnts, -1, 1, cv2.FILLED)
            return ego, True

        # Xe da lech ra mep: chon nhanh duong o day anh gan phia cu nhat, tranh nhay qua dao do
        wide = lbl[PROC_H - 64:PROC_H, :]
        counts = np.bincount(wide.ravel())
        counts[0] = 0
        cands = np.flatnonzero(counts >= 100)
        if len(cands) == 0:
            return None, False
        best_cid, best_score = None, 1e9
        for cid in cands:
            xs = np.where(wide == cid)[1]
            cx = float(xs.mean())
            d = abs(cx - half)
            if d < best_score:
                best_score = d
                best_cid = int(cid)
        ego = (lbl == best_cid).astype(np.uint8)
        cnts = cv2.findContours(ego, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if cnts:
            cv2.drawContours(ego, cnts, -1, 1, cv2.FILLED)
        return ego, False

    def _trace_medial_path(self, ego_road, half_lane_px=76.0):
        """Lan theo truc giua tu mui xe theo tung buoc cung ds = 8 px (xu ly duoc cua 180 do!)."""
        outside = self.bev_visible & (ego_road == 0)
        if np.count_nonzero(outside) < 20:
            # Duong den phu kin man hinh -> di thang
            return [(PROC_W * 0.5, float(PROC_H - 4 - k * 10)) for k in range(18)], None

        dist = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 5)
        # Loai bo vung ngoai ego_road
        dist_in = np.where(ego_road > 0, dist, -1.0)

        # Diem xuat phat: diem gan mui xe nhat nam trong ego_road
        half = PROC_W * 0.5
        y_start = PROC_H - 12
        row_xs = np.flatnonzero(ego_road[y_start, :] > 0)
        if len(row_xs) == 0:
            ys_all, xs_all = np.nonzero(ego_road)
            if len(ys_all) == 0:
                return [], dist
            idx = int(np.argmin((xs_all - half) ** 2 + (ys_all - PROC_H) ** 2))
            cur_x, cur_y = float(xs_all[idx]), float(ys_all[idx])
        else:
            # Neu ca 2 mep deu thay -> lay giua; neu chi thay 1 mep -> uu tien khoang cach = half_lane_px
            vals = dist_in[y_start, row_xs]
            if vals.max() > half_lane_px + 6.0:
                # 1 mep: diem gan `half_lane_px` nhat
                score = np.abs(vals - half_lane_px) + 0.15 * np.abs(row_xs - half)
                cur_x = float(row_xs[int(np.argmin(score))])
            else:
                # 2 mep: dinh distance transform
                cur_x = float(row_xs[int(np.argmax(vals))])
            cur_y = float(y_start)

        path = [(cur_x, cur_y)]
        # Huong di chuyen ban dau: thang len tren (-pi/2 trong toa do anh BEV)
        heading = -math.pi * 0.5
        ds = 10.0
        # Quet cung tron truoc mat [-78 deg .. +78 deg] quanh huong hien tai
        rel_angles = np.linspace(-1.35, 1.35, 29)

        for step in range(24):
            best_score = -1e9
            best_pt = None
            best_ang = heading
            for da in rel_angles:
                ang = heading + da
                nx = cur_x + ds * math.cos(ang)
                ny = cur_y + ds * math.sin(ang)
                ix, iy = int(round(nx)), int(round(ny))
                if ix < 4 or ix >= PROC_W - 4 or iy < 4 or iy >= PROC_H - 4:
                    continue
                if ego_road[iy, ix] == 0:
                    continue
                d_wall = float(dist[iy, ix])
                # Uu tien bam dung khoang cach = half_lane_px (khi 1 mep) va toi da hoa d_wall (khi 2 mep)
                if d_wall <= half_lane_px:
                    clearance_score = d_wall
                else:
                    # Khong bi hut qua xa vao vung mo (khong co tuong ben kia) -> giu sat khoang cach half_lane_px!
                    clearance_score = half_lane_px - 0.85 * (d_wall - half_lane_px)
                # Phat quay gat dot ngot de duong muot
                smooth_pen = 14.0 * (da * da)
                # Phat di nguoc xuong day anh
                back_pen = 35.0 if ny > cur_y + 2.0 else 0.0
                sc = clearance_score - smooth_pen - back_pen
                if sc > best_score:
                    best_score = sc
                    best_pt = (nx, ny)
                    best_ang = ang
            if best_pt is None:
                break
            # Giam cham cap nhat heading de tranh rung
            heading = 0.35 * heading + 0.65 * best_ang
            cur_x, cur_y = best_pt
            path.append((cur_x, cur_y))
            if cur_y <= 14.0:
                break
        return path, dist

    def step(self, road_logits, dt, params, fixed):
        road_bev = cv2.warpPerspective(
            (road_logits > 0.0).astype(np.uint8), self.M_seg,
            (PROC_W, PROC_H), flags=cv2.INTER_NEAREST,
        )
        ego_road, on_center = self._extract_ego_road(road_bev)
        self.debug_road_bev = ego_road

        # Doc dung gioi han toc do & servo CO DINH tu configs/default.yaml
        v_straight = float(fixed['v_straight'])
        v_corner = float(fixed['v_corner'])
        v_min = float(fixed['v_min'])
        rise_rate = float(fixed['throttle_rise_rate'])
        fall_rate = float(fixed['throttle_fall_rate'])
        servo_lim = float(fixed['steering_output_max'])
        steer_gain_abs = max(1e-6, abs(float(fixed['steering_gain'])))
        # Lenh lai tuyen tinh toi da khop dung tran servo default.yaml (±0.40 / 0.65 = ±0.615, kẹp theo steer_max)
        steer_span = min(float(fixed['corner_steer_max']), servo_lim / steer_gain_abs)

        ld_px = float(params.get('pp_lookahead_px', 84.0))
        pp_gain = float(params.get('pp_gain', 1.15))
        half_lane_px = float(params.get('half_lane_px', 82.0))
        wheel_at_lim_rad = math.radians(float(params.get('wheel_deg_at_limit', 34.0)))

        if ego_road is None:
            # Mat duong: giu huong lai cu nhe + giam ga ve v_min
            steer = float(np.clip(self._prev_steer * 0.9, -steer_span, steer_span))
            self._throttle = max(v_min, self._throttle - fall_rate * dt)
            self.debug_bev_pts = []
            self.debug_target = None
            self.debug_mode = 'MAT'
            return {
                'found': False, 'cte': 0.0, 'lookahead': 0.0, 'curvature': 0.0,
                'steer': steer, 'throttle': self._throttle, 'mode': 'CUA',
                'n_bands': 0,
            }

        path, dist = self._trace_medial_path(ego_road, half_lane_px=half_lane_px)
        self.debug_bev_pts = path
        if len(path) < 2:
            ys, xs = np.nonzero(ego_road)
            cx = float(np.median(xs))
            cte = float(np.clip((cx - PROC_W * 0.5) / (PROC_W * 0.5), -1.0, 1.0))
            steer = float(np.clip(cte * 1.2, -1.0, 1.0)) * steer_span
            self._prev_steer = steer
            self._throttle = v_corner
            self.debug_target = (cx, float(np.median(ys)))
            self.debug_mode = 'CUA'
            return {
                'found': True, 'cte': cte, 'lookahead': cte, 'curvature': 0.0,
                'steer': steer, 'throttle': v_corner, 'mode': 'CUA', 'n_bands': 1,
            }

        pts = np.array(path, dtype=np.float64)
        seg_d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        cum_s = np.concatenate([[0.0], np.cumsum(seg_d)])
        total_s = float(cum_s[-1])

        # Goc cua phia xa CHI dung cho giam toc truoc cua & rut ngan L_d, KHONG cong vao lai!
        far_idx = int(np.searchsorted(cum_s, min(total_s, max(ld_px * 1.6, 150.0))))
        far_idx = min(far_idx, len(pts) - 1)
        far_dx = pts[far_idx, 0] - PROC_W * 0.5
        far_dy = max(12.0, (PROC_H + 18.0) - pts[far_idx, 1])
        far_angle = math.atan2(far_dx, far_dy)
        self.debug_far_angle = far_angle
        curv_proxy = float(np.clip(far_angle / 0.85, -1.0, 1.0))

        eff_ld = ld_px * (1.0 - 0.30 * min(1.0, abs(curv_proxy)))
        eff_ld = max(48.0, min(total_s, eff_ld))
        t_idx = int(np.searchsorted(cum_s, eff_ld))
        t_idx = min(t_idx, len(pts) - 1)
        tx, ty = float(pts[t_idx, 0]), float(pts[t_idx, 1])
        self.debug_target = (tx, ty)

        # Pure Pursuit tu vi tri truc xe -> goc banh mong muon delta (rad)
        cam_off_bev = TrackWorld.CAM_NEAR_OFFSET / TrackWorld.SCALE_Y
        dx_car = tx - PROC_W * 0.5
        dy_car = (PROC_H + cam_off_bev) - ty
        dx_m = dx_car * TrackWorld.SCALE_X
        dy_m = dy_car * TrackWorld.SCALE_Y
        dist_m = max(10.0, math.hypot(dx_m, dy_m))
        alpha = math.atan2(dx_m, dy_m)
        wb_m = float(params.get('car_length', 76.0)) * 0.62
        delta = math.atan2(2.0 * wb_m * math.sin(alpha), dist_m) * pp_gain

        # Dem an toan mep: neu mui xe qua gan tuong 1 ben thi day nhe ra giua
        if dist is not None:
            outside = self.bev_visible & (ego_road == 0)
            left_out = outside[PROC_H - 42:PROC_H - 6, :PROC_W // 2]
            right_out = outside[PROC_H - 42:PROC_H - 6, PROC_W // 2:]
            has_l = left_out.any(axis=1)
            has_r = right_out.any(axis=1)
            d_l = float(np.median(left_out[has_l, ::-1].argmax(axis=1) + 1)) if has_l.any() else 999.0
            d_r = float(np.median(right_out[has_r].argmax(axis=1) + 1)) if has_r.any() else 999.0
            margin = 54.0
            if d_r < margin and d_r + 6.0 < d_l:
                delta -= 0.45 * ((margin - d_r) / margin) * wheel_at_lim_rad
            elif d_l < margin and d_l + 6.0 < d_r:
                delta += 0.45 * ((margin - d_l) / margin) * wheel_at_lim_rad

        # Anh xa TUYEN TINH dung vao dai [-steer_span, +steer_span] de khong bi vo vanh/bao hoa som o ±0.40!
        u_norm = float(np.clip(delta / max(0.15, wheel_at_lim_rad), -1.0, 1.0))
        steer = u_norm * steer_span
        steer = 0.72 * steer + 0.28 * self._prev_steer
        self._prev_steer = steer

        # TACH DOC LAP TOC DO: dung dung v_straight / v_corner / rise / fall cua default.yaml
        corner_intensity = max(abs(curv_proxy), abs(u_norm))
        mode = 'CUA' if corner_intensity >= 0.26 else 'THANG'
        target_v = v_straight - (v_straight - v_corner) * min(1.0, corner_intensity * 1.35)
        target_v = max(v_corner, min(v_straight, target_v))
        rate = rise_rate if target_v > self._throttle else fall_rate
        dv = target_v - self._throttle
        self._throttle += max(-rate * dt, min(rate * dt, dv))

        cte = float(np.clip((pts[0, 0] - PROC_W * 0.5) / (PROC_W * 0.5), -1.0, 1.0))
        look = float(np.clip((tx - PROC_W * 0.5) / (PROC_W * 0.5), -1.0, 1.0))
        self.debug_mode = mode
        return {
            'found': True,
            'cte': cte,
            'lookahead': look,
            'curvature': curv_proxy,
            'steer': steer,
            'throttle': self._throttle,
            'mode': mode,
            'n_bands': min(10, len(pts)),
        }


class RoadOnlyFollowTheGap(object):
    """Dieu khien chi dung Ch0 (drivable_road) theo thuat toan F1TENTH Follow-The-Gap:
      - Ban 31 tia tu mui xe trong BEV ra toi ranh gioi ngoai duong (actual outside).
      - Tao vung dem (safety bubble) quanh goc tuong gan nhat (chinh la mui dao do!).
      - Chon huong trong khe trong rong & sau nhat -> tu dong tranh cat cua dao do!
    """

    N_RAYS = 31

    def __init__(self):
        self._seg_to_proc = np.array([
            [PROC_W / float(SEG_W), 0.0, 0.0],
            [0.0, (PROC_H - ROI_Y0) / float(SEG_H), float(ROI_Y0)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.M_seg = M_BEV_STD.dot(self._seg_to_proc)
        valid = cv2.warpPerspective(
            np.ones((SEG_H, SEG_W), np.uint8), self.M_seg,
            (PROC_W, PROC_H), flags=cv2.INTER_NEAREST,
        )
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.bev_visible = cv2.erode(valid, ker, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.angles = np.linspace(-math.radians(72.0), math.radians(72.0), self.N_RAYS)
        self.reset()

    def reset(self):
        self._prev_steer = 0.0
        self._throttle = 0.0
        self.debug_rays = []
        self.debug_best_ray = None
        self.debug_road_bev = None

    def step(self, road_logits, dt, params, fixed):
        road_bev = cv2.warpPerspective(
            (road_logits > 0.0).astype(np.uint8), self.M_seg,
            (PROC_W, PROC_H), flags=cv2.INTER_NEAREST,
        )
        # Giu ego_road
        n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(road_bev, connectivity=8)
        ego_road = None
        if n_cc > 1:
            half = PROC_W // 2
            near = lbl[PROC_H - 42:PROC_H, half - 36:half + 36].ravel()
            near = near[near > 0]
            if len(near) >= 30:
                cid = int(np.bincount(near).argmax())
                ego_road = (lbl == cid).astype(np.uint8)
            else:
                wide = lbl[PROC_H - 72:PROC_H, :]
                cnt = np.bincount(wide.ravel())
                cnt[0] = 0
                if cnt.max() >= 80:
                    ego_road = (lbl == int(cnt.argmax())).astype(np.uint8)
        if ego_road is not None:
            ego_road = fill_small_holes(ego_road, max_hole_area=180)
        self.debug_road_bev = ego_road

        v_straight = float(fixed['v_straight'])
        v_corner = float(fixed['v_corner'])
        v_min = float(fixed['v_min'])
        rise_rate = float(fixed['throttle_rise_rate'])
        fall_rate = float(fixed['throttle_fall_rate'])
        servo_lim = float(fixed['steering_output_max'])
        steer_gain_abs = max(1e-6, abs(float(fixed['steering_gain'])))
        steer_span = min(float(fixed['corner_steer_max']), servo_lim / steer_gain_abs)

        ftg_gain = float(params.get('ftg_gain', 1.25))
        bubble_rays = int(params.get('ftg_bubble', 5))

        if ego_road is None:
            self.debug_rays = []
            self.debug_best_ray = None
            steer = float(np.clip(self._prev_steer * 0.9, -steer_span, steer_span))
            self._throttle = max(v_min, self._throttle - fall_rate * dt)
            return {
                'found': False, 'cte': 0.0, 'lookahead': 0.0, 'curvature': 0.0,
                'steer': steer, 'throttle': self._throttle, 'mode': 'CUA', 'n_bands': 0,
            }

        outside = self.bev_visible & (ego_road == 0)
        ox, oy = float(PROC_W // 2), float(PROC_H - 4)
        max_r = 225.0
        dists = np.zeros(self.N_RAYS, dtype=np.float64)
        hit_wall = np.zeros(self.N_RAYS, dtype=bool)
        rays_vis = []

        for idx, ang in enumerate(self.angles):
            dx = math.sin(ang)
            dy = -math.cos(ang)
            hit_r = max_r
            is_wall = False
            for r in range(8, int(max_r), 3):
                px = int(round(ox + dx * r))
                py = int(round(oy + dy * r))
                if px < 0 or px >= PROC_W or py < 0 or py >= PROC_H:
                    hit_r = float(r)
                    break
                if outside[py, px]:
                    hit_r = float(r)
                    is_wall = True
                    break
                if not self.bev_visible[py, px]:
                    hit_r = float(r)
                    break
            dists[idx] = hit_r
            hit_wall[idx] = is_wall
            rays_vis.append((ox + dx * hit_r, oy + dy * hit_r, hit_r, is_wall))

        # Safety bubble quanh diem cham tuong gan nhat (tranh cat vao mui dao do!)
        safe = dists.copy()
        wall_idxs = np.flatnonzero(hit_wall)
        if len(wall_idxs) > 0:
            nearest_i = int(wall_idxs[int(np.argmin(dists[wall_idxs]))])
            if dists[nearest_i] < 130.0:
                lo = max(0, nearest_i - bubble_rays)
                hi = min(self.N_RAYS, nearest_i + bubble_rays + 1)
                safe[lo:hi] = np.minimum(safe[lo:hi], dists[nearest_i] * 0.25)

        k = np.array([0.15, 0.25, 0.20, 0.25, 0.15], dtype=np.float64)
        smooth_r = np.convolve(np.pad(safe, 2, mode='edge'), k, mode='valid')
        weights = np.maximum(0.0, smooth_r - 35.0) ** 2.0
        if weights.sum() < 1e-6:
            best_ang = 0.0
            best_i = self.N_RAYS // 2
        else:
            thr = float(smooth_r.max()) * 0.85
            top_mask = smooth_r >= thr
            w_top = weights * top_mask
            best_ang = float(np.sum(self.angles * w_top) / max(1e-6, float(w_top.sum())))
            best_i = int(np.argmin(np.abs(self.angles - best_ang)))

        self.debug_rays = rays_vis
        self.debug_best_ray = (
            ox + math.sin(best_ang) * dists[best_i],
            oy - math.cos(best_ang) * dists[best_i],
        )

        # Anh xa tuyen tinh vao dung dai [-steer_span, +steer_span] cua default.yaml
        u_norm = float(np.clip((best_ang / math.radians(42.0)) * ftg_gain, -1.0, 1.0))
        steer = u_norm * steer_span
        steer = 0.70 * steer + 0.30 * self._prev_steer
        self._prev_steer = steer

        fwd_depth = float(np.median(dists[self.N_RAYS // 2 - 2:self.N_RAYS // 2 + 3]))
        depth_factor = float(np.clip((fwd_depth - 85.0) / 115.0, 0.0, 1.0))
        turn_factor = 1.0 - min(1.0, abs(best_ang) / math.radians(38.0))
        speed_frac = min(depth_factor, turn_factor)
        target_v = v_corner + (v_straight - v_corner) * speed_frac
        rate = rise_rate if target_v > self._throttle else fall_rate
        dv = target_v - self._throttle
        self._throttle += max(-rate * dt, min(rate * dt, dv))

        mode = 'CUA' if speed_frac < 0.55 else 'THANG'
        return {
            'found': True,
            'cte': 0.0,
            'lookahead': float(np.clip(best_ang / 1.0, -1.0, 1.0)),
            'curvature': float(np.clip(best_ang / 0.9, -1.0, 1.0)),
            'steer': steer,
            'throttle': self._throttle,
            'mode': mode,
            'n_bands': int(np.sum(safe > 50.0)),
        }


# =============================================================================
# 3. SIMULATOR ENGINE (ACKERMANN KINEMATICS + REAL ONNX + PROJECT-STYLE PANELS)
# =============================================================================

class SimulatorSession(object):
    def __init__(self, map_path, models_dir):
        self.lock = threading.Lock()
        self.world = TrackWorld(map_path)
        self.models_dir = os.path.abspath(models_dir)
        self.available_models = self._scan_models()
        if not self.available_models:
            raise RuntimeError('Khong tim thay file .onnx nao trong %s' % self.models_dir)

        self._engine_cache = {}
        self.model_name = self._pick_default_model()
        self.controller_name = 'road_pure_pursuit'

        self.fixed_default = load_fixed_default_yaml()
        self.params = {
            # Mac dinh toc do mo phong an toan (0.30 / 0.12) + mo khoa 100% servo cua chu U
            'v_straight': 0.30,
            'v_corner': 0.12,
            'servo_limit': 1.00,
            'pre_corner_decel': 0.50,
            'road_turn_steer_gain': 1.40,
            'car_length': 76.0,          # dung chieu dai chu START
            'car_width': 21.0,           # dung chieu rong chu START
            # Goc be banh vat ly toi da cua xe tai servo ±1.00 (100% tam lai)
            'wheel_deg_at_limit': 36.0,
            'speed_px_per_s': 260.0,     # toc do o throttle_out=1.0 (map px/s)
            'latency_frames': 1,         # do tre pipeline (so frame o 30 Hz)
            'glare': 0.0,
            'shadow': 0.0,
            # Pure Pursuit params
            'pp_lookahead_px': 78.0,
            'pp_gain': 1.20,
            'half_lane_px': 82.0,
            # Follow-The-Gap params
            'ftg_gain': 1.25,
            'ftg_bubble': 5,
            # Project Current params (override tu configs/cnn_option_b.yaml)
            'lookahead': 0.60,
            'steer_lookahead_weight': 0.70,
            'curve_enter': 0.12,
            'curve_exit': 0.08,
            'curve_feedforward': 1.80,
            'corner_steer_gain': 2.40,
            'corner_enter_delay_s': 0.18,
            'kp': 1.10,
            'kd': 0.12,
            'smooth_alpha': 0.60,
        }

        self.pp_ctrl = RoadPurePursuitController(build_default_project_cfg(
            self._get_model_path(self.model_name), self.params, self.fixed_default))
        # Giu dung steering_gain / throttle_gain CO DINH cua default.yaml cho RoadPurePursuitController
        self.pp_ctrl.drv_steer_gain = float(self.fixed_default['steering_gain'])
        self.pp_ctrl.drv_out_max = float(self.fixed_default['steering_output_max'])
        self.ftg_ctrl = RoadOnlyFollowTheGap()
        self._proj_det = None
        self._proj_pid = None
        self._proj_corner = None
        self._rebuild_project_stack()
        self.reset_car()

    def _scan_models(self):
        found = []
        for fn in sorted(os.listdir(self.models_dir)):
            if fn.endswith('.onnx'):
                full = os.path.join(self.models_dir, fn)
                hint = MODEL_HINTS.get(fn, {})
                found.append({
                    'name': fn,
                    'path': full,
                    'size_kb': round(os.path.getsize(full) / 1024.0, 1),
                    'label': hint.get('label', fn),
                    'nano_fps': hint.get('nano_fps', '20-30 FPS'),
                })
        return found

    def _pick_default_model(self):
        names = [m['name'] for m in self.available_models]
        for pref in ('lane_road_only_fast.onnx', 'lane_option_b_fast.onnx',
                     'lane_option_b_resse.onnx', 'lane_option_b_fastscnn.onnx'):
            if pref in names:
                return pref
        return names[0]

    def _get_model_path(self, name):
        for m in self.available_models:
            if m['name'] == name:
                return m['path']
        raise KeyError('Khong tim thay model: %s' % name)

    def _get_onnx_engine(self, name):
        if name not in self._engine_cache:
            self._engine_cache[name] = OnnxEngine(self._get_model_path(name))
        return self._engine_cache[name]

    def _rebuild_project_stack(self):
        mpath = self._get_model_path(self.model_name)
        cfg = build_default_project_cfg(mpath, self.params, self.fixed_default)
        self._proj_cfg = cfg
        # Tai su dung OnnxEngine de doi tham so tuc thi khong nap lai file ONNX
        if self._proj_det is None or self._proj_det.engine_path != mpath:
            self._proj_det = CnnLaneDetector(cfg)
            self._engine_cache[self.model_name] = self._proj_det.engine
        else:
            self._proj_det.update_config(cfg)
        self._proj_pid = PID(cfg.get('control.pid.kp'), 0.0, cfg.get('control.pid.kd'), out_limit=1.0)
        self._proj_corner = CornerController(cfg)

    def reset_car(self, x=None, y=None, theta=None):
        self.car_x = TrackWorld.START_X if x is None else float(x)
        self.car_y = TrackWorld.START_Y if y is None else float(y)
        self.car_theta = TrackWorld.START_THETA if theta is None else float(theta)
        self.wheel_angle = 0.0
        self.sim_time = 0.0
        self.step_count = 0
        self.distance_px = 0.0
        self.off_road_steps = 0
        self.in_red_steps = 0
        self.collision_steps = 0
        self.min_clearance = 999.0
        self.cp_idx = 0
        self.laps_done = 0
        self.last_lap_time = None
        self.trail = []
        self.cte_hist = deque(maxlen=180)
        self.steer_hist = deque(maxlen=180)
        self.servo_hist = deque(maxlen=180)
        self._cmd_queue = deque()
        self.pp_ctrl.reset()
        self.ftg_ctrl.reset()
        if self._proj_det is not None:
            self._proj_det.reset()
        if self._proj_pid is not None:
            self._proj_pid.reset()
        if self._proj_corner is not None:
            self._proj_corner.reset()
        self.last_panel_b64 = ''
        self.last_state = None
        # Chay 1 buoc preview (khong di chuyen) de co anh cam ngay khi reset
        self.step_sim(move=False)

    def configure(self, model_name=None, controller_name=None, params=None):
        need_proj = False
        if model_name and model_name != self.model_name:
            self._get_model_path(model_name)
            self.model_name = model_name
            need_proj = True
        if controller_name and controller_name in ('project_current', 'road_pure_pursuit', 'road_ftg'):
            if controller_name != self.controller_name:
                self.controller_name = controller_name
                # Khi chon `project_current`, goi y nguoi dung thay ro hanh vi default.yaml neu ho chua doi
                self.pp_ctrl.reset()
                self.ftg_ctrl.reset()
                if self._proj_pid is not None:
                    self._proj_pid.reset()
                if self._proj_corner is not None:
                    self._proj_corner.reset()
        if params:
            for k, v in params.items():
                if k in self.params:
                    self.params[k] = float(v) if not isinstance(self.params[k], int) else int(round(float(v)))
                    need_proj = True
        if need_proj:
            self._rebuild_project_stack()

    def _infer_road_and_center(self, cam_bgr):
        """Chay ONNX that tren anh cam gia lap -> tra (road_logits, center_logits, infer_ms)."""
        eng = self._get_onnx_engine(self.model_name)
        roi = cam_bgr[ROI_Y0:, :]
        blob = cv2.dnn.blobFromImage(roi, 1.0, (IN_W, IN_H), (0.0, 0.0, 0.0), swapRB=False, crop=False)
        t0 = time.perf_counter()
        outs = eng.infer(blob)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        seg = outs.get('seg')
        if seg is None:
            for v in outs.values():
                if v.size in (SEG_H * SEG_W, 2 * SEG_H * SEG_W, 3 * SEG_H * SEG_W):
                    seg = v
                    break
        if seg is None:
            raise RuntimeError('Model %s khong co output seg 64x128' % self.model_name)
        if seg.size == 3 * SEG_H * SEG_W:
            s3 = seg.reshape(3, SEG_H, SEG_W)
            return s3[0], s3[2], infer_ms
        if seg.size == 2 * SEG_H * SEG_W:
            s2 = seg.reshape(2, SEG_H, SEG_W)
            return s2[0], None, infer_ms
        # Model 1 kenh cu (chi co centerline)
        s1 = seg.reshape(SEG_H, SEG_W)
        return s1, s1, infer_ms

    def _render_project_panel(self, cam_bgr, road_logits, center_logits, ctrl_out,
                              servo, servo_clipped, infer_ms, road_iou):
        """Ve panel 2x2 giong het `tuning_ui.py` de theo doi cam, mask, BEV va do thi."""
        pw, ph = PROC_W, PROC_H
        roi_h = ph - ROI_Y0

        r_mask = (road_logits > 0.0).astype(np.uint8) * 255
        r_up = cv2.resize(r_mask, (pw, roi_h), interpolation=cv2.INTER_NEAREST)

        # (1) Camera + Overlay mat duong xanh (+ tam vang neu o che do project_current)
        ov = cam_bgr.copy()
        ov[ROI_Y0:][r_up > 0] = (40, 180, 40)
        binary = np.zeros((ph, pw, 3), np.uint8)
        binary[ROI_Y0:][r_up > 0] = (25, 115, 25)

        if self.controller_name == 'project_current' and center_logits is not None:
            c_mask = (center_logits > 0.0).astype(np.uint8) * 255
            c_up = cv2.resize(c_mask, (pw, roi_h), interpolation=cv2.INTER_NEAREST)
            ov[ROI_Y0:][c_up > 0] = (0, 230, 255)
            binary[ROI_Y0:][c_up > 0] = (0, 235, 255)
            title1 = 'CAM + MAT DUONG (XANH) + TAM (VANG)'
            title2 = 'PROJECT CURRENT (Ch0+Ch2) | IoU=%.2f' % road_iou
        else:
            title1 = 'CAM + MAT DUONG CH0 (XANH) [NO WHITE/LIDAR]'
            title2 = 'ROAD-ONLY CH0 (%s) | IoU=%.2f' % (
                'PURE PURSUIT' if self.controller_name == 'road_pure_pursuit' else 'FOLLOW-GAP',
                road_iou,
            )

        overlay = cv2.addWeighted(ov, 0.50, cam_bgr, 0.50, 0)
        cv2.line(overlay, (0, ROI_Y0), (pw, ROI_Y0), (0, 255, 255), 1)
        cv2.line(binary, (0, ROI_Y0), (pw, ROI_Y0), (70, 70, 70), 1)

        # (3) Bird's-Eye View + quy dao / tia
        bev = np.zeros((ph, pw, 3), np.uint8)
        if self.controller_name == 'project_current':
            r_bev = cv2.warpPerspective(r_mask, self.pp_ctrl.M_seg, (pw, ph), flags=cv2.INTER_NEAREST)
            bev[r_bev > 0] = (25, 95, 25)
            if self._proj_det.last_bev is not None:
                bev[self._proj_det.last_bev > 0] = (0, 220, 255)
            cv2.line(bev, (pw // 2, 0), (pw // 2, ph), (90, 90, 90), 1)
            fit = ctrl_out.get('fit')
            if fit is not None:
                pts = [[int(fit[0] * t * t + fit[1] * t + fit[2]), ph - int(t * ph)]
                       for t in np.linspace(0.0, 1.0, 35)]
                cv2.polylines(bev, [np.array(pts, np.int32)], False, (0, 0, 255), 2)
                tl = float(self.params['lookahead'])
                xl = fit[0] * tl * tl + fit[1] * tl + fit[2]
                cv2.circle(bev, (int(xl), ph - int(tl * ph)), 5, (255, 0, 255), -1)
            title3 = "BEV + PARABOL FIT (Project Current)"
        elif self.controller_name == 'road_pure_pursuit':
            bev = self.pp_ctrl.render_bev_debug()
            title3 = "BEV CH0 (160x120 FAST) + PURE PURSUIT"
        else:
            if self.ftg_ctrl.debug_road_bev is not None:
                bev[self.ftg_ctrl.debug_road_bev > 0] = (28, 115, 28)
            ox, oy = pw // 2, ph - 4
            for rx, ry, rr, is_wall in self.ftg_ctrl.debug_rays:
                col = (50, 90, 220) if is_wall and rr < 120 else (110, 170, 110)
                cv2.line(bev, (ox, oy), (int(rx), int(ry)), col, 1)
            if self.ftg_ctrl.debug_best_ray is not None:
                bx, by = int(self.ftg_ctrl.debug_best_ray[0]), int(self.ftg_ctrl.debug_best_ray[1])
                cv2.line(bev, (ox, oy), (bx, by), (0, 240, 255), 2)
                cv2.circle(bev, (bx, by), 6, (0, 80, 255), -1)
            title3 = "BEV CH0 + 31 POLAR RAYS (FOLLOW-THE-GAP)"

        # (4) Bieu do CTE / Steer / Servo
        chart = np.full((ph, pw, 3), 22, dtype=np.uint8)
        mid = ph // 2
        cv2.line(chart, (0, mid), (pw, mid), (70, 70, 70), 1)
        for frac in (0.25, 0.75):
            cv2.line(chart, (0, int(ph * frac)), (pw, int(ph * frac)), (45, 45, 45), 1)

        def _plot(vals, col, thick=1):
            if len(vals) < 2:
                return
            arr = list(vals)
            step = float(pw) / max(1, len(arr) - 1)
            pts = [[int(i * step), int(mid - np.clip(v, -1.0, 1.0) * (ph * 0.46))] for i, v in enumerate(arr)]
            cv2.polylines(chart, [np.array(pts, np.int32)], False, col, thick)

        _plot(self.cte_hist, (60, 220, 255), 1)
        _plot(self.steer_hist, (110, 235, 110), 1)
        _plot(self.servo_hist, (80, 120, 255), 2)

        for img, title in ((overlay, title1), (binary, title2), (bev, title3),
                           (chart, 'CTE(vang) | LAI(xanh) | SERVO(do) +-1')):
            cv2.rectangle(img, (0, 0), (pw, 17), (15, 15, 15), -1)
            cv2.putText(img, title, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 240, 240), 1)

        panel = np.vstack([np.hstack([overlay, binary]), np.hstack([bev, chart])])

        # Banner giong tuning_ui.py
        banner = np.full((44, panel.shape[1], 3), 18, dtype=np.uint8)
        lim_txt = ' HARD-LIMIT' if servo_clipped else ''
        lim_col = (70, 70, 255) if servo_clipped else (140, 240, 140)
        cv2.putText(
            banner,
            'MODEL: %s | ONNX: %.1fms | %s' % (self.model_name, infer_ms, ctrl_out['mode']),
            (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 240, 140), 1,
        )
        cv2.putText(
            banner,
            'cte=%+.2f  look=%+.2f  curv=%+.2f  |  lai=%+.2f  servo=%+.2f%s  ga=%.2f' % (
                ctrl_out['cte'], ctrl_out['lookahead'], ctrl_out['curvature'],
                ctrl_out['steer'], servo, lim_txt, ctrl_out['throttle'],
            ),
            (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, lim_col, 1,
        )
        full = np.vstack([banner, panel])
        ok, buf = cv2.imencode('.jpg', full, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return ''
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')

    def step_sim(self, move=True, manual_steer=None, manual_throttle=None, render_panel=True):
        """Chay dung 1 buoc mo phong: render cam -> ONNX infer -> controller -> Ackermann physics.

        Moi gioi han driver / toc do / tan so dieu khien deu doc CO DINH tu configs/default.yaml.
        """
        fx = self.fixed_default
        dt = 1.0 / max(10.0, float(fx['control_hz']))
        car_len = float(self.params['car_length'])
        car_wid = float(self.params['car_width'])
        cam_near = max(4.0, car_len * 0.20)

        cam_bgr, gt_cam, fov_poly, bev_poly, A_bev2map, _ = self.world.render_camera(
            self.car_x, self.car_y, self.car_theta,
            cam_near_offset=cam_near,
            glare=self.params['glare'],
            shadow=self.params['shadow'],
        )

        # 1. Chay suy dien ONNX that
        if self.controller_name == 'project_current':
            res = self._proj_det.process(cam_bgr, need_aux=True, compute_bev=render_panel)
            infer_ms = float(self._proj_det.last_infer_ms)
            road_logits, center_logits, _ = self._infer_road_and_center(cam_bgr)
            look_w = self._proj_corner.effective_lookahead_weight(
                float(self.params['steer_lookahead_weight']))
            err = (1.0 - look_w) * res.cte + look_w * res.cte_lookahead
            pid_out = self._proj_pid.step(err, dt)
            rt = self._proj_det.last_road_turn
            if not res.found or res.cte_lookahead * rt <= 0.0:
                rt = 0
            steer, thr, mode = self._proj_corner.step(
                pid_out, res.cte, res.curvature, dt,
                lane_found=res.found, road_turn=rt, cte_lookahead=res.cte_lookahead,
            )
            ctrl_out = {
                'found': bool(res.found),
                'cte': float(res.cte),
                'lookahead': float(res.cte_lookahead),
                'curvature': float(res.curvature),
                'steer': float(steer),
                'throttle': float(thr),
                'mode': mode,
                'n_bands': int(res.n_bands),
                'fit': res.fit,
            }
            drv_steer_gain = float(self._proj_cfg.get('control.driver.steering_gain', fx['steering_gain']))
            drv_thr_gain = float(self._proj_cfg.get('control.driver.throttle_gain', fx['throttle_gain']))
        else:
            road_logits, center_logits, infer_ms = self._infer_road_and_center(cam_bgr)
            v_str_ui = float(self.params.get('v_straight', fx['v_straight']))
            v_cor_ui = float(self.params.get('v_corner', fx['v_corner']))
            srv_lim_ui = float(self.params.get('servo_limit', 1.00))
            drv_steer_gain = -1.00 if self.controller_name == 'road_pure_pursuit' else float(fx['steering_gain'])
            drv_thr_gain = 1.00 if self.controller_name == 'road_pure_pursuit' else float(fx['throttle_gain'])
            if self.controller_name == 'road_pure_pursuit':
                self.pp_ctrl.v_straight = v_str_ui
                self.pp_ctrl.v_corner = v_cor_ui
                self.pp_ctrl.v_max = min(1.0, max(0.375, float(fx['v_max']), v_str_ui, v_cor_ui))
                self.pp_ctrl.v_min = min(float(fx['v_min']), v_cor_ui)
                self.pp_ctrl.throttle_rise_rate = 1.40
                self.pp_ctrl.drv_steer_gain = drv_steer_gain
                self.pp_ctrl.drv_out_max = srv_lim_ui
                self.pp_ctrl.steer_max = min(1.0, srv_lim_ui)
                self.pp_ctrl.corner_steer_max = min(1.0, srv_lim_ui)
                self.pp_ctrl.servo_ref_at_limit = 0.72
                self.pp_ctrl.edge_margin_px = 56.0
                self.pp_ctrl.edge_push_gain = 0.65
                self.pp_ctrl.curve_enter = 0.06
                ctrl_out = self.pp_ctrl.step(road_logits, dt, overrides=self.params)
            else:
                fx_ui = dict(fx)
                fx_ui['v_straight'] = v_str_ui
                fx_ui['v_corner'] = v_cor_ui
                fx_ui['v_max'] = max(float(fx['v_max']), v_str_ui, v_cor_ui)
                fx_ui['steering_output_max'] = srv_lim_ui
                fx_ui['steering_output_min'] = -srv_lim_ui
                ctrl_out = self.ftg_ctrl.step(road_logits, dt, self.params, fx_ui)

        # Tinh Road IoU tuc thoi giua ONNX Ch0 va ground-truth ban do
        pred_road_cam = cv2.resize(
            (road_logits > 0.0).astype(np.uint8) * 255,
            (PROC_W, PROC_H - ROI_Y0),
            interpolation=cv2.INTER_NEAREST,
        )
        gt_roi = gt_cam[ROI_Y0:, :]
        inter = float(np.sum((pred_road_cam > 127) & (gt_roi > 127)))
        union = float(np.sum((pred_road_cam > 127) | (gt_roi > 127)))
        road_iou = (inter / union) if union > 1.0 else 1.0

        # Manual keyboard override neu nguoi dung dang lai tay kiem tra goc nhin
        if manual_steer is not None or manual_throttle is not None:
            ctrl_out = dict(ctrl_out)
            if manual_steer is not None:
                ctrl_out['steer'] = float(np.clip(manual_steer, -fx['steer_max'], fx['steer_max']))
            if manual_throttle is not None:
                ctrl_out['throttle'] = float(np.clip(manual_throttle, -0.3, fx['v_max']))
            ctrl_out['mode'] = 'MANUAL'

        # 2. Ap dung chuoi hard-limit servo (mac dinh dung ±0.40 cua configs/default.yaml)
        if self.controller_name == 'project_current':
            servo_min = float(fx['steering_output_min'])
            servo_max = float(fx['steering_output_max'])
        else:
            servo_max = abs(float(self.params.get('servo_limit', fx['steering_output_max'])))
            servo_min = -servo_max
        servo_offset = float(fx['steering_offset'])
        servo_out, servo_raw = servo_output_for(
            ctrl_out['steer'], drv_steer_gain, servo_offset, servo_min, servo_max)
        # Quy uoc: steering_gain am -> servo_out am la re phai; doi dau lai de + = re phai cho mo phong
        eff_servo = (-servo_out if drv_steer_gain < 0.0 else servo_out)
        servo_clipped = abs(servo_raw - servo_out) > 1e-6

        # 3. Gia lap do tre pipeline (latency_frames) + dong hoc Ackermann bicycle
        lat = max(0, int(self.params.get('latency_frames', 1)))
        eff_thr = float(np.clip(ctrl_out['throttle'] * drv_thr_gain, -1.0, 1.0))
        self._cmd_queue.append((eff_servo, eff_thr))
        while len(self._cmd_queue) > lat + 1:
            self._cmd_queue.popleft()
        app_servo, app_thr = self._cmd_queue[0]

        # Anh xa servo -> goc banh vat ly:
        # - road_pure_pursuit dung full tam servo +-1.00 (wheel_deg_at_limit = goc banh o full lock 1.00)
        # - project_current / road_ftg dung moc chuan default.yaml +-0.40
        nom_span = (1.0 if self.controller_name == 'road_pure_pursuit'
                    else max(1e-6, abs(float(fx['steering_output_max']))))
        wheel_at_lim_rad = math.radians(float(self.params['wheel_deg_at_limit']))
        max_mech_rad = math.radians(45.0)
        self.wheel_angle = float(np.clip((app_servo / nom_span) * wheel_at_lim_rad, -max_mech_rad, max_mech_rad))

        if move:
            v_px = app_thr * float(self.params['speed_px_per_s'])
            wb_px = max(12.0, car_len * 0.62)
            # Sub-step 4 lan de tich phan quy dao cua gat chinh xac
            sub_dt = dt / 4.0
            for _ in range(4):
                self.car_x += v_px * math.cos(self.car_theta) * sub_dt
                self.car_y += v_px * math.sin(self.car_theta) * sub_dt
                self.car_theta += (v_px / wb_px) * math.tan(self.wheel_angle) * sub_dt
            self.car_theta = (self.car_theta + math.pi) % (2.0 * math.pi) - math.pi
            self.sim_time += dt
            self.step_count += 1
            self.distance_px += abs(v_px) * dt

        fp = self.world.check_car_footprint(self.car_x, self.car_y, self.car_theta, car_len, car_wid)
        if move:
            if not fp['on_road_all']:
                self.off_road_steps += 1
            if fp['in_red_any']:
                self.in_red_steps += 1
            if fp['collided_obs']:
                self.collision_steps += 1
            self.min_clearance = min(self.min_clearance, fp['min_clearance_px'])

            # Cap nhat checkpoint & dem vong dua
            cps = self.world.checkpoints
            if self.cp_idx < len(cps):
                cpx, cpy = cps[self.cp_idx]
                if math.hypot(self.car_x - cpx, self.car_y - cpy) < 76.0:
                    self.cp_idx += 1
            elif self.car_x <= TrackWorld.START_LINE_X and self.car_y < 230.0 and self.car_x > 640.0:
                self.laps_done += 1
                self.last_lap_time = round(self.sim_time, 2)
                self.cp_idx = 0

            status_code = 0 if fp['on_road_all'] else (2 if fp['in_red_any'] else 1)
            self.trail.append([round(self.car_x, 1), round(self.car_y, 1), status_code])
            if len(self.trail) > 3000:
                self.trail = self.trail[-3000:]

            self.cte_hist.append(ctrl_out['cte'])
            self.steer_hist.append(ctrl_out['steer'])
            self.servo_hist.append(eff_servo)

        # Chieu diem target Pure Pursuit / FTG nguoc len ban do 2D de ve truc quan
        target_map = None
        if self.controller_name == 'road_pure_pursuit' and self.pp_ctrl.debug_target is not None:
            tx, ty = self.pp_ctrl.debug_target
            pt = A_bev2map.dot(np.array([tx, ty, 1.0]))
            target_map = [round(float(pt[0]), 1), round(float(pt[1]), 1)]
        elif self.controller_name == 'road_ftg' and self.ftg_ctrl.debug_best_ray is not None:
            bx, by = self.ftg_ctrl.debug_best_ray
            pt = A_bev2map.dot(np.array([bx, by, 1.0]))
            target_map = [round(float(pt[0]), 1), round(float(pt[1]), 1)]

        if render_panel:
            self.last_panel_b64 = self._render_project_panel(
                cam_bgr, road_logits, center_logits, ctrl_out,
                eff_servo, servo_clipped, infer_ms, road_iou,
            )

        out_of_bounds = (self.car_x < 10 or self.car_x > self.world.w - 10 or
                         self.car_y < 80 or self.car_y > 630)
        off_pct = (100.0 * self.off_road_steps / max(1, self.step_count))
        red_pct = (100.0 * self.in_red_steps / max(1, self.step_count))

        state = {
            'car': {
                'x': round(self.car_x, 1),
                'y': round(self.car_y, 1),
                'theta': round(self.car_theta, 4),
                'wheel_angle': round(self.wheel_angle, 4),
                'length': car_len,
                'width': car_wid,
                'corners': fp['corners'],
                'on_road_all': fp['on_road_all'],
                'in_red_any': fp['in_red_any'],
                'collided_obs': fp['collided_obs'],
                'min_clearance_px': fp['min_clearance_px'],
            },
            'obstacles': self.world.obstacles,
            'fov_poly': [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in fov_poly],
            'bev_poly': [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in bev_poly],
            'target_map': target_map,
            'telemetry': {
                'model': self.model_name,
                'controller': self.controller_name,
                'infer_ms': round(infer_ms, 2),
                'road_iou': round(road_iou, 3),
                'found': ctrl_out['found'],
                'cte': round(ctrl_out['cte'], 3),
                'lookahead': round(ctrl_out['lookahead'], 3),
                'curvature': round(ctrl_out['curvature'], 3),
                'steer': round(ctrl_out['steer'], 3),
                'servo': round(eff_servo, 3),
                'servo_clipped': bool(servo_clipped),
                'throttle': round(ctrl_out['throttle'], 3),
                'mode': ctrl_out['mode'],
                'sim_time': round(self.sim_time, 2),
                'steps': self.step_count,
                'progress_pct': round(100.0 * self.cp_idx / float(len(self.world.checkpoints)), 1),
                'laps_done': self.laps_done,
                'last_lap_time': self.last_lap_time,
                'off_road_steps': self.off_road_steps,
                'off_road_pct': round(off_pct, 1),
                'in_red_steps': self.in_red_steps,
                'in_red_pct': round(red_pct, 1),
                'collision_steps': self.collision_steps,
                'obstacle_hole': bool(ctrl_out.get('obstacle_hole', False)),
                'out_of_bounds': bool(out_of_bounds),
            },
            'trail_tail': self.trail[-1] if self.trail else None,
            'panel': self.last_panel_b64 if render_panel else '',
        }
        self.last_state = state
        return state

    def run_benchmark_lap(self, max_steps=3200):
        """Chay nhanh 1 vong hoan chinh (khong encode JPEG tung frame) de cham diem trong ~3-5 giay."""
        self.reset_car()
        for _ in range(max_steps):
            st = self.step_sim(move=True, render_panel=False)
            if st['telemetry']['laps_done'] >= 1 or st['telemetry']['out_of_bounds']:
                break
        # Render 1 panel cuoi cung de hien thi tren UI
        st = self.step_sim(move=False, render_panel=True)
        st['full_trail'] = self.trail
        return st


# =============================================================================
# 4. LOCAL HTTP SERVER (STDLIB ONLY - ZERO EXTRA DEPENDENCIES)
# =============================================================================

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def make_handler(session, html_path):
    class SimHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Giam spam console khi web poll 20 Hz
            pass

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, data, content_type, code=200):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self):
            n = int(self.headers.get('Content-Length', '0'))
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode('utf-8'))

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                with open(html_path, 'rb') as fh:
                    self._send_bytes(fh.read(), 'text/html; charset=utf-8')
            elif path == '/api/map.jpg':
                with open(session.world.map_path, 'rb') as fh:
                    self._send_bytes(fh.read(), 'image/jpeg')
            elif path == '/api/info':
                with session.lock:
                    self._send_json({
                        'map_width': session.world.w,
                        'map_height': session.world.h,
                        'models': session.available_models,
                        'model': session.model_name,
                        'controller': session.controller_name,
                        'fixed_default': session.fixed_default,
                        'obstacles': session.world.obstacles,
                        'params': session.params,
                        'state': session.last_state,
                        'full_trail': session.trail,
                    })
            else:
                self._send_json({'error': 'Not found: ' + path}, code=404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._read_json()
                with session.lock:
                    if path == '/api/step':
                        n_steps = max(1, min(10, int(body.get('steps', 1))))
                        man_s = body.get('manual_steer')
                        man_t = body.get('manual_throttle')
                        st = None
                        for k in range(n_steps):
                            st = session.step_sim(
                                move=bool(body.get('move', True)),
                                manual_steer=man_s,
                                manual_throttle=man_t,
                                render_panel=(k == n_steps - 1),
                            )
                        self._send_json(st)
                    elif path == '/api/config':
                        session.configure(
                            model_name=body.get('model'),
                            controller_name=body.get('controller'),
                            params=body.get('params'),
                        )
                        st = session.step_sim(move=False, render_panel=True)
                        self._send_json({
                            'ok': True,
                            'model': session.model_name,
                            'controller': session.controller_name,
                            'params': session.params,
                            'state': st,
                        })
                    elif path == '/api/reset':
                        session.reset_car(
                            x=body.get('x'),
                            y=body.get('y'),
                            theta=body.get('theta'),
                        )
                        self._send_json({
                            'ok': True,
                            'state': session.last_state,
                            'full_trail': session.trail,
                        })
                    elif path == '/api/benchmark':
                        max_steps = int(body.get('max_steps', 3200))
                        st = session.run_benchmark_lap(max_steps=max_steps)
                        self._send_json(st)
                    elif path == '/api/obstacles':
                        action = body.get('action', 'set')
                        cur = list(session.world.obstacles)
                        if action == 'clear':
                            session.world.set_obstacles([])
                        elif action == 'preset':
                            session.world.set_obstacles(TrackWorld.sample_obstacles())
                        elif action == 'add':
                            cur.append({
                                'x': float(body.get('x', 520.0)),
                                'y': float(body.get('y', 163.0)),
                                'r': float(body.get('r', 12.0)),
                            })
                            session.world.set_obstacles(cur)
                        elif action == 'toggle_at':
                            tx, ty = float(body.get('x', 0.0)), float(body.get('y', 0.0))
                            tr = float(body.get('r', 12.0))
                            kept = [o for o in cur if math.hypot(o['x'] - tx, o['y'] - ty) > max(o['r'] + 8.0, 20.0)]
                            if len(kept) == len(cur):
                                kept.append({'x': tx, 'y': ty, 'r': tr})
                            session.world.set_obstacles(kept)
                        else:
                            session.world.set_obstacles(body.get('obstacles', []))
                        st = session.step_sim(move=False, render_panel=True)
                        self._send_json({
                            'ok': True,
                            'obstacles': session.world.obstacles,
                            'state': st,
                        })
                    else:
                        self._send_json({'error': 'Not found: ' + path}, code=404)
            except Exception as exc:
                self._send_json({'error': str(exc)}, code=500)

    return SimHandler


def run_headless_suite(map_path, models_dir, save_img=None, with_obstacles=False):
    """Chay benchmark tat ca bo dieu khien tren ONNX that voi configs/default.yaml CO DINH."""
    sess = SimulatorSession(map_path, models_dir)
    if with_obstacles:
        sess.world.set_obstacles(TrackWorld.sample_obstacles())
    fx = sess.fixed_default
    print('=== FIXED configs/default.yaml | Obstacles=%d ===' % len(sess.world.obstacles))
    print('  servo=[%.2f, %.2f] PWM=%s | steer_gain=%.2f | thr_gain=%.2f | v_str=%.2f v_cor=%.2f | %d Hz' % (
        fx['steering_output_min'], fx['steering_output_max'], fx['pulse_width_range'],
        fx['steering_gain'], fx['throttle_gain'], fx['v_straight'], fx['v_corner'], fx['control_hz'],
    ))
    cases = [
        ('lane_road_only_fast.onnx',  'road_pure_pursuit', {}, 'Road PurePursuit (RoadOnly-Fast 1ch)'),
        ('lane_road_only_resse.onnx', 'road_pure_pursuit', {}, 'Road PurePursuit (RoadOnly-ResSE 1ch)'),
        ('lane_road_only_fast.onnx',  'road_ftg',          {}, 'Road FollowGap (RoadOnly-Fast 1ch)'),
        ('lane_road_only_resse.onnx', 'road_ftg',          {}, 'Road FollowGap (RoadOnly-ResSE 1ch)'),
    ]
    if not with_obstacles:
        cases = [
            ('lane_option_b_resse.onnx', 'project_current', {}, 'Project Current (ResSE, default.yaml)'),
            ('lane_option_b_fast.onnx',  'project_current', {}, 'Project Current (Fast, default.yaml)'),
        ] + cases
    print('\n================ HEADLESS CLOSED-LOOP ONNX BENCHMARK (default.yaml FIXED) ================')
    print('%-40s | %5s | %6s | %4s | %6s | %6s | %5s | %5s | %s' % (
        'Case', 'Steps', 'Prog%', 'Laps', 'OffRd%', 'InRed%', 'Cols', 'IoU', 'Time'))
    print('-' * 108)
    trails = {}
    for mname, cname, p_ovr, label in cases:
        try:
            sess.configure(model_name=mname, controller_name=cname, params=p_ovr)
            st = sess.run_benchmark_lap(max_steps=3200)
            t = st['telemetry']
            print('%-40s | %5d | %5.1f%% | %4d | %5.1f%% | %5.1f%% | %5d | %.3f | %.1fs%s' % (
                label, t['steps'], t['progress_pct'], t['laps_done'],
                t['off_road_pct'], t['in_red_pct'], t['collision_steps'], t['road_iou'], t['sim_time'],
                ' [OOB]' if t['out_of_bounds'] else (' [DONE!]' if t['laps_done'] >= 1 else ''),
            ))
            trails[label] = st['full_trail']
        except Exception as exc:
            print('%-40s | ERROR: %s' % (label, exc))

    if save_img and trails:
        vis = sess.world.raw_bgr.copy()
        for o in sess.world.obstacles:
            cx, cy, rad = int(round(o['x'])), int(round(o['y'])), int(round(o['r']))
            cv2.circle(vis, (cx, cy), rad + 2, (255, 255, 255), -1)
            cv2.circle(vis, (cx, cy), rad, (22, 115, 249), -1)
        colors = [
            (60, 240, 60),
            (255, 220, 40),
            (255, 90, 220),
            (255, 160, 100),
            (0, 220, 255),
            (0, 140, 255),
        ]
        y_leg = 22
        for (label, tr), col in zip(trails.items(), colors):
            if len(tr) >= 2:
                pts = np.array([[int(p[0]), int(p[1])] for p in tr], np.int32)
                cv2.polylines(vis, [pts], False, col, 2)
            cv2.line(vis, (16, y_leg - 4), (44, y_leg - 4), col, 3)
            cv2.putText(vis, label, (52, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 2)
            cv2.putText(vis, label, (52, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
            y_leg += 18
        cv2.imwrite(save_img, vis)
        print('Saved comparison trajectories:', save_img)


def main():
    ap = argparse.ArgumentParser(description='Local ONNX Track Simulator Web App')
    ap.add_argument('--map', default=DEFAULT_MAP, help='Duong dan anh sa ban')
    ap.add_argument('--models', default=DEFAULT_MODELS_DIR, help='Thu muc chua cac file .onnx')
    ap.add_argument('--html', default=DEFAULT_HTML, help='File giao dien HTML')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--headless-test', action='store_true', help='Chay test vong kin tu dong tat ca bo dieu khien')
    ap.add_argument('--with-obstacles', action='store_true', help='Dat 3 vat can tinh mau khi chay --headless-test')
    ap.add_argument('--save-plot', default=None, help='Luu anh quy dao so sanh khi chay --headless-test')
    args = ap.parse_args()

    if args.headless_test:
        run_headless_suite(args.map, args.models, save_img=args.save_plot, with_obstacles=args.with_obstacles)
        return

    session = SimulatorSession(args.map, args.models)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session, args.html))
    print('=== JetRacer Local ONNX Track Simulator ===')
    print('Map    :', session.world.map_path)
    print('Models :', ', '.join(m['name'] for m in session.available_models))
    print('URL    : http://%s:%d' % (args.host, args.port))
    print('Nhan Ctrl+C de dung server.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nDa dung simulator.')
        server.server_close()


if __name__ == '__main__':
    main()
