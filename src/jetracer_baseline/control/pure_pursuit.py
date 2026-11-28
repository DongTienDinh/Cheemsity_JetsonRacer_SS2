# -*- coding: utf-8 -*-
"""Dieu khien bam duong chi dung kenh mat duong Ch0 (drivable_road) bang Pure Pursuit.

Toi uu hoa rieng cho Jetson Nano 5W (ARM Cortex-A57 2-core @ 918 MHz, 25-30 FPS):
  - Xu ly BEV o do phan giai 160x120 (giam 4x so pixel so voi 320x240, van cao gap
    2.3x do phan giai 64x128 cua model -> giam ~4x thoi gian warp/CC/distanceTransform).
  - Gop `fill_small_holes` + phat hien lo vat can thanh DUNG 1 lan `connectedComponents`.
  - Dung `cv2.distanceTransform(..., DIST_L2, 3)` (chamfer 3x3 nhanh hon ~2x so voi 5x5).
  - Tinh san `score_map` 1 lan cho ca frame va dung `memoryview` C-level trong vong
    `_trace_medial_path` (bo toan bo phep tinh cos/sin & cap phat mang moi buoc).
  - Giu NGUYEN he don vi tham so theo 320x240 (`lookahead_px=84`, `half_lane_px=82`...)
    de khong phai doi bat ky config hay slider nao.

Tuong thich 100% Python 3.6 (Jetson Nano JetPack 4.5.1).
"""

from __future__ import print_function

import math

import cv2
import numpy as np

from ..perception.lane_cnn import PROC_H, PROC_W, SEG_H, SEG_W, _build_warp

STRAIGHT = 'THANG'
CORNER = 'CUA'

# Do phan giai xu ly BEV noi bo: 160x120 (= 0.5x cua 320x240 -> nhanh gap ~4-5x tren ARM Nano)
BEV_SCALE = 0.5
BEV_W = int(PROC_W * BEV_SCALE)   # 160
BEV_H = int(PROC_H * BEV_SCALE)   # 120

DEFAULT_SCALE_X = 60.0 / 158.0
DEFAULT_SCALE_Y = 60.0 / 115.0
DEFAULT_CAM_NEAR_OFFSET = 14.0
DEFAULT_ROI_TOP = 0.34
DEFAULT_WARP_SRC = [[0.00, 0.92], [1.00, 0.92], [0.66, 0.42], [0.33, 0.42]]
DEFAULT_WARP_DST_MARGIN = 0.32


def fill_small_holes(binary_u8, max_hole_area=180, return_has_large=False):
    """Chi lap cac lo nhieu nho ben trong `ego_road` (chu START, dom choi),
    GIU NGUYEN cac lo lon (area >= max_hole_area) cua vat can tinh tren lan duong.
    """
    h, w = binary_u8.shape[:2]
    inv = (binary_u8 == 0).astype(np.uint8)
    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    has_large = False
    if n_cc <= 1:
        return (binary_u8, False) if return_has_large else binary_u8

    areas = stats[1:, cv2.CC_STAT_AREA]
    x0s = stats[1:, cv2.CC_STAT_LEFT]
    y0s = stats[1:, cv2.CC_STAT_TOP]
    x1s = x0s + stats[1:, cv2.CC_STAT_WIDTH]
    y1s = y0s + stats[1:, cv2.CC_STAT_HEIGHT]
    interior = (x0s > 0) & (y0s > 0) & (x1s < w) & (y1s < h)
    if not interior.any():
        return (binary_u8, False) if return_has_large else binary_u8

    has_large = bool(np.any(interior & (areas >= max_hole_area)))
    small_ids = np.flatnonzero(interior & (areas < max_hole_area)) + 1
    if len(small_ids) == 0:
        return (binary_u8, has_large) if return_has_large else binary_u8

    lut = np.zeros(n_cc, dtype=np.uint8)
    lut[small_ids] = 1
    out = np.where(lut[lbl] > 0, np.uint8(1), binary_u8)
    return (out, has_large) if return_has_large else out


class RoadPurePursuitController(object):
    """Bo dieu khien Pure Pursuit chi dung kenh Ch0 (drivable_road) tu ONNX/TensorRT."""

    def __init__(self, cfg=None):
        rel_angles = np.linspace(-1.35, 1.35, 29, dtype=np.float64)
        # Precompute (da, cos, sin, smooth_pen) thanh tuple Python de vong tracing
        # dung memoryview C-level (nhanh gap ~3x so voi goi 12 ufunc NumPy/buoc).
        self._fan = tuple(
            (float(da), math.cos(float(da)), math.sin(float(da)),
             (14.0 * BEV_SCALE) * float(da) * float(da))
            for da in rel_angles
        )
        self._geom_key = None
        self.M_seg = None        # ma tran warp 64x128 -> BEV 320x240 (cho simulator / debug)
        self._M_seg_fast = None  # ma tran warp 64x128 -> BEV 160x120 (cho runtime sieu nhanh)
        self.bev_visible = None
        self._bev_vis_fast = None
        if cfg is not None:
            self.reset_from_config(cfg)
        else:
            self._init_defaults()
        self.reset()

    def _init_defaults(self):
        self.roi_top = DEFAULT_ROI_TOP
        self.warp_src = DEFAULT_WARP_SRC
        self.warp_dst_margin = DEFAULT_WARP_DST_MARGIN
        self.v_straight = 0.30
        self.v_corner = 0.12
        self.v_min = 0.10
        self.v_max = 0.35
        self.throttle_rise_rate = 0.80
        self.throttle_fall_rate = 3.00
        self.steer_max = 0.60
        self.corner_steer_max = 0.60
        self.drv_steer_gain = -0.65
        self.drv_out_max = 0.40
        self.servo_ref_at_limit = 0.39
        self.pre_corner_decel = 0.0
        self.road_turn_steer_gain = 1.0
        self.curve_enter = 0.25
        self.lookahead_px = 84.0
        self.pp_gain = 1.15
        self.half_lane_px = 82.0
        self.wheel_deg_at_limit = 34.0
        self.car_length_px = 76.0
        self.edge_margin_px = 54.0
        self.edge_push_gain = 0.45
        self.steer_smooth = 0.72
        self.max_hole_area = 180
        self.scale_x = DEFAULT_SCALE_X
        self.scale_y = DEFAULT_SCALE_Y
        self.cam_near_offset = DEFAULT_CAM_NEAR_OFFSET
        self._build_geometry()

    def _build_geometry(self):
        key = (round(self.roi_top, 4), repr(self.warp_src), round(self.warp_dst_margin, 4))
        if key == self._geom_key and self._M_seg_fast is not None:
            return
        self._geom_key = key
        roi_y0 = int(PROC_H * self.roi_top)
        roi_h = max(1, PROC_H - roi_y0)
        M_bev = _build_warp(PROC_W, PROC_H, self.warp_src, self.warp_dst_margin)
        seg_to_proc = np.array([
            [PROC_W / float(SEG_W), 0.0, 0.0],
            [0.0, roi_h / float(SEG_H), float(roi_y0)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.M_seg = M_bev.dot(seg_to_proc)

        S_half = np.array([
            [BEV_SCALE, 0.0, 0.0],
            [0.0, BEV_SCALE, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self._M_seg_fast = S_half.dot(self.M_seg)

        valid_fast = cv2.warpPerspective(
            np.ones((SEG_H, SEG_W), np.uint8), self._M_seg_fast,
            (BEV_W, BEV_H), flags=cv2.INTER_NEAREST,
        )
        ker_fast = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._bev_vis_fast = cv2.erode(
            valid_fast, ker_fast, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.bev_visible = cv2.resize(
            self._bev_vis_fast.astype(np.uint8), (PROC_W, PROC_H),
            interpolation=cv2.INTER_NEAREST) > 0

    def reset_from_config(self, cfg):
        """Doc lai tham so tu config (ton trong tuyet doi default.yaml)."""
        c = cfg.get
        self.roi_top = float(c('lane.cnn.roi_top', DEFAULT_ROI_TOP))
        self.warp_src = c('lane.cnn.warp_src', DEFAULT_WARP_SRC)
        self.warp_dst_margin = float(c('lane.cnn.warp_dst_margin', DEFAULT_WARP_DST_MARGIN))

        self.v_straight = float(c('control.v_straight', 0.30))
        self.v_corner = float(c('control.v_corner', 0.12))
        self.v_min = float(c('control.v_min', 0.10))
        self.v_max = float(c('control.v_max', 0.35))
        self.throttle_rise_rate = float(c('control.throttle_rise_rate', 0.80))
        self.throttle_fall_rate = float(c('control.throttle_fall_rate', 3.00))
        self.steer_max = float(c('control.steer_max', 0.60))
        self.corner_steer_max = float(c('control.corner_steer_max', self.steer_max))
        self.drv_steer_gain = float(c('control.driver.steering_gain', -0.65))
        self.drv_out_max = float(c('control.driver.steering_output_max', 0.40))
        self.servo_ref_at_limit = float(c('control.pure_pursuit.servo_ref_at_limit', 0.39))
        self.pre_corner_decel = float(c('control.pre_corner_decel', 0.0))
        self.road_turn_steer_gain = float(c('control.road_turn_steer_gain', 1.0))
        self.curve_enter = float(c('control.curve_enter', 0.25))

        self.lookahead_px = float(c('control.pure_pursuit.lookahead_px', 84.0))
        self.pp_gain = float(c('control.pure_pursuit.gain', 1.15))
        self.half_lane_px = float(c('control.pure_pursuit.half_lane_px', 82.0))
        self.wheel_deg_at_limit = float(c('control.pure_pursuit.wheel_deg_at_limit', 34.0))
        self.car_length_px = float(c('control.pure_pursuit.car_length_px', 76.0))
        self.edge_margin_px = float(c('control.pure_pursuit.edge_margin_px', 54.0))
        self.edge_push_gain = float(c('control.pure_pursuit.edge_push_gain', 0.45))
        self.steer_smooth = float(c('control.pure_pursuit.steer_smooth', 0.72))
        self.max_hole_area = int(c('control.pure_pursuit.max_hole_area', 180))
        self.scale_x = float(c('control.pure_pursuit.scale_x', DEFAULT_SCALE_X))
        self.scale_y = float(c('control.pure_pursuit.scale_y', DEFAULT_SCALE_Y))
        self.cam_near_offset = float(c('control.pure_pursuit.cam_near_offset', DEFAULT_CAM_NEAR_OFFSET))
        self._build_geometry()

    def reset(self):
        self.mode = STRAIGHT
        self._prev_steer = 0.0
        self._throttle = 0.0
        self.steer_wanted = 0.0
        self.steer_limit = self.corner_steer_max
        self.curvy_zone = False
        self.corner_delay_active = False
        self._road_start_streak = 0
        self.debug_bev_pts = []
        self.debug_target = None
        self.debug_road_bev = None
        self._debug_road_fast = None
        self.debug_far_angle = 0.0
        self.debug_obstacle_hole = False

    @property
    def road_start_ready(self):
        return self._road_start_streak >= 5

    @property
    def throttle(self):
        return self._throttle

    def _effective_steer_span(self):
        gain_abs = max(1e-6, abs(self.drv_steer_gain))
        return min(self.corner_steer_max, abs(self.drv_out_max) / gain_abs)

    def _extract_ego_road(self, road_bev_fast):
        """Tach ego_road o 160x120 + lap lo nhieu nho + phat hien lo vat can trong 1 luot."""
        n_cc, lbl, _, _ = cv2.connectedComponentsWithStats(road_bev_fast, connectivity=8)
        if n_cc <= 1:
            return None, False, False
        half = BEV_W // 2
        near = lbl[BEV_H - 17:BEV_H, half - 14:half + 14].ravel()
        near = near[near > 0]
        hole_thr = max(8, int(round(self.max_hole_area * (BEV_SCALE * BEV_SCALE))))
        if len(near) >= 10:
            cid = int(np.bincount(near).argmax())
            ego = (lbl == cid).astype(np.uint8)
            ego, has_obs = fill_small_holes(ego, max_hole_area=hole_thr, return_has_large=True)
            return ego, True, has_obs

        wide = lbl[BEV_H - 32:BEV_H, :]
        counts = np.bincount(wide.ravel())
        counts[0] = 0
        cands = np.flatnonzero(counts >= 25)
        if len(cands) == 0:
            return None, False, False
        best_cid, best_score = None, 1e9
        for cid in cands:
            xs = np.where(wide == cid)[1]
            d = abs(float(xs.mean()) - half)
            if d < best_score:
                best_score = d
                best_cid = int(cid)
        ego = (lbl == best_cid).astype(np.uint8)
        ego, has_obs = fill_small_holes(ego, max_hole_area=hole_thr, return_has_large=True)
        return ego, False, has_obs

    def _trace_medial_path(self, ego_fast, half_lane_fast):
        """Lan truc tam theo do dai cung o 160x120 (vector hoa NumPy 29 goc/buoc -> <0.3ms!)."""
        outside = self._bev_vis_fast & (ego_fast == 0)
        if int(np.count_nonzero(outside)) < 6:
            return [(BEV_W * 0.5, float(BEV_H - 2 - k * 5)) for k in range(18)], None, outside

        dist = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 3)

        # Tinh san ban do diem theo khoang cach mep 1 lan cho ca frame -> vong lap chi tra memoryview
        score_map = np.where(
            dist <= half_lane_fast,
            dist,
            half_lane_fast - 0.85 * (dist - half_lane_fast),
        ).astype(np.float32)
        score_map[ego_fast == 0] = -1e6
        score_map[:2, :] = -1e6
        score_map[-2:, :] = -1e6
        score_map[:, :2] = -1e6
        score_map[:, -2:] = -1e6
        score_mv = memoryview(score_map.ravel())

        half = BEV_W * 0.5
        y_start = BEV_H - 6
        row_xs = np.flatnonzero(ego_fast[y_start, :] > 0)
        if len(row_xs) == 0:
            ys_all, xs_all = np.nonzero(ego_fast)
            if len(ys_all) == 0:
                return [], dist, outside
            idx = int(np.argmin((xs_all - half) ** 2 + (ys_all - BEV_H) ** 2))
            cur_x, cur_y = float(xs_all[idx]), float(ys_all[idx])
        else:
            vals = dist[y_start, row_xs]
            if float(vals.max()) > half_lane_fast + 3.0:
                score = np.abs(vals - half_lane_fast) + 0.15 * np.abs(row_xs - half)
                cur_x = float(row_xs[int(np.argmin(score))])
            else:
                cur_x = float(row_xs[int(np.argmax(vals))])
            cur_y = float(y_start)

        path = [(cur_x, cur_y)]
        heading = -math.pi * 0.5
        ds = 10.0 * BEV_SCALE   # 5.0 px trong 160x120 == 10.0 px trong 320x240
        back_pen_val = 35.0 * BEV_SCALE
        fan = self._fan
        w_bev = BEV_W
        h_lim = BEV_H - 2
        w_lim = BEV_W - 2
        cos_fn = math.cos
        sin_fn = math.sin

        for _ in range(24):
            ch = cos_fn(heading)
            sh = sin_fn(heading)
            best_sc = -1e5
            best_nx = 0.0
            best_ny = 0.0
            best_da = 0.0
            y_back = cur_y + 1.0
            for da, cda, sda, sp in fan:
                nx = cur_x + ds * (ch * cda - sh * sda)
                ny = cur_y + ds * (sh * cda + ch * sda)
                ix = int(nx + 0.5)
                iy = int(ny + 0.5)
                if 2 <= ix < w_lim and 2 <= iy < h_lim:
                    cs = score_mv[iy * w_bev + ix]
                    if cs > -1e5:
                        sc = cs - sp
                        if ny > y_back:
                            sc -= back_pen_val
                        if sc > best_sc:
                            best_sc = sc
                            best_nx = nx
                            best_ny = ny
                            best_da = da
            if best_sc <= -1e5:
                break
            heading = heading + 0.65 * best_da
            cur_x = best_nx
            cur_y = best_ny
            path.append((cur_x, cur_y))
            if cur_y <= 7.0:
                break
        return path, dist, outside

    def step(self, road_logits, dt, overrides=None, compute_bev=False):
        """Nhan `road_logits` (64x128 float logits hoac 0/255 mask) -> dict lenh & debug."""
        if dt <= 0.0:
            dt = 1.0 / 30.0
        ov = overrides or {}
        ld_px = float(ov.get('pp_lookahead_px', self.lookahead_px))
        pp_gain = float(ov.get('pp_gain', self.pp_gain))
        half_lane_px = float(ov.get('half_lane_px', self.half_lane_px))
        wheel_deg = float(ov.get('wheel_deg_at_limit', self.wheel_deg_at_limit))
        car_len = float(ov.get('car_length', self.car_length_px))

        wheel_at_lim_rad = math.radians(wheel_deg)
        steer_span = self._effective_steer_span()
        self.steer_limit = steer_span

        if road_logits is None:
            ego_fast, on_center, has_obs = None, False, False
        else:
            road_bin = (road_logits > 0.0).astype(np.uint8)
            road_bev_fast = cv2.warpPerspective(
                road_bin, self._M_seg_fast, (BEV_W, BEV_H), flags=cv2.INTER_NEAREST)
            ego_fast, on_center, has_obs = self._extract_ego_road(road_bev_fast)

        self._debug_road_fast = ego_fast
        if compute_bev and ego_fast is not None:
            self.debug_road_bev = cv2.resize(
                ego_fast, (PROC_W, PROC_H), interpolation=cv2.INTER_NEAREST)
        else:
            self.debug_road_bev = None

        band_h = BEV_H // 10
        road_seen = (ego_fast is not None and on_center and
                     int(np.count_nonzero(ego_fast[-3 * band_h:-2 * band_h])) >= 5 * band_h)
        self._road_start_streak = self._road_start_streak + 1 if road_seen else 0

        if ego_fast is None:
            steer = float(np.clip(self._prev_steer * 0.90, -steer_span, steer_span))
            self._prev_steer = steer
            self.steer_wanted = steer
            self._throttle = max(self.v_min, self._throttle - self.throttle_fall_rate * dt)
            self.mode = CORNER
            self.debug_bev_pts = []
            self.debug_target = None
            self.debug_obstacle_hole = False
            return {
                'found': False, 'cte': 0.0, 'lookahead': 0.0, 'curvature': 0.0,
                'steer': steer, 'throttle': self._throttle, 'mode': self.mode,
                'n_bands': 0, 'obstacle_hole': False,
            }

        self.debug_obstacle_hole = has_obs
        path_fast, dist_fast, outside_fast = self._trace_medial_path(
            ego_fast, half_lane_fast=half_lane_px * BEV_SCALE)

        inv_scale = 1.0 / BEV_SCALE
        # Quy toa do path ve he 320x240 cho Pure Pursuit & debug panel
        self.debug_bev_pts = [(p[0] * inv_scale, p[1] * inv_scale) for p in path_fast]
        if len(path_fast) < 2:
            ys, xs = np.nonzero(ego_fast)
            cx = float(np.median(xs)) * inv_scale
            cy = float(np.median(ys)) * inv_scale
            cte = float(np.clip((cx - PROC_W * 0.5) / (PROC_W * 0.5), -1.0, 1.0))
            steer = float(np.clip(cte * 1.2, -1.0, 1.0)) * steer_span
            self._prev_steer = steer
            self.steer_wanted = steer
            self._throttle = self.v_corner
            self.mode = CORNER
            self.debug_target = (cx, cy)
            return {
                'found': True, 'cte': cte, 'lookahead': cte, 'curvature': 0.0,
                'steer': steer, 'throttle': self._throttle, 'mode': self.mode,
                'n_bands': 1, 'obstacle_hole': self.debug_obstacle_hole,
            }

        pts = np.array(self.debug_bev_pts, dtype=np.float64)
        seg_d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        cum_s = np.concatenate([[0.0], np.cumsum(seg_d)])
        total_s = float(cum_s[-1])

        far_idx = int(np.searchsorted(cum_s, min(total_s, max(ld_px * 1.6, 150.0))))
        far_idx = min(far_idx, len(pts) - 1)
        far_dx = pts[far_idx, 0] - PROC_W * 0.5
        far_dy = max(12.0, (PROC_H + 18.0) - pts[far_idx, 1])
        far_angle = math.atan2(far_dx, far_dy)
        self.debug_far_angle = far_angle
        curv_proxy = float(np.clip(far_angle / 0.85, -1.0, 1.0))

        # Do goc be cong o doan xa cua truc tam (tu nua sau den ngon path):
        # tren cua chu U 180 do, ngon truc tam uon ngang >= 35-90 do ngay khi moi chom vao cua!
        mid_idx = max(0, len(pts) // 2 - 1)
        tip_dx = pts[-1, 0] - pts[mid_idx, 0]
        tip_dy = pts[mid_idx, 1] - pts[-1, 1]
        tip_turn = math.atan2(tip_dx, max(6.0, tip_dy)) if math.hypot(tip_dx, tip_dy) > 8.0 else far_angle
        same_turn_sign = (far_angle * tip_turn >= 0.0)
        u_metric = max(abs(curv_proxy), abs(tip_turn) / 1.05 if same_turn_sign else 0.0)
        # u_level: 0 = duong thang / cua nhe, 1 = cua chu U gat -> danh HET LAI (100% tran servo)
        u_level = float(np.clip((u_metric - 0.36) / 0.34, 0.0, 1.0))
        u_sign = 1.0 if (far_angle + 0.5 * tip_turn) >= 0.0 else -1.0

        # Rut ngan tam ngam khi vao cua chu U (an thanh CUA SOM) - KHONG rut qua ngan tren doan thang
        cua_som = max(0.0, min(1.0, (0.30 - float(self.curve_enter)) / 0.24))
        max_shrink = 0.28 + 0.12 * cua_som
        shrink = max_shrink * min(1.0, abs(curv_proxy) * 1.10)
        if self.debug_obstacle_hole:
            shrink = max(shrink, 0.24)
        eff_ld = max(46.0, min(total_s, ld_px * (1.0 - shrink)))
        t_idx = int(np.searchsorted(cum_s, eff_ld))
        t_idx = min(t_idx, len(pts) - 1)
        tx, ty = float(pts[t_idx, 0]), float(pts[t_idx, 1])
        self.debug_target = (tx, ty)

        cam_off_bev = self.cam_near_offset / self.scale_y
        dx_m = (tx - PROC_W * 0.5) * self.scale_x
        dy_m = ((PROC_H + cam_off_bev) - ty) * self.scale_y
        dist_m = max(10.0, math.hypot(dx_m, dy_m))
        alpha = math.atan2(dx_m, dy_m)
        wb_m = car_len * 0.62
        delta = math.atan2(2.0 * wb_m * math.sin(alpha), dist_m) * pp_gain

        # Dem day mep gan mui xe o 160x120
        if dist_fast is not None and self.edge_margin_px > 0.0 and self.edge_push_gain > 0.0:
            half_f = BEV_W // 2
            left_out = outside_fast[BEV_H - 21:BEV_H - 3, :half_f]
            right_out = outside_fast[BEV_H - 21:BEV_H - 3, half_f:]
            has_l = left_out.any(axis=1)
            has_r = right_out.any(axis=1)
            d_l = (float(np.median(left_out[has_l, ::-1].argmax(axis=1) + 1)) * inv_scale
                   if has_l.any() else 999.0)
            d_r = (float(np.median(right_out[has_r].argmax(axis=1) + 1)) * inv_scale
                   if has_r.any() else 999.0)
            m_px = self.edge_margin_px
            if d_r < m_px and d_r + 6.0 < d_l:
                delta -= self.edge_push_gain * ((m_px - d_r) / m_px) * wheel_at_lim_rad
            elif d_l < m_px and d_l + 6.0 < d_r:
                delta += self.edge_push_gain * ((m_px - d_l) / m_px) * wheel_at_lim_rad

        # Nguyen ly dua cua chuan:
        #   - `far_angle` / `u_level` (nhin xa) -> CHI DUNG DE PHANH SOM truoc cua (khong be som cat dao do!).
        #   - `raw_u` (goc Pure Pursuit tai diem ngam gan) -> khi xe THUC SU vao cua chu U
        #     (|raw_u| > 0.30), tang luc luy tien theo `LAI CUA GAT` (1.40x) va cam ket
        #     be toi HET LAI 100% tran servo (+-1.00) de om chat qua cua chu U!
        gain_abs = max(1e-6, abs(self.drv_steer_gain))
        steer_ref = min(steer_span, max(0.10, self.servo_ref_at_limit) / gain_abs)
        raw_u = delta / max(0.15, wheel_at_lim_rad)
        u_turn_gain = max(1.0, float(ov.get('road_turn_steer_gain', self.road_turn_steer_gain)))
        # Chi boi lai khi chinh goc Pure Pursuit `|raw_u|` dang vao cua (khong boi tren duong thang!)
        if u_turn_gain > 1.0 and abs(raw_u) > 0.28:
            ramp_u = min(1.0, (abs(raw_u) - 0.28) / 0.35)
            raw_u = raw_u * (1.0 + (u_turn_gain - 1.0) * ramp_u)
        # Khi xe dang o GIUA cua chu U gat (ca `|raw_u|` lan `u_level` cung cao & cung huong):
        # day thang toi 100% tran `steer_span` (HET LAI) va tang do nhay EMA de kip om cua!
        in_u_apex = 0.0
        if abs(raw_u) > 0.55 and u_level > 0.35 and (raw_u * u_sign > 0.0):
            in_u_apex = min(1.0, (abs(raw_u) - 0.55) / 0.35) * u_level
        raw_steer = raw_u * steer_ref
        if in_u_apex > 0.0:
            full_target = steer_span * (0.80 + 0.20 * in_u_apex)
            if abs(raw_steer) < full_target:
                raw_steer = (1.0 if raw_steer >= 0.0 else -1.0) * full_target
        self.steer_wanted = float(raw_steer)
        steer = float(np.clip(raw_steer, -steer_span, steer_span))
        u_norm = float(np.clip(steer / max(1e-6, steer_span), -1.0, 1.0))
        a = max(0.1, min(1.0, self.steer_smooth + 0.20 * in_u_apex))
        steer = a * steer + (1.0 - a) * self._prev_steer
        self._prev_steer = steer

        corner_intensity = max(abs(curv_proxy), u_level * 0.85, min(1.0, abs(raw_u)))
        if self.debug_obstacle_hole:
            corner_intensity = max(corner_intensity, 0.70)
        self.mode = CORNER if corner_intensity >= 0.26 else STRAIGHT
        target_v = self.v_straight - (self.v_straight - self.v_corner) * min(1.0, corner_intensity * 1.35)
        # Phanh truoc & trong dinh cua chu U theo thanh GIAM TOC TRUOC CUA (pre_corner_decel)
        decel = max(0.0, min(1.0, float(ov.get('pre_corner_decel', self.pre_corner_decel))))
        if decel > 0.0 and corner_intensity > 0.38 and self.v_corner > self.v_min:
            sharp = min(1.0, (corner_intensity - 0.38) / 0.50)
            target_v -= (self.v_corner - self.v_min) * (0.75 * decel * sharp)
        v_lo = self.v_min if decel > 0.0 else min(self.v_corner, self.v_straight)
        target_v = max(self.v_min, min(self.v_max, max(v_lo, min(self.v_straight, target_v))))

        rate = self.throttle_rise_rate if target_v > self._throttle else self.throttle_fall_rate
        dv = target_v - self._throttle
        self._throttle += max(-rate * dt, min(rate * dt, dv))
        self._throttle = max(0.0, min(self.v_max, self._throttle))

        cte = float(np.clip((pts[0, 0] - PROC_W * 0.5) / (PROC_W * 0.5), -1.0, 1.0))
        look = float(np.clip((tx - PROC_W * 0.5) / (PROC_W * 0.5), -1.0, 1.0))
        return {
            'found': True,
            'cte': cte,
            'lookahead': look,
            'curvature': curv_proxy,
            'steer': steer,
            'throttle': self._throttle,
            'mode': self.mode,
            'n_bands': min(10, len(pts)),
            'obstacle_hole': self.debug_obstacle_hole,
        }

    def render_bev_debug(self):
        """Tra ve anh BEV 320x240 BGR de hien thi truc quan tren tuning_ui.py / sim_app."""
        bev = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
        if self._debug_road_fast is not None:
            up = cv2.resize(
                self._debug_road_fast, (PROC_W, PROC_H), interpolation=cv2.INTER_NEAREST)
            bev[up > 0] = (28, 115, 28)
        cv2.line(bev, (PROC_W // 2, 0), (PROC_W // 2, PROC_H), (90, 90, 90), 1)
        if len(self.debug_bev_pts) >= 2:
            pts = np.rint(np.array(self.debug_bev_pts)).astype(np.int32)
            cv2.polylines(bev, [pts], False, (0, 235, 255), 2)
        if self.debug_target is not None:
            tx = int(round(self.debug_target[0]))
            ty = int(round(self.debug_target[1]))
            cv2.line(bev, (PROC_W // 2, PROC_H - 1), (tx, ty), (255, 220, 60), 1)
            cv2.circle(bev, (tx, ty), 6, (0, 80, 255), -1)
            cv2.circle(bev, (tx, ty), 8, (255, 255, 255), 1)
        return bev
