# -*- coding: utf-8 -*-
r"""Train & benchmark 5 kien truc LaneNet/LaneTiny Option B (3 kenh + reg) tren GPU CUDA.

Moi model nhan CUNG input & tra CUNG output tuong thich 100% voi TensorRT 7.1.3 (Jetson Nano):
  - Input : x float32 (1, 3, 128, 256) BGR thang 0..255 (crop tu roi_top=0.34)
            (chuan hoa /255 va CoordConv nam BEN TRONG graph ONNX)
  - Output:
      seg : (1, 3, 64, 128) logits [Ch0=drivable_road, Ch1=white_line, Ch2=centerline]
      reg : (1, 3) [cte, cte_lookahead, curvature] trong [-1, 1]

Chia du lieu chong ro ri (Technical Paper FA2026 §V-A):
  - Stratified Contiguous Block Split (~60% Train / ~20% Val / ~20% Test)
  - Buffer gap = 5 frames (~1s) o moi ranh gioi chuyen tap

Cach chay:
  .\.venv_cuda\Scripts\python.exe tools/train_lane_option_b.py --data data/lane_option_b --epochs 80
"""

from __future__ import print_function

import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_W, IMG_H = 640, 480
ROI_TOP = 0.34
ROI_Y0 = int(IMG_H * ROI_TOP)   # 163
IN_W, IN_H = 256, 128
SEG_W, SEG_H = 128, 64
PROC_W, PROC_H = 320, 240

# BEV homography de tinh ground truth reg [cte, cte_lookahead, curvature] TRUC TIEP
# tu mask centerline da duyet (dong bo voi luc inference tren xe!)
WARP_SRC_NORM = [[0.00, 0.92], [1.00, 0.92], [0.66, 0.42], [0.33, 0.42]]
WARP_DST_MARGIN = 0.32


def _build_bev_matrix(w, h):
    src = np.float32([[p[0] * w, p[1] * h] for p in WARP_SRC_NORM])
    dst = np.float32([
        [WARP_DST_MARGIN * w, h],
        [(1.0 - WARP_DST_MARGIN) * w, h],
        [(1.0 - WARP_DST_MARGIN) * w, 0.0],
        [WARP_DST_MARGIN * w, 0.0],
    ])
    return cv2.getPerspectiveTransform(src, dst)


BEV_MAT_PROC = _build_bev_matrix(PROC_W, PROC_H)


def compute_reg_from_centerline_mask(center_mask_640x480):
    """Tinh [cte, cte_lookahead, curvature] tu mask centerline 640x480 theo dung duong BEV."""
    small = cv2.resize(center_mask_640x480, (PROC_W, PROC_H), interpolation=cv2.INTER_NEAREST)
    bev = cv2.warpPerspective(small, BEV_MAT_PROC, (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)
    n_bands = 10
    band_h = PROC_H // n_bands
    pts = []
    for b in range(n_bands):
        y1 = PROC_H - b * band_h
        y0 = max(0, y1 - band_h)
        band = bev[y0:y1, :]
        xs = np.where(band > 127)[1]
        if len(xs) >= 8:
            pts.append(((b + 0.5) / float(n_bands), float(np.median(xs))))
    if len(pts) < 2:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    ts = np.array([p[0] for p in pts], dtype=np.float64)
    xs = np.array([p[1] for p in pts], dtype=np.float64)
    deg = 2 if len(pts) >= 3 else 1
    co = np.polyfit(ts, xs, deg)
    if deg == 1:
        co = np.array([0.0, co[0], co[1]])
    half = PROC_W / 2.0
    t_max = float(ts.max())
    t_look = min(0.60, t_max)
    cte = float(np.clip((co[2] - half) / half, -1.0, 1.0))
    look = float(np.clip((co[0] * t_look * t_look + co[1] * t_look + co[2] - half) / half, -1.0, 1.0))
    curv = float(np.clip((co[0] * t_max * t_max) / half, -1.0, 1.0))
    return np.array([cte, look, curv], dtype=np.float32)


def build_block_split(n_frames, gap=5):
    """Stratified Contiguous Block Split (~60/20/20) voi khoang dem gap=5 frame chong ro ri."""
    # 10 khoi lien tiep:
    #   Val  = Block 1, Block 6 (~20% gom thang + cua phai/trai)
    #   Test = Block 3, Block 8 (~20% gom cua trai/thang + cua phai/trai)
    #   Train = Block 0, 2, 4, 5, 7, 9 (~60% gom du thang + cua trai + cua phai)
    block_role = {}
    for b in range(10):
        if b in (1, 6):
            block_role[b] = 'val'
        elif b in (3, 8):
            block_role[b] = 'test'
        else:
            block_role[b] = 'train'

    bounds = [int(round(b * n_frames / 10.0)) for b in range(11)]
    frame_role = [''] * n_frames
    for b in range(10):
        for i in range(bounds[b], bounds[b + 1]):
            frame_role[i] = block_role[b]

    # Loai bo khoang dem `gap` frame tai moi ranh gioi chuyen giua hai tap khac nhau
    keep = [True] * n_frames
    for i in range(n_frames - 1):
        if frame_role[i] != frame_role[i + 1]:
            # Bo gap frame ve ca hai phia de khoang cach toi thieu >= 2*gap?
            # Bo tong `gap` frame quanh bien (3 phia truoc, 2 phia sau -> cach nhau >= 6 frame = 1.2s)
            left_drop = (gap + 1) // 2
            right_drop = gap // 2
            for j in range(max(0, i - left_drop + 1), min(n_frames, i + 1 + right_drop)):
                keep[j] = False

    splits = {'train': [], 'val': [], 'test': [], 'dropped_gap': []}
    for i in range(n_frames):
        if not keep[i]:
            splits['dropped_gap'].append(i)
        else:
            splits[frame_role[i]].append(i)
    return splits


# =============================================================================
# CAC KHOI MANG TUONG THICH 100% TENSORRT 7.1.3 (Conv/DWConv/BN/ReLU/Pool/Resize)
# =============================================================================

class InputPrep(nn.Module):
    """Chuan hoa 0..255 -> 0..1 va noi CoordConv (x, y) BEN TRONG graph ONNX."""

    def __init__(self, use_coord=True):
        super(InputPrep, self).__init__()
        self.use_coord = use_coord
        if use_coord:
            ys = np.linspace(-1.0, 1.0, IN_H, dtype=np.float32)
            xs = np.linspace(-1.0, 1.0, IN_W, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            grid = np.stack([gx, gy], axis=0)[None, :, :, :]  # (1, 2, 128, 256)
            self.register_buffer('coord', torch.from_numpy(grid))

    def forward(self, x):
        # x: (B, 3, 128, 256) float32 0..255
        xn = x * (1.0 / 255.0)
        if not self.use_coord:
            return xn
        if torch.onnx.is_in_onnx_export():
            # Batch=1 tinh khi export ONNX -> tranh Expand/Where/Shape khong tuong thich TRT 7.1.3
            return torch.cat([xn, self.coord], dim=1)
        b = xn.shape[0]
        coord = self.coord.expand(b, -1, -1, -1)
        return torch.cat([xn, coord], dim=1)


class ConvBNReLU(nn.Module):
    def __init__(self, ci, co, k=3, s=1, p=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ci, co, k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DWBlock(nn.Module):
    def __init__(self, ci, co, s=1):
        super(DWBlock, self).__init__()
        self.net = nn.Sequential(
            ConvBNReLU(ci, ci, k=3, s=s, p=1, groups=ci),
            ConvBNReLU(ci, co, k=1, s=1, p=0),
            ConvBNReLU(co, co, k=3, s=1, p=1, groups=co),
            ConvBNReLU(co, co, k=1, s=1, p=0),
        )

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):
    """Squeeze-Excitation dung GlobalAveragePool (AdaptiveAvgPool2d(1,1)) tuong thich TRT 7.1.3."""

    def __init__(self, c, r=4):
        super(SEBlock, self).__init__()
        mid = max(8, c // r)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, (1, 1))
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w), inplace=True)))
        return x * w


class InvertedResSE(nn.Module):
    """MobileNetV2 Inverted Residual + SE attention."""

    def __init__(self, ci, co, s=1, expand=3):
        super(InvertedResSE, self).__init__()
        mid = ci * expand
        self.use_res = (s == 1 and ci == co)
        self.net = nn.Sequential(
            ConvBNReLU(ci, mid, k=1, s=1, p=0),
            ConvBNReLU(mid, mid, k=3, s=s, p=1, groups=mid),
            SEBlock(mid),
            nn.Conv2d(mid, co, 1, bias=False),
            nn.BatchNorm2d(co),
        )

    def forward(self, x):
        out = self.net(x)
        if self.use_res:
            return F.relu(out + x, inplace=True)
        return F.relu(out, inplace=True)


class RegHead(nn.Module):
    """Head hoi quy [cte, cte_lookahead, curvature] dung GlobalAveragePool + Linear (TRT 7.1.3)."""

    def __init__(self, c_in):
        super(RegHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
            nn.Tanh(),
        )

    def forward(self, feat):
        v = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return self.fc(v)


# --- MODEL 1 & 2: LaneTiny-Fast (base=16) & LaneTiny-Base (base=24) ---
class LaneTinyDWUNet(nn.Module):
    """Stride-2 stem -> encoder/decoder DW-UNet -> seg (3, 64, 128) + reg (3)."""

    def __init__(self, base=24, use_coord=True):
        super(LaneTinyDWUNet, self).__init__()
        in_c = 5 if use_coord else 3
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 6
        self.prep = InputPrep(use_coord=use_coord)
        # Stem stride-2: 128x256 -> 64x128
        self.stem = nn.Sequential(
            ConvBNReLU(in_c, c1, k=3, s=2, p=1),
            DWBlock(c1, c1, s=1),
        )
        self.enc2 = DWBlock(c1, c2, s=2)   # 32x64
        self.enc3 = DWBlock(c2, c3, s=2)   # 16x32
        self.enc4 = DWBlock(c3, c4, s=2)   # 8x16

        self.dec3 = DWBlock(c4 + c3, c3, s=1)
        self.dec2 = DWBlock(c3 + c2, c2, s=1)
        self.dec1 = DWBlock(c2 + c1, c1, s=1)
        self.seg_head = nn.Conv2d(c1, 3, 1)
        self.reg_head = RegHead(c4)

    def forward(self, x):
        x = self.prep(x)
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        u3 = F.interpolate(e4, scale_factor=2.0, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(d3, scale_factor=2.0, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, scale_factor=2.0, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.seg_head(d1), self.reg_head(e4)


# --- MODEL 3: LaneTiny-Wide (Hybrid Standard Conv + DWConv, base=32) ---
class LaneTinyWide(nn.Module):
    """Phien ban LaneTiny rong hon (tuong duong ngan sach lane_tiny.onnx cu ~350k params)."""

    def __init__(self, base=32):
        super(LaneTinyWide, self).__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 6
        self.prep = InputPrep(use_coord=True)
        self.stem = nn.Sequential(
            ConvBNReLU(5, c1, k=3, s=2, p=1),
            ConvBNReLU(c1, c1, k=3, s=1, p=1),
        )
        self.enc2 = nn.Sequential(
            ConvBNReLU(c1, c2, k=3, s=2, p=1),
            DWBlock(c2, c2, s=1),
        )
        self.enc3 = DWBlock(c2, c3, s=2)
        self.enc4 = DWBlock(c3, c4, s=2)

        self.dec3 = DWBlock(c4 + c3, c3, s=1)
        self.dec2 = DWBlock(c3 + c2, c2, s=1)
        self.dec1 = DWBlock(c2 + c1, c1, s=1)
        self.seg_head = nn.Conv2d(c1, 3, 1)
        self.reg_head = RegHead(c4)

    def forward(self, x):
        x = self.prep(x)
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        u3 = F.interpolate(e4, scale_factor=2.0, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(d3, scale_factor=2.0, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, scale_factor=2.0, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.seg_head(d1), self.reg_head(e4)


# --- MODEL 4: LaneTiny-ResSE (MobileNetV2 Inverted-Residual + SE Attention) ---
class LaneTinyResSE(nn.Module):
    """Encoder MobileNetV2-SE nhe (expand=2, base=16 -> ~160k params) + DW Decoder cho Edge."""

    def __init__(self, base=16):
        super(LaneTinyResSE, self).__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 6
        self.prep = InputPrep(use_coord=True)
        self.stem = nn.Sequential(
            ConvBNReLU(5, c1, k=3, s=2, p=1),
            DWBlock(c1, c1, s=1),
        )
        self.enc2 = InvertedResSE(c1, c2, s=2, expand=2)
        self.enc3 = nn.Sequential(
            InvertedResSE(c2, c3, s=2, expand=2),
            InvertedResSE(c3, c3, s=1, expand=2),
        )
        self.enc4 = nn.Sequential(
            InvertedResSE(c3, c4, s=2, expand=2),
            InvertedResSE(c4, c4, s=1, expand=2),
        )

        self.dec3 = DWBlock(c4 + c3, c3, s=1)
        self.dec2 = DWBlock(c3 + c2, c2, s=1)
        self.dec1 = DWBlock(c2 + c1, c1, s=1)
        self.seg_head = nn.Conv2d(c1, 3, 1)
        self.reg_head = RegHead(c4)

    def forward(self, x):
        x = self.prep(x)
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        u3 = F.interpolate(e4, scale_factor=2.0, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(d3, scale_factor=2.0, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, scale_factor=2.0, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.seg_head(d1), self.reg_head(e4)


# --- MODEL 5: FastSCNN-Lane (Dual-Branch Spatial Detail + Pyramid Context) ---
class PyramidPooling(nn.Module):
    """PPM dung fixed AvgPool2d (tuong thich tuyet doi TRT 7.1.3) cho feature 8x16."""

    def __init__(self, c_in, c_out):
        super(PyramidPooling, self).__init__()
        mid = c_in // 4
        self.p1 = nn.Sequential(nn.AvgPool2d(kernel_size=(8, 16)), ConvBNReLU(c_in, mid, k=1, p=0))
        self.p2 = nn.Sequential(nn.AvgPool2d(kernel_size=(4, 8)), ConvBNReLU(c_in, mid, k=1, p=0))
        self.p3 = nn.Sequential(nn.AvgPool2d(kernel_size=(2, 4)), ConvBNReLU(c_in, mid, k=1, p=0))
        self.fuse = ConvBNReLU(c_in + mid * 3, c_out, k=1, p=0)

    def forward(self, x):
        b1 = F.interpolate(self.p1(x), scale_factor=(8.0, 16.0), mode='bilinear', align_corners=False)
        b2 = F.interpolate(self.p2(x), scale_factor=(4.0, 8.0), mode='bilinear', align_corners=False)
        b3 = F.interpolate(self.p3(x), scale_factor=(2.0, 4.0), mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x, b1, b2, b3], dim=1))


class FastSCNNLane(nn.Module):
    """Kien truc Fast-SCNN + CoordConv chuyen dung cho Edge Real-Time Road/Lane Segmentation."""

    def __init__(self):
        super(FastSCNNLane, self).__init__()
        self.prep = InputPrep(use_coord=True)
        # Spatial Detail Branch (giu bien vach trang sac net o 64x128 & 32x64)
        self.detail1 = nn.Sequential(
            ConvBNReLU(5, 32, k=3, s=2, p=1),     # 64x128
            DWBlock(32, 32, s=1),
        )
        self.detail2 = DWBlock(32, 48, s=2)       # 32x64

        # Global Context Branch (nhin ngu canh rong cua ca khuc cua)
        self.ctx1 = nn.Sequential(
            InvertedResSE(48, 64, s=2, expand=3),  # 16x32
            InvertedResSE(64, 64, s=1, expand=3),
        )
        self.ctx2 = nn.Sequential(
            InvertedResSE(64, 112, s=2, expand=3), # 8x16
            InvertedResSE(112, 112, s=1, expand=3),
            PyramidPooling(112, 96),
        )

        # Feature Fusion Module (FFM)
        self.ffm_low = DWBlock(96 + 48, 64, s=1)   # tai 32x64
        self.ffm_hi = DWBlock(64 + 32, 40, s=1)    # tai 64x128
        self.seg_head = nn.Conv2d(40, 3, 1)
        self.reg_head = RegHead(96)

    def forward(self, x):
        x = self.prep(x)
        s1 = self.detail1(x)
        s2 = self.detail2(s1)
        c1 = self.ctx1(s2)
        c2 = self.ctx2(c1)

        u_low = F.interpolate(c2, scale_factor=4.0, mode='bilinear', align_corners=False)
        f_low = self.ffm_low(torch.cat([u_low, s2], dim=1))
        u_hi = F.interpolate(f_low, scale_factor=2.0, mode='bilinear', align_corners=False)
        f_hi = self.ffm_hi(torch.cat([u_hi, s1], dim=1))
        return self.seg_head(f_hi), self.reg_head(c2)


MODEL_REGISTRY = {
    'lanetiny_fast': lambda: LaneTinyDWUNet(base=16, use_coord=True),
    'lanetiny_base': lambda: LaneTinyDWUNet(base=24, use_coord=True),
    'lanetiny_wide': lambda: LaneTinyWide(base=32),
    'lanetiny_resse': lambda: LaneTinyResSE(base=16),
    'fastscnn_lane': lambda: FastSCNNLane(),
}


# =============================================================================
# DATA LOADING & AUGMENTATION
# =============================================================================

def augment_batch_gpu(bx, by, br):
    """Augmentation 100% tren GPU CUDA (nhanh hon ~20x so voi vong lap CPU OpenCV)."""
    B = bx.shape[0]
    dev = bx.device

    # 1. Horizontal flip (50%) -> doi dau reg [cte, look, curv]
    flip_mask = torch.rand(B, device=dev) < 0.5
    if flip_mask.any():
        bx = bx.clone()
        by = by.clone()
        br = br.clone()
        bx[flip_mask] = torch.flip(bx[flip_mask], dims=[3])
        by[flip_mask] = torch.flip(by[flip_mask], dims=[3])
        br[flip_mask] = -br[flip_mask]

    # 2. Random affine tren GPU (xoay +-6 deg, scale 0.93..1.07, dich ngang/doc)
    aff_mask = torch.rand(B, device=dev) < 0.60
    if aff_mask.any():
        angles = (torch.rand(B, device=dev) * 12.0 - 6.0) * (np.pi / 180.0)
        scales = torch.rand(B, device=dev) * 0.14 + 0.93
        tx = torch.rand(B, device=dev) * 0.10 - 0.05
        ty = torch.rand(B, device=dev) * 0.06 - 0.03
        angles = torch.where(aff_mask, angles, torch.zeros_like(angles))
        scales = torch.where(aff_mask, scales, torch.ones_like(scales))
        tx = torch.where(aff_mask, tx, torch.zeros_like(tx))
        ty = torch.where(aff_mask, ty, torch.zeros_like(ty))

        cos_a = torch.cos(angles) / scales
        sin_a = torch.sin(angles) / scales
        theta = torch.zeros((B, 2, 3), device=dev, dtype=bx.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = -tx
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = -ty

        grid_x = F.affine_grid(theta, bx.size(), align_corners=False)
        grid_y = F.affine_grid(theta, by.size(), align_corners=False)
        bx = F.grid_sample(bx, grid_x, mode='bilinear', padding_mode='border', align_corners=False)
        by = F.grid_sample(by, grid_y, mode='nearest', padding_mode='zeros', align_corners=False)
        br = br.clone()
        br[:, 0] = torch.clamp(br[:, 0] + tx, -1.0, 1.0)
        br[:, 1] = torch.clamp(br[:, 1] + tx, -1.0, 1.0)

    # 3. Photometric jitter (do sang, tuong phan, can bang trang)
    alpha = (torch.rand((B, 1, 1, 1), device=dev) * 0.70 + 0.65)
    beta = (torch.rand((B, 1, 1, 1), device=dev) * 70.0 - 35.0)
    gains = (torch.rand((B, 3, 1, 1), device=dev) * 0.24 + 0.88)
    bx = torch.clamp((bx * alpha + beta) * gains, 0.0, 255.0)

    # 4. Random directional shadow gradient (hoc bat vach trang trong bong toi!)
    sh_mask = (torch.rand(B, device=dev) < 0.55).view(B, 1, 1, 1)
    if sh_mask.any():
        xs = torch.linspace(0.0, 1.0, IN_W, device=dev).view(1, 1, 1, IN_W)
        ys = torch.linspace(0.0, 1.0, IN_H, device=dev).view(1, 1, IN_H, 1)
        ang = torch.rand((B, 1, 1, 1), device=dev) * (2.0 * np.pi)
        grad = torch.cos(ang) * xs + torch.sin(ang) * ys
        gmin = grad.amin(dim=(2, 3), keepdim=True)
        gmax = grad.amax(dim=(2, 3), keepdim=True)
        grad = (grad - gmin) / (gmax - gmin + 1e-5)
        dark = torch.rand((B, 1, 1, 1), device=dev) * 0.35 + 0.45
        factor = torch.where(sh_mask, dark + (1.0 - dark) * grad, torch.ones_like(grad))
        bx = torch.clamp(bx * factor, 0.0, 255.0)

    # 5. Random synthetic glare blob (chong loa den tran tren bat den!)
    gl_mask = (torch.rand(B, device=dev) < 0.50).view(B, 1, 1, 1)
    if gl_mask.any():
        xs = torch.linspace(0.0, 1.0, IN_W, device=dev).view(1, 1, 1, IN_W)
        ys = torch.linspace(0.0, 1.0, IN_H, device=dev).view(1, 1, IN_H, 1)
        cx = torch.rand((B, 1, 1, 1), device=dev) * 0.8 + 0.1
        cy = torch.rand((B, 1, 1, 1), device=dev) * 0.8 + 0.15
        sx = torch.rand((B, 1, 1, 1), device=dev) * 0.12 + 0.05
        sy = torch.rand((B, 1, 1, 1), device=dev) * 0.12 + 0.05
        blob = torch.exp(-(((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2))
        strength = torch.rand((B, 1, 1, 1), device=dev) * 85.0 + 45.0
        bx = torch.clamp(bx + torch.where(gl_mask, blob * strength, torch.zeros_like(blob)), 0.0, 255.0)

    return bx, by, br


def multitask_loss(seg_logits, reg_pred, seg_gt, reg_gt):
    ch_w = torch.tensor([1.0, 2.0, 2.5], dtype=seg_logits.dtype, device=seg_logits.device).view(1, 3, 1, 1)
    bce = (F.binary_cross_entropy_with_logits(seg_logits, seg_gt, reduction='none') * ch_w).mean()

    probs = torch.sigmoid(seg_logits)
    inter = (probs * seg_gt).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + seg_gt.sum(dim=(0, 2, 3)) + 1e-5
    dice_per_c = 1.0 - (2.0 * inter + 1e-5) / denom
    cw = torch.tensor([1.0, 2.0, 2.5], dtype=seg_logits.dtype, device=seg_logits.device)
    dice = (dice_per_c * cw).sum() / cw.sum()

    seg_loss = 0.5 * bce + 0.5 * dice
    reg_loss = F.smooth_l1_loss(reg_pred, reg_gt, beta=0.1)
    return seg_loss + 0.35 * reg_loss


def evaluate_model(model, items, device):
    model.eval()
    inter = np.zeros(3, dtype=np.float64)
    union = np.zeros(3, dtype=np.float64)
    reg_abs_err = []
    seg_cte_err = []

    with torch.no_grad():
        for it in items:
            x = torch.from_numpy(it['img'].transpose(2, 0, 1)[None].astype(np.float32)).to(device)
            seg_logits, reg_pred = model(x)
            seg_p = (seg_logits[0].cpu().numpy() > 0.0).astype(np.float32)
            reg_p = reg_pred[0].cpu().numpy()
            gt_s = it['masks']
            gt_r = it['reg']

            for c in range(3):
                inter[c] += float(np.sum((seg_p[c] > 0.5) & (gt_s[c] > 0.5)))
                union[c] += float(np.sum((seg_p[c] > 0.5) | (gt_s[c] > 0.5)))
            reg_abs_err.append(np.abs(reg_p - gt_r))

            # Tinh CTE thuc te tu mask centerline du doan (giong lane_cnn.py tren xe!)
            c_up = cv2.resize((seg_p[2] * 255).astype(np.uint8), (IMG_W, IMG_H - ROI_Y0), interpolation=cv2.INTER_NEAREST)
            c_full = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            c_full[ROI_Y0:, :] = c_up
            pred_reg_from_seg = compute_reg_from_centerline_mask(c_full)
            seg_cte_err.append(np.abs(pred_reg_from_seg - gt_r))

    ious = inter / np.maximum(1.0, union)
    reg_mae = np.mean(reg_abs_err, axis=0)
    seg_mae = np.mean(seg_cte_err, axis=0)
    return {
        'road_iou': float(ious[0]),
        'white_iou': float(ious[1]),
        'center_iou': float(ious[2]),
        'miou': float(np.mean(ious)),
        'reg_cte_mae': float(reg_mae[0]),
        'reg_look_mae': float(reg_mae[1]),
        'reg_curv_mae': float(reg_mae[2]),
        'seg_cte_mae': float(seg_mae[0]),
        'seg_look_mae': float(seg_mae[1]),
    }


def measure_flops_and_speed(model, device):
    """Do chinh xac MFLOPs (MACs*2) va latency tren GPU + CPU ma khong can xuat ONNX truoc."""
    model.eval()
    macs = [0]
    hooks = []

    def _conv_hook(mod, inp, out):
        # out: (B, Cout, H, W), kernel: (Cout, Cin/g, kh, kw)
        b, co, h, w = out.shape
        ci_per_g = mod.in_channels // mod.groups
        kh, kw = mod.kernel_size
        macs[0] += int(co * h * w * ci_per_g * kh * kw)

    def _lin_hook(mod, inp, out):
        macs[0] += int(mod.in_features * mod.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_lin_hook))

    x_gpu = torch.zeros((1, 3, IN_H, IN_W), dtype=torch.float32, device=device)
    with torch.no_grad():
        model(x_gpu)
    for h in hooks:
        h.remove()
    mflops = (macs[0] * 2.0) / 1e6

    with torch.no_grad():
        for _ in range(20):
            model(x_gpu)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            model(x_gpu)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000.0 / 100.0

    model_cpu = model.to('cpu').eval()
    x_cpu = torch.zeros((1, 3, IN_H, IN_W), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(5):
            model_cpu(x_cpu)
        t0 = time.perf_counter()
        for _ in range(25):
            model_cpu(x_cpu)
        cpu_ms = (time.perf_counter() - t0) * 1000.0 / 25.0
    model.to(device).eval()
    return mflops, gpu_ms, cpu_ms


def export_onnx(model, out_path):
    model_cpu = model.to('cpu').eval()
    dummy = torch.zeros((1, 3, IN_H, IN_W), dtype=torch.float32)
    # Use legacy TorchScript exporter (opset 11) for maximum TensorRT 7.1.3 compatibility
    kwargs = {
        'input_names': ['input'],
        'output_names': ['seg', 'reg'],
        'opset_version': 11,
        'do_constant_folding': True,
    }
    try:
        torch.onnx.export(model_cpu, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, dummy, out_path, **kwargs)
    import onnx
    om = onnx.load(out_path)
    # Loai bo node Identity thua va ep ir_version <= 6 de tuong thich tuyet doi TensorRT 7.1.3
    id_map = {}
    new_nodes = []
    for n in om.graph.node:
        if n.op_type == 'Identity':
            src = n.input[0]
            while src in id_map:
                src = id_map[src]
            id_map[n.output[0]] = src
        else:
            for i, inp in enumerate(n.input):
                if inp in id_map:
                    n.input[i] = id_map[inp]
            new_nodes.append(n)
    if id_map:
        del om.graph.node[:]
        om.graph.node.extend(new_nodes)
        for out in om.graph.output:
            if out.name in id_map:
                out.name = id_map[out.name]
    if om.ir_version > 6:
        om.ir_version = 6
    onnx.checker.check_model(om)
    onnx.save(om, out_path)
    return os.path.getsize(out_path) / 1024.0


def main():
    parser = argparse.ArgumentParser(description='Train & benchmark 5 Option B lane models on CUDA')
    parser.add_argument('--data', default='data/lane_option_b')
    parser.add_argument('--out-dir', default='models/option_b_bench')
    parser.add_argument('--models', nargs='+', default=[
        'lanetiny_fast', 'lanetiny_base', 'lanetiny_wide', 'lanetiny_resse', 'fastscnn_lane'
    ])
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3.5e-3)
    parser.add_argument('--gap', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--export-onnx', action='store_true', help='Chi xuat .onnx khi duoc bat (mac dinh tat luc benchmark)')
    parser.add_argument('--resume', action='store_true', help='Bo qua model da co .pt trong out-dir')
    parser.add_argument('--visual-out', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print('=== Option B Multi-Model Benchmark ===')
    print('Device:', device, ('(' + torch.cuda.get_device_name(0) + ', FP32 fast DWConv)') if device.type == 'cuda' else '')

    data_dir = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(data_dir, 'metadata.json'), 'r') as fh:
        meta = json.load(fh)

    frames = [f for f in meta['frames'] if f['status'] == 'approved']
    splits = build_block_split(len(frames), gap=args.gap)
    print('Split (gap=%d): Train=%d | Val=%d | Test=%d | Dropped_gap=%d' % (
        args.gap, len(splits['train']), len(splits['val']), len(splits['test']), len(splits['dropped_gap'])
    ))

    # Luu split de truy vet cho Technical Paper!
    split_record = {
        'strategy': 'stratified_contiguous_block_60_20_20',
        'gap_frames': args.gap,
        'val_blocks': [1, 6],
        'test_blocks': [3, 8],
        'train_blocks': [0, 2, 4, 5, 7, 9],
        'counts': {k: len(v) for k, v in splits.items()},
        'train_ids': [frames[i]['id'] for i in splits['train']],
        'val_ids': [frames[i]['id'] for i in splits['val']],
        'test_ids': [frames[i]['id'] for i in splits['test']],
    }
    with open(os.path.join(out_dir, 'split.json'), 'w') as fh:
        json.dump(split_record, fh, indent=2)

    # Preload toan bo 1039 anh vao RAM (chi ~120 MB -> GPU train 80 epochs chi mat ~25s/model!)
    print('Dang nap %d anh + 3 mask + tinh reg ground truth vao RAM...' % len(frames))
    t_load = time.time()
    all_items = []
    for rec in frames:
        fid = rec['id']
        img = cv2.imread(os.path.join(data_dir, 'images', rec['image']))
        r = cv2.imread(os.path.join(data_dir, 'masks', 'road', fid + '.png'), 0)
        w = cv2.imread(os.path.join(data_dir, 'masks', 'white', fid + '.png'), 0)
        c = cv2.imread(os.path.join(data_dir, 'masks', 'centerline', fid + '.png'), 0)

        img_roi = cv2.resize(img[ROI_Y0:, :, :], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        m_r = (cv2.resize(r[ROI_Y0:, :], (SEG_W, SEG_H), interpolation=cv2.INTER_AREA) > 127).astype(np.float32)
        m_w = (cv2.resize(w[ROI_Y0:, :], (SEG_W, SEG_H), interpolation=cv2.INTER_AREA) > 64).astype(np.float32)
        m_c = (cv2.resize(c[ROI_Y0:, :], (SEG_W, SEG_H), interpolation=cv2.INTER_AREA) > 64).astype(np.float32)
        reg = compute_reg_from_centerline_mask(c)

        all_items.append({
            'id': fid,
            'full_img': img,
            'img': img_roi,
            'masks': np.stack([m_r, m_w, m_c], axis=0),
            'reg': reg,
        })
    print('Nap xong trong %.1fs.' % (time.time() - t_load))

    train_items = [all_items[i] for i in splits['train']]
    val_items = [all_items[i] for i in splits['val']]
    test_items = [all_items[i] for i in splits['test']]

    # Dua san tap train len GPU (603 anh 256x128 chi ton ~230 MB VRAM / 4 GB -> train cuc nhanh!)
    train_x_all = torch.from_numpy(
        np.stack([it['img'].transpose(2, 0, 1).astype(np.float32) for it in train_items], axis=0)
    ).to(device)
    train_y_all = torch.from_numpy(
        np.stack([it['masks'] for it in train_items], axis=0)
    ).to(device)
    train_r_all = torch.from_numpy(
        np.stack([it['reg'] for it in train_items], axis=0)
    ).to(device)
    # Profiler do tren RTX 3050 Ti + torch 2.11+cu128: FP32 nhanh hon AMP FP16 4.5x voi Depthwise Conv!
    # (Khi deploy len Jetson Nano van build TensorRT FP16 binh thuong tu file .onnx.)
    results_path = os.path.join(out_dir, 'benchmark_results.json')
    existing_map = {}
    if args.resume and os.path.isfile(results_path):
        with open(results_path, 'r') as fh:
            for r in json.load(fh):
                existing_map[r['model']] = r

    results = []
    trained_models = {}

    for mname in args.models:
        if mname not in MODEL_REGISTRY:
            print('Bo qua model khong ton tai:', mname)
            continue
        pt_path = os.path.join(out_dir, mname + '.pt')
        onnx_path = os.path.join(out_dir, mname + '.onnx')
        if args.resume and mname in existing_map and os.path.isfile(pt_path):
            print('\n--- Resume [%s] tu checkpoint da luu ---' % mname)
            model = MODEL_REGISTRY[mname]().to(device)
            model.load_state_dict(torch.load(pt_path, map_location=device))
            model.eval()
            trained_models[mname] = model
            results.append(existing_map[mname])
            continue

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)

        model = MODEL_REGISTRY[mname]().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

        print('\n--- Train [%s] | params=%d ---' % (mname, n_params))
        t0 = time.time()
        best_val_score = -1.0
        best_state = None
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(len(train_items), device=device)
            losses = []
            for st in range(0, len(train_items), args.batch_size):
                bidx = perm[st:st + args.batch_size]
                if len(bidx) < 4:
                    continue
                bx, by, br = augment_batch_gpu(train_x_all[bidx], train_y_all[bidx], train_r_all[bidx])

                opt.zero_grad(set_to_none=True)
                seg_out, reg_out = model(bx)
                loss = multitask_loss(seg_out, reg_out, by, br)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            sched.step()

            if epoch % 5 == 0 or epoch == args.epochs:
                vm = evaluate_model(model, val_items, device)
                vscore = 0.25 * vm['road_iou'] + 0.35 * vm['white_iou'] + 0.40 * vm['center_iou']
                if vscore > best_val_score:
                    best_val_score = vscore
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print('  Ep %2d/%d (%.1fs) | loss=%.4f | Val IoU: road=%.3f white=%.3f center=%.3f | Val seg_CTE=%.4f' % (
                    epoch, args.epochs, time.time() - t0, np.mean(losses),
                    vm['road_iou'], vm['white_iou'], vm['center_iou'], vm['seg_cte_mae']
                ))

        train_sec = time.time() - t0
        model.load_state_dict(best_state)
        model.to(device).eval()

        val_metrics = evaluate_model(model, val_items, device)
        test_metrics = evaluate_model(model, test_items, device)

        torch.save(best_state, pt_path)
        est_fp32_kb = (n_params * 4.0) / 1024.0
        onnx_kb = None
        if args.export_onnx:
            onnx_kb = export_onnx(model, onnx_path)
            model.to(device).eval()
        mflops, gpu_ms, cpu_ms = measure_flops_and_speed(model, device)

        row = {
            'model': mname,
            'params': int(n_params),
            'est_onnx_kb': round(onnx_kb if onnx_kb is not None else est_fp32_kb, 1),
            'mflops': round(mflops, 1),
            'train_sec': round(train_sec, 1),
            'best_epoch': best_epoch,
            'gpu_ms': round(gpu_ms, 2),
            'cpu_ms': round(cpu_ms, 2),
            'val': val_metrics,
            'test': test_metrics,
            'pt_path': pt_path,
            'onnx_path': onnx_path if args.export_onnx else None,
        }
        results.append(row)
        trained_models[mname] = model
        with open(results_path, 'w') as fh:
            json.dump(results, fh, indent=2)
        print('  >>> TEST [%s]: road=%.3f | white=%.3f | center=%.3f | mIoU=%.3f | CTE_MAE=%.4f | LookMAE=%.4f | MFLOPs=%.1f | Size~%.0fKB | GPU=%.2fms | CPU=%.1fms' % (
            mname, test_metrics['road_iou'], test_metrics['white_iou'], test_metrics['center_iou'],
            test_metrics['miou'], test_metrics['seg_cte_mae'], test_metrics['seg_look_mae'],
            mflops, row['est_onnx_kb'], gpu_ms, cpu_ms
        ))

    with open(results_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    print('\n================ BANG TONG HOP TREN TAP TEST KHOA KIN (%d ANH) ================' % len(test_items))
    print('%-16s %7s %8s %7s %7s %7s %7s %8s %8s %7s %7s' % (
        'Model', 'Params', 'Size_KB', 'MFLOPs', 'Road', 'White', 'Center', 'CTE_MAE', 'LookMAE', 'GPU_ms', 'CPU_ms'
    ))
    for r in results:
        t = r['test']
        print('%-16s %7d %7.0fK %7.1f %7.3f %7.3f %7.3f %8.4f %8.4f %6.2fms %6.1fms' % (
            r['model'], r['params'], r['est_onnx_kb'], r['mflops'],
            t['road_iou'], t['white_iou'], t['center_iou'],
            t['seg_cte_mae'], t['seg_look_mae'], r['gpu_ms'], r['cpu_ms']
        ))

    # Xuat anh so sanh truc quan tren 4 frame Test da dang (Thang + Cua trai + Cua phai + Bong toi)
    if args.visual_out and trained_models:
        # Chon 4 frame Test: 2 tu Block 3 (cua trai/thang) + 2 tu Block 8 (cua phai/trai)
        pick_indices = [
            int(len(test_items) * 0.15),
            int(len(test_items) * 0.42),
            int(len(test_items) * 0.65),
            int(len(test_items) * 0.88),
        ]
        picks = [test_items[i] for i in pick_indices]
        model_names = list(trained_models.keys())

        def _overlay_mask(full_img, m_3ch, label_txt):
            ov = full_img.copy()
            r_up = cv2.resize((m_3ch[0] > 0.5).astype(np.uint8) * 255, (IMG_W, IMG_H - ROI_Y0), interpolation=cv2.INTER_NEAREST)
            w_up = cv2.resize((m_3ch[1] > 0.5).astype(np.uint8) * 255, (IMG_W, IMG_H - ROI_Y0), interpolation=cv2.INTER_NEAREST)
            c_up = cv2.resize((m_3ch[2] > 0.5).astype(np.uint8) * 255, (IMG_W, IMG_H - ROI_Y0), interpolation=cv2.INTER_NEAREST)
            roi_ov = ov[ROI_Y0:, :, :]
            roi_ov[r_up > 127] = (40, 200, 40)
            roi_ov[w_up > 127] = (240, 55, 235)
            roi_ov[c_up > 127] = (20, 215, 250)
            vis = cv2.addWeighted(ov, 0.45, full_img, 0.55, 0)
            cv2.rectangle(vis, (0, 0), (640, 34), (18, 18, 18), -1)
            cv2.putText(vis, label_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
            return cv2.resize(vis, (320, 240))

        rows = []
        for it in picks:
            cols = [_overlay_mask(it['full_img'], it['masks'], it['id'] + ' | GT')]
            x = torch.from_numpy(it['img'].transpose(2, 0, 1)[None].astype(np.float32)).to(device)
            for mn in model_names:
                with torch.no_grad():
                    sl, _ = trained_models[mn](x)
                    sp = (sl[0].cpu().numpy() > 0.0).astype(np.float32)
                cols.append(_overlay_mask(it['full_img'], sp, mn))
            rows.append(np.hstack(cols))
        grid = np.vstack(rows)
        cv2.imwrite(args.visual_out, grid)
        print('Da luu anh so sanh truc quan cac model tren tap Test tai:', args.visual_out)


if __name__ == '__main__':
    main()
