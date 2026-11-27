# -*- coding: utf-8 -*-
"""Active-learning pre-labeler cho Option B (3 kenh: road, white, centerline).

Huan luyen mang UNet nhe (CoordConv + Dice/BCE + augmentation manh) tren cac anh
da duoc nguoi dung duyet (`status == 'approved'`), sau do du doan lai (co TTA
lat ngang + hau xu ly sac gon) de cai thien pre-label cho toan bo anh `pending`.

AN TOAN:
  - CHI doc anh `approved` de train.
  - CHI ghi de anh `pending`. TUYET DOI KHONG dung vao anh `approved` hay `skipped`.
  - Co the chay lap lai nhieu vong (vi du sau 100 anh, 250 anh, 500 anh...).

Cach chay:
  python tools/active_prelabel_option_b.py --data data/lane_option_b --epochs 35
"""

from __future__ import print_function

import argparse
import json
import os
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from prelabel_option_b import (  # noqa: E402
    CENTERLINE_THICKNESS_PX,
    DEFAULT_HALF_W_BEV,
    IMG_H,
    IMG_W,
    ROI_TOP_FRAC,
    compute_centerline_from_masks,
)


# Crop tu y=144..480 (336x640 -> resize 128x256) de giu ngu canh ngay tren duong ROI y=163
CROP_Y0 = 144
CROP_H = IMG_H - CROP_Y0  # 336
NET_W, NET_H = 256, 128
ROI_Y = int(IMG_H * ROI_TOP_FRAC)  # 163


def _build_coord_grid(h, w):
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=0)  # (2, H, W)


COORD_GRID = _build_coord_grid(NET_H, NET_W)


class DWConvBlock(nn.Module):
    """Depthwise-separable block (MobileNet-style): nhanh hon 6x tren CPU & Jetson Nano."""

    def __init__(self, c_in, c_out, first=False):
        super(DWConvBlock, self).__init__()
        if first:
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, groups=c_out, bias=False),
                nn.Conv2d(c_out, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False),
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, groups=c_out, bias=False),
                nn.Conv2d(c_out, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x)


class OptionBUNet(nn.Module):
    """Compact 4-stage Depthwise-Separable UNet voi CoordConv (5 kenh vao -> 3 kenh ra)."""

    def __init__(self, base=24):
        super(OptionBUNet, self).__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 6
        self.enc1 = DWConvBlock(5, c1, first=True)
        self.enc2 = DWConvBlock(c1, c2)
        self.enc3 = DWConvBlock(c2, c3)
        self.enc4 = DWConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.dec3 = DWConvBlock(c4 + c3, c3)
        self.dec2 = DWConvBlock(c3 + c2, c2)
        self.dec1 = DWConvBlock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        u3 = F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.head(d1)


def _crop_and_resize_img(img_bgr):
    crop = img_bgr[CROP_Y0:, :, :]
    return cv2.resize(crop, (NET_W, NET_H), interpolation=cv2.INTER_AREA)


def _crop_and_resize_mask(mask):
    crop = mask[CROP_Y0:, :]
    resized = cv2.resize(crop, (NET_W, NET_H), interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.float32)


def _img_to_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    x = np.concatenate([rgb, COORD_GRID], axis=0)
    return torch.from_numpy(x)


def _augment_sample(img, masks):
    """Augmentation manh de tu 100 anh tong quat sang ca cua phai, bong toi, loa den."""
    h, w = img.shape[:2]

    # 1. Horizontal flip (bien cua trai thanh cua phai doi xung!)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        masks = np.flip(masks, axis=2).copy()

    # 2. Random affine (xoay nhe, dich ngang/doc, phong to/thu nho)
    if random.random() < 0.75:
        angle = random.uniform(-8.0, 8.0)
        scale = random.uniform(0.90, 1.10)
        tx = random.uniform(-0.08 * w, 0.08 * w)
        ty = random.uniform(-0.04 * h, 0.04 * h)
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.65), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for c in range(3):
            masks[c] = cv2.warpAffine(
                masks[c], M, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )

    # 3. Photometric jitter (do sang, tuong phan, mau sac)
    img_f = img.astype(np.float32)
    if random.random() < 0.85:
        alpha = random.uniform(0.65, 1.35)
        beta = random.uniform(-35.0, 35.0)
        img_f = np.clip(img_f * alpha + beta, 0, 255)
        # Channel gain jitter (giong thay doi can bang trang)
        gains = np.array([
            random.uniform(0.88, 1.12),
            random.uniform(0.90, 1.10),
            random.uniform(0.88, 1.12),
        ], dtype=np.float32)
        img_f = np.clip(img_f * gains[None, None, :], 0, 255)

    # 4. Random directional shadow gradient (giup nhan dien vach trang trong vung toi!)
    if random.random() < 0.55:
        gx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        gy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        ang = random.uniform(0, 2 * np.pi)
        grad = np.cos(ang) * gx + np.sin(ang) * gy
        grad = (grad - grad.min()) / max(1e-5, (grad.max() - grad.min()))
        dark = random.uniform(0.42, 0.78)
        factor = dark + (1.0 - dark) * grad
        img_f = np.clip(img_f * factor[:, :, None], 0, 255)

    # 5. Random synthetic white glare blob (giup mat duong den chiu loa den tran!)
    if random.random() < 0.50:
        cx = random.randint(int(w * 0.1), int(w * 0.9))
        cy = random.randint(int(h * 0.25), int(h * 0.95))
        rx = random.randint(15, 55)
        ry = random.randint(8, 28)
        glare = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(glare, (cx, cy), (rx, ry), random.uniform(-30, 30), 0, 360, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (21, 21), 0)
        strength = random.uniform(55.0, 145.0)
        img_f = np.clip(img_f + glare[:, :, None] * strength, 0, 255)

    return img_f.astype(np.uint8), masks


def _compute_sample_weights(frames_data):
    """Tang trong so cho cac frame cua/chuyen tiep va giam trong so frame trung lap."""
    weights = []
    # Tinh do khac biet so voi frame approved truoc do + do cong/che do cua
    prev_small = None
    for item in frames_data:
        rec = item['rec']
        img = item['img']
        sm = cv2.cvtColor(cv2.resize(img, (64, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = 10.0 if prev_small is None else float(np.mean(np.abs(sm - prev_small)))
        prev_small = sm

        w = 1.0
        # Frame cua gat / 1 vach rat hiem -> uu tien cao
        if rec.get('mode') == '1_line_parallel':
            w *= 4.5
        elif abs(float(rec.get('steer_deg', 0.0))) >= 8.0:
            w *= 3.0
        elif abs(float(rec.get('cte_lookahead', 0.0))) >= 0.20:
            w *= 2.0
        # Frame ngoai vung 0..64 (vung da dang) tang nhe
        if int(rec['id'].split('_')[-1]) >= 70:
            w *= 2.0
        # Giam trong so frame gan nhu dung yen so voi frame truoc
        if diff < 2.0:
            w *= 0.4
        weights.append(w)
    arr = np.array(weights, dtype=np.float64)
    return arr / np.sum(arr)


def _dice_bce_loss(logits, targets):
    # Trong so cao hon cho vach trang & line vang (vi dien tich nho hon road)
    ch_weights = torch.tensor([1.0, 2.0, 2.5], dtype=logits.dtype, device=logits.device).view(1, 3, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    bce_loss = (bce * ch_weights).mean()

    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3)) + 1e-5
    dice = 1.0 - (2.0 * inter + 1e-5) / denom
    cw = torch.tensor([1.0, 2.0, 2.5], dtype=logits.dtype, device=logits.device)
    dice_loss = (dice * cw).sum() / cw.sum()
    return 0.5 * bce_loss + 0.5 * dice_loss


def _eval_iou(model, val_items):
    model.eval()
    inter = np.zeros(3, dtype=np.float64)
    union = np.zeros(3, dtype=np.float64)
    with torch.no_grad():
        for item in val_items:
            x = _img_to_tensor(item['img']).unsqueeze(0)
            p = torch.sigmoid(model(x))[0].numpy()
            pred = (p > 0.5).astype(np.float32)
            gt = item['masks']
            for c in range(3):
                inter[c] += float(np.sum((pred[c] > 0.5) & (gt[c] > 0.5)))
                union[c] += float(np.sum((pred[c] > 0.5) | (gt[c] > 0.5)))
    ious = inter / np.maximum(1.0, union)
    return ious


def _predict_probs_tta(model, full_img_bgr):
    """Du doan 3 kenh xac suat co TTA (anh goc + lat ngang) va dua ve 640x480."""
    crop = _crop_and_resize_img(full_img_bgr)
    crop_flip = cv2.flip(crop, 1)
    batch = torch.stack([_img_to_tensor(crop), _img_to_tensor(crop_flip)], dim=0)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).numpy()  # (2, 3, NET_H, NET_W)
    p_orig = probs[0]
    p_flip = np.flip(probs[1], axis=2)
    p_avg = 0.5 * (p_orig + p_flip)  # (3, NET_H, NET_W)

    full_probs = np.zeros((3, IMG_H, IMG_W), dtype=np.float32)
    for c in range(3):
        up = cv2.resize(p_avg[c], (IMG_W, CROP_H), interpolation=cv2.INTER_LINEAR)
        full_probs[c, CROP_Y0:, :] = up
    return full_probs


def _clean_centerline_mask(prob_c, road_mask, white_mask):
    """Lam sach mask centerline du doan truc tiep tu UNet thanh 1 dai muot 14px."""
    raw = (prob_c > 0.42).astype(np.uint8) * 255
    raw[:ROI_Y, :] = 0
    # Xoa dom nho, giu dai lien thong lon nhat
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if num <= 1:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    best_i, best_area = 0, 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_i = i
    if best_area < 250:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    comp = (lbl == best_i).astype(np.uint8) * 255
    # Trich truc giua cua dai du doan de ve lai net deu dep 14px (khong lom dom)
    # Hoac giu nguyen comp neu no cua ngang (xet theo ca hang y lan cot x)
    ys, xs = np.where(comp > 0)
    span_y = int(ys.max() - ys.min())
    span_x = int(xs.max() - xs.min())

    # Sap xep diem bang step-centroid tu diem gan day nhat (max y) de xu ly duoc ca cua ngang!
    pts_all = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts_all) < 30:
        return comp

    # Bat dau tu vung day nhat cua dai
    y_max = float(ys.max())
    bot_pts = pts_all[pts_all[:, 1] >= y_max - 12.0]
    curr = np.mean(bot_pts, axis=0)
    path = [curr]
    visited = np.zeros(len(pts_all), dtype=bool)
    for _ in range(80):
        dists = np.linalg.norm(pts_all - curr, axis=1)
        visited |= (dists < 14.0)
        ring = (~visited) & (dists >= 14.0) & (dists <= 34.0)
        if not np.any(ring):
            break
        nxt = np.mean(pts_all[ring], axis=0)
        path.append(nxt)
        curr = nxt

    if len(path) < 3:
        return comp

    ordered = np.array(path, dtype=np.float32)
    if len(ordered) >= 5:
        k = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float32)
        sx = np.convolve(np.pad(ordered[:, 0], 2, mode='edge'), k, mode='valid')
        sy = np.convolve(np.pad(ordered[:, 1], 2, mode='edge'), k, mode='valid')
        ordered = np.stack([sx, sy], axis=1)

    clean = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    curve = ordered.astype(np.int32)
    cv2.polylines(clean, [curve], False, 255, thickness=CENTERLINE_THICKNESS_PX, lineType=cv2.LINE_AA)
    clean[:ROI_Y, :] = 0
    return clean


def postprocess_predictions(probs, half_w_bev=DEFAULT_HALF_W_BEV):
    """Chuyen 3 ban do xac suat UNet thanh 3 mask sach (road, white, centerline) + metrics."""
    # 1. White line: nguong 0.42, cat tren ROI, bo dom nho
    white_raw = (probs[1] > 0.42).astype(np.uint8) * 255
    white_raw[:ROI_Y - 4, :] = 0
    white_raw = cv2.morphologyEx(
        white_raw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    num_w, lbl_w, stats_w, _ = cv2.connectedComponentsWithStats(white_raw, connectivity=8)
    white_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for i in range(1, num_w):
        if stats_w[i, cv2.CC_STAT_AREA] >= 180:
            white_mask[lbl_w == i] = 255

    # 2. Drivable road: nguong 0.50, cat tren ROI, lay thanh phan lon nhat cham day, lap lo, tru white_mask
    road_raw = (probs[0] > 0.50).astype(np.uint8) * 255
    road_raw[:ROI_Y, :] = 0
    road_raw = cv2.morphologyEx(
        road_raw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    num_r, lbl_r, stats_r, _ = cv2.connectedComponentsWithStats(road_raw, connectivity=8)
    best_r, best_score = 0, 0
    bot = lbl_r[int(IMG_H * 0.78):, :]
    for i in range(1, num_r):
        touch = int(np.sum(bot == i))
        area = int(stats_r[i, cv2.CC_STAT_AREA])
        score = touch * 3 + area
        if touch > 50 and score > best_score:
            best_score = score
            best_r = i
    road_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    if best_r > 0:
        comp_r = (lbl_r == best_r).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(road_mask, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    # Loai bo vach trang xa khong giap mat duong
    if np.count_nonzero(road_mask) > 500:
        road_near = cv2.dilate(road_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
        num_w2, lbl_w2, _, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
        ego_white = np.zeros_like(white_mask)
        for i in range(1, num_w2):
            comp = (lbl_w2 == i)
            if np.sum(comp & (road_near > 0)) > 60:
                ego_white[comp] = 255
        white_mask = ego_white

    road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(white_mask))
    road_mask[:ROI_Y, :] = 0

    # 3. Centerline: dung truc tiep UNet (da hoc dung phong cach ve cua nguoi dung!),
    # fallback sang BEV Distance Transform (tu road+white sieu sach cua UNet) neu
    # line vang UNet qua ngan (span < 140 px hoac dien tich < 2000 px).
    bev_center, bev_info = compute_centerline_from_masks(
        road_mask, white_mask, half_w_bev=half_w_bev
    )
    unet_center = _clean_centerline_mask(probs[2], road_mask, white_mask)
    ys_u, xs_u = np.where(unet_center > 0)
    unet_span = (
        max(int(ys_u.max() - ys_u.min()), int(xs_u.max() - xs_u.min()))
        if len(ys_u) > 0 else 0
    )

    if len(ys_u) >= 2000 and unet_span >= 140:
        center_mask = unet_center
        # Tinh lookahead_img truc tiep tu unet_center de Preview Pure Pursuit khop voi line vang
        y_target = int(np.percentile(ys_u, 35))
        near_row = xs_u[np.abs(ys_u - y_target) <= 8]
        if len(near_row):
            lx = float(np.median(near_row))
            bev_info['lookahead_img'] = [round(lx, 1), round(float(y_target), 1)]
    else:
        center_mask = bev_center

    return road_mask, white_mask, center_mask, bev_info


def main():
    parser = argparse.ArgumentParser(description='Active-learning pre-label Option B')
    parser.add_argument('--data', default='data/lane_option_b')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--steps-per-epoch', type=int, default=22)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weights', default=None, help='Nap san weights .pt da train (bo qua buoc train)')
    parser.add_argument('--dry-run', action='store_true', help='Chi train & xuat anh so sanh, chua ghi de pending')
    parser.add_argument('--preview-out', default=None, help='Luu anh so sanh Truoc vs Sau tren 6 frame pending')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = os.path.abspath(args.data)
    meta_path = os.path.join(data_dir, 'metadata.json')
    with open(meta_path, 'r') as fh:
        meta = json.load(fh)

    approved_recs = [f for f in meta['frames'] if f['status'] == 'approved']
    pending_recs = [f for f in meta['frames'] if f['status'] == 'pending']
    print('Approved (dung de train): %d | Pending (se cai thien): %d' % (
        len(approved_recs), len(pending_recs)))

    model = OptionBUNet(base=24)
    if args.weights and os.path.isfile(args.weights):
        print('Nap san model tu:', args.weights)
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
        model.eval()
    else:
        if len(approved_recs) < 10:
            raise RuntimeError('Can it nhat 10 anh approved de train!')

        # Load toan bo anh approved vao RAM (100 anh 256x128 chi ton ~10 MB -> train nhanh!)
        items = []
        for rec in approved_recs:
            fid = rec['id']
            img = cv2.imread(os.path.join(data_dir, 'images', rec['image']))
            r = cv2.imread(os.path.join(data_dir, 'masks', 'road', fid + '.png'), 0)
            w = cv2.imread(os.path.join(data_dir, 'masks', 'white', fid + '.png'), 0)
            c = cv2.imread(os.path.join(data_dir, 'masks', 'centerline', fid + '.png'), 0)
            if img is None or r is None or w is None or c is None:
                continue
            masks = np.stack([
                _crop_and_resize_mask(r),
                _crop_and_resize_mask(w),
                _crop_and_resize_mask(c),
            ], axis=0)
            items.append({
                'rec': rec,
                'img': _crop_and_resize_img(img),
                'masks': masks,
            })

        # Chia train / val phan tang (giu 12 anh dai dien de theo doi IoU thuc)
        val_indices = set(np.linspace(0, len(items) - 1, min(12, max(4, len(items) // 8))).astype(int).tolist())
        train_items = [it for i, it in enumerate(items) if i not in val_indices]
        val_items = [it for i, it in enumerate(items) if i in val_indices]
        train_probs = _compute_sample_weights(train_items)

        print('Train: %d anh | Val: %d anh | Thiet bi: CPU (%d threads)' % (
            len(train_items), len(val_items), torch.get_num_threads()))

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

        best_score = -1.0
        best_state = None
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for _ in range(args.steps_per_epoch):
                idxs = np.random.choice(len(train_items), size=args.batch_size, replace=True, p=train_probs)
                xs, ys = [], []
                for idx in idxs:
                    it = train_items[int(idx)]
                    aug_img, aug_m = _augment_sample(it['img'].copy(), it['masks'].copy())
                    xs.append(_img_to_tensor(aug_img))
                    ys.append(torch.from_numpy(aug_m))
                bx = torch.stack(xs, dim=0)
                by = torch.stack(ys, dim=0)

                opt.zero_grad()
                logits = model(bx)
                loss = _dice_bce_loss(logits, by)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            sched.step()

            if epoch % 5 == 0 or epoch == args.epochs:
                ious = _eval_iou(model, val_items)
                score = float(0.30 * ious[0] + 0.35 * ious[1] + 0.35 * ious[2])
                if score > best_score:
                    best_score = score
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print('Epoch %2d/%d (%.1fs) | loss=%.4f | Val IoU -> road=%.3f  white=%.3f  center=%.3f' % (
                    epoch, args.epochs, time.time() - t0, np.mean(losses), ious[0], ious[1], ious[2]))

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        ckpt_path = os.path.join(data_dir, 'active_prelabel_unet.pt')
        torch.save(model.state_dict(), ckpt_path)
        print('Da luu model active-learning tai:', ckpt_path)

    # Tao anh so sanh Truoc (CV cu) vs Sau (UNet moi) tren 6 frame pending da dang (bao gom cac ca kho!)
    if args.preview_out and pending_recs:
        hard_ids = ['frame_000393', 'frame_000608', 'frame_000951', 'frame_000772', 'frame_000327', 'frame_000510']
        pending_map = dict((r['id'], r) for r in pending_recs)
        preview_recs = [pending_map[fid] for fid in hard_ids if fid in pending_map]
        if len(preview_recs) < 6:
            extra = [pending_recs[i] for i in np.linspace(0, len(pending_recs) - 1, 6).astype(int)]
            for r in extra:
                if r not in preview_recs and len(preview_recs) < 6:
                    preview_recs.append(r)

        rows = []
        for rec in preview_recs[:6]:
            fid = rec['id']
            img = cv2.imread(os.path.join(data_dir, 'images', rec['image']))
            old_r = cv2.imread(os.path.join(data_dir, 'masks', 'road', fid + '.png'), 0)
            old_w = cv2.imread(os.path.join(data_dir, 'masks', 'white', fid + '.png'), 0)
            old_c = cv2.imread(os.path.join(data_dir, 'masks', 'centerline', fid + '.png'), 0)

            probs = _predict_probs_tta(model, img)
            new_r, new_w, new_c, _ = postprocess_predictions(probs)

            def _render(rm, wm, cm, tag):
                ov = img.copy()
                ov[rm > 127] = (40, 200, 40)
                ov[wm > 127] = (240, 55, 235)
                ov[cm > 127] = (20, 215, 250)
                v = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
                cv2.rectangle(v, (0, 0), (640, 32), (20, 20, 20), -1)
                cv2.putText(v, '%s | %s' % (fid, tag), (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                return cv2.resize(v, (480, 360))

            pair = np.hstack([_render(old_r, old_w, old_c, 'TRUOC (CV cu)'),
                              _render(new_r, new_w, new_c, 'SAU (UNet hoc tu 100 anh)')])
            rows.append(pair)
        # 3 hang x 2 cap (tong 6 frame so sanh Truoc/Sau -> ghep 2 cot lon)
        left_col = np.vstack(rows[:3])
        right_col = np.vstack(rows[3:]) if len(rows) > 3 else left_col
        comp = np.hstack([left_col, right_col]) if len(rows) > 3 else left_col
        cv2.imwrite(args.preview_out, comp)
        print('Da luu anh so sanh Truoc/Sau tai:', args.preview_out)

    if args.dry_run:
        print('Che do --dry-run: KHONG ghi de anh pending.')
        return

    # Backup metadata truoc khi cap nhat pending
    backup_meta = os.path.join(data_dir, 'metadata.before_active.json')
    if not os.path.isfile(backup_meta):
        shutil.copy2(meta_path, backup_meta)

    print('Bat dau cap nhat pre-label cho %d anh pending...' % len(pending_recs))
    t_inf = time.time()
    for idx, rec in enumerate(pending_recs):
        fid = rec['id']
        img = cv2.imread(os.path.join(data_dir, 'images', rec['image']))
        if img is None:
            continue
        probs = _predict_probs_tta(model, img)
        road_m, white_m, center_m, info = postprocess_predictions(
            probs, half_w_bev=float(rec.get('half_w_bev', DEFAULT_HALF_W_BEV))
        )
        cv2.imwrite(os.path.join(data_dir, 'masks', 'road', fid + '.png'), road_m)
        cv2.imwrite(os.path.join(data_dir, 'masks', 'white', fid + '.png'), white_m)
        cv2.imwrite(os.path.join(data_dir, 'masks', 'centerline', fid + '.png'), center_m)

        rec['mode'] = info['mode']
        rec['cte'] = info['cte']
        rec['cte_lookahead'] = info['cte_lookahead']
        rec['curvature'] = info['curvature']
        rec['steer_deg'] = info['steer_deg']
        rec['lookahead_img'] = info['lookahead_img']
        rec['prelabel_source'] = 'active_unet_100'

        if (idx + 1) % 100 == 0 or (idx + 1) == len(pending_recs):
            print('  Da cap nhat %d / %d anh pending (%.1fs)' % (
                idx + 1, len(pending_recs), time.time() - t_inf))

    tmp_meta = meta_path + '.tmp'
    with open(tmp_meta, 'w') as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp_meta, meta_path)
    print('HOAN TAT! Da nang cap pre-label cho %d anh pending (giu nguyen %d anh approved).' % (
        len(pending_recs), len(approved_recs)))


if __name__ == '__main__':
    main()
