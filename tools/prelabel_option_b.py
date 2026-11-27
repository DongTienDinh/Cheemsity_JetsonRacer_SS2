# -*- coding: utf-8 -*-
"""Pre-label Option B (3 kenh: drivable_road, white_line, centerline) tu dataset_raw.zip.

Tao bo du lieu tai: data/lane_option_b/
  images/      : ~200 frame da dang loc tu dataset_raw.zip
  masks/
    road/      : mask nhi phan 0/255 (drivable_road - mat duong den)
    white/     : mask nhi phan 0/255 (white_line - vach trang sat duong)
    centerline/: mask nhi phan 0/255 (centerline - duong tam thuc te, KHONG ep vao (w/2, h))
  metadata.json: cau hinh BEV, ROI, trang thai review & reg [cte, cte_lookahead, curvature]

Chay:
  python tools/prelabel_option_b.py --zip ../dataset_raw.zip --out data/lane_option_b --count 200
"""

from __future__ import print_function

import argparse
import json
import os
import zipfile

import cv2
import numpy as np


# --- Cau hinh hinh hoc BEV & ROI (dong bo voi label_app.py va luc train) ---
IMG_W, IMG_H = 640, 480
ROI_TOP_FRAC = 0.34
ROI_Y = int(IMG_H * ROI_TOP_FRAC)

BEV_W, BEV_H = 500, 360
BEV_SRC = np.float32([
    [0.0,   440.0],
    [640.0, 440.0],
    [420.0, 200.0],
    [210.0, 200.0],
])
BEV_DST = np.float32([
    [205.0, 355.0],
    [295.0, 355.0],
    [295.0, 110.0],
    [205.0, 110.0],
])
H_MAT = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)
H_INV = cv2.getPerspectiveTransform(BEV_DST, BEV_SRC)
DEFAULT_HALF_W_BEV = 48.0
CENTERLINE_THICKNESS_PX = 14


def segment_road_and_white(img):
    """Pre-label CV nhap cho 2 lop: drivable_road (mat den) + white_line (vach trang sat duong)."""
    h, w = img.shape[:2]
    roi_y = int(h * ROI_TOP_FRAC)

    b, g, r = cv2.split(img)
    min_c = np.minimum(np.minimum(b, g), r)
    max_c = np.maximum(np.maximum(b, g), r)

    # Vach trang: sang va do bao hoa thap
    white_raw = ((min_c > 145) & ((max_c - min_c) < 55)).astype(np.uint8) * 255

    # Nen do xung quanh: R cao hon han G va B
    red_mat = (
        (r.astype(np.int16) - g.astype(np.int16) > 25) &
        (r.astype(np.int16) - b.astype(np.int16) > 25) &
        (r > 70)
    ).astype(np.uint8) * 255
    red_mat = cv2.morphologyEx(
        red_mat, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )

    white_raw[:roi_y, :] = 0
    white_raw = cv2.morphologyEx(
        white_raw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    # Chi giu vach trang giap nen do (loai dom loa den giua mat duong den)
    red_near = cv2.dilate(
        red_mat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
    num_w, lbl_w, stats_w, _ = cv2.connectedComponentsWithStats(white_raw, connectivity=8)
    white_line = np.zeros_like(white_raw)
    for i in range(1, num_w):
        if stats_w[i, cv2.CC_STAT_AREA] < 200:
            continue
        comp = (lbl_w == i)
        if np.sum(comp & (red_near > 0)) > 50:
            white_line[comp] = 255

    # Hang rao chan = white_line OR red_mat
    barrier = cv2.bitwise_or(white_line, red_mat)
    barrier = cv2.dilate(
        barrier, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    barrier[:roi_y, :] = 255
    cand = cv2.bitwise_not(barrier)

    # Lay thanh phan lien thong cham vung day giua (ego-lane)
    num_r, lbl_r, _, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    best_lbl, best_touch = 0, 0
    bot_slice = lbl_r[int(h * 0.85):, int(w * 0.20):int(w * 0.80)]
    for i in range(1, num_r):
        t = int(np.sum(bot_slice == i))
        if t > best_touch:
            best_touch = t
            best_lbl = i

    road_mask = (lbl_r == best_lbl).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_road = np.zeros_like(road_mask)
    if cnts:
        cv2.drawContours(
            clean_road, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED
        )
    # Cat sach khong cho tran sang vach trang hay nen do
    clean_road = cv2.bitwise_and(clean_road, cv2.bitwise_not(white_line))
    clean_road = cv2.bitwise_and(clean_road, cv2.bitwise_not(red_mat))

    # Chi giu vach trang giap mat duong den (loai vach trang vien ngoai cung cua bat do)
    road_near = cv2.dilate(
        clean_road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    )
    num_w2, lbl_w2, _, _ = cv2.connectedComponentsWithStats(white_line, connectivity=8)
    ego_white = np.zeros_like(white_line)
    for i in range(1, num_w2):
        comp = (lbl_w2 == i)
        if np.sum(comp & (road_near > 0)) > 80:
            ego_white[comp] = 255

    return clean_road, ego_white


def compute_centerline_from_masks(road_mask, white_mask, half_w_bev=DEFAULT_HALF_W_BEV):
    """Sinh mask centerline (line vang thuc te cua mat duong) + reg [cte, lookahead, curvature].

    Luu y: Ground truth centerline KHONG ep noi vao (w/2, h) de giu dung CTE luc train.
    """
    h, w = road_mask.shape[:2]
    roi_y = int(h * ROI_TOP_FRAC)
    center_mask = np.zeros((h, w), dtype=np.uint8)

    road_bev = cv2.warpPerspective(
        road_mask, H_MAT, (BEV_W, BEV_H), flags=cv2.INTER_NEAREST
    )
    white_bev = cv2.warpPerspective(
        white_mask, H_MAT, (BEV_W, BEV_H), flags=cv2.INTER_NEAREST
    )

    info = {
        'mode': 'none',
        'cte': 0.0,
        'cte_lookahead': 0.0,
        'curvature': 0.0,
        'steer_deg': 0.0,
        'lookahead_img': None,
        'points_img': [],
    }

    if np.count_nonzero(road_bev) < 100:
        return center_mask, info

    has_white = np.count_nonzero(white_bev) >= 20
    car_bev = np.array([BEV_W / 2.0, BEV_H - 2.0], dtype=np.float32)
    ordered = None

    if has_white:
        dist_w = cv2.distanceTransform(cv2.bitwise_not(white_bev), cv2.DIST_L2, 5)
        w_closed = cv2.morphologyEx(
            white_bev, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        num_wc, _, stats_wc, _ = cv2.connectedComponentsWithStats(w_closed, connectivity=8)
        major_lines = [i for i in range(1, num_wc) if stats_wc[i, cv2.CC_STAT_AREA] >= 120]

        if len(major_lines) >= 2:
            pts = []
            for y in range(BEV_H - 10, 50, -10):
                xs = np.where(road_bev[y, :] > 0)[0]
                if len(xs) < 20:
                    continue
                x_l, x_r = float(xs[0]), float(xs[-1])
                w_row = x_r - x_l
                if 55.0 <= w_row <= 160.0:
                    pts.append([0.5 * (x_l + x_r), float(y)])
                else:
                    best_x = float(xs[int(np.argmax(dist_w[y, xs]))])
                    pts.append([best_x, float(y)])
            if len(pts) >= 4:
                ordered = np.array(pts, dtype=np.float32)
                info['mode'] = '2_lines'
        else:
            band = (
                (road_bev > 0) &
                (np.abs(dist_w - float(half_w_bev)) <= 3.5)
            ).astype(np.uint8) * 255
            band = cv2.morphologyEx(
                band, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            )
            num_b, lbl_b, stats_b, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
            best_i, best_score = 0, -1e9
            for i in range(1, num_b):
                area = stats_b[i, cv2.CC_STAT_AREA]
                if area < 40:
                    continue
                ys_i, xs_i = np.where(lbl_b == i)
                pts_i = np.stack([xs_i, ys_i], axis=1).astype(np.float32)
                min_d = float(np.min(np.linalg.norm(pts_i - car_bev, axis=1)))
                score = area - 2.0 * min_d
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i > 0:
                ys_i, xs_i = np.where(lbl_b == best_i)
                pts_all = np.stack([xs_i, ys_i], axis=1).astype(np.float32)
                start_idx = int(np.argmax(pts_all[:, 1]))
                curr = pts_all[start_idx].copy()
                path = [curr]
                visited = np.zeros(len(pts_all), dtype=bool)
                for _ in range(90):
                    dists = np.linalg.norm(pts_all - curr, axis=1)
                    visited |= (dists < 10.0)
                    ring = (~visited) & (dists >= 10.0) & (dists <= 28.0)
                    if not np.any(ring):
                        break
                    nxt = np.mean(pts_all[ring], axis=0)
                    path.append(nxt)
                    curr = nxt
                if len(path) >= 3:
                    ordered = np.array(path, dtype=np.float32)
                    info['mode'] = '1_line_parallel'

    # Fallback: neu khong co vach trang nao, dung trung diem road_bev
    if ordered is None:
        pts = []
        for y in range(BEV_H - 10, 50, -10):
            xs = np.where(road_bev[y, :] > 0)[0]
            if len(xs) >= 20:
                pts.append([0.5 * (float(xs[0]) + float(xs[-1])), float(y)])
        if len(pts) >= 4:
            ordered = np.array(pts, dtype=np.float32)
            info['mode'] = 'road_only'

    if ordered is None or len(ordered) < 3:
        return center_mask, info

    # Lam muot duong tam tren BEV
    if len(ordered) >= 5:
        k = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float32)
        sx = np.convolve(np.pad(ordered[:, 0], 2, mode='edge'), k, mode='valid')
        sy = np.convolve(np.pad(ordered[:, 1], 2, mode='edge'), k, mode='valid')
        ordered = np.stack([sx, sy], axis=1)

    # Tinh CTE (gan xe), CTE lookahead (y_bev ~ 180), curvature va goc lai Pure Pursuit
    near_idx = int(np.argmax(ordered[:, 1]))
    look_idx = int(np.argmin(np.abs(ordered[:, 1] - 180.0)))
    near_bev = ordered[near_idx]
    look_bev = ordered[look_idx]

    cte = float(np.clip((near_bev[0] - car_bev[0]) / float(half_w_bev), -1.0, 1.0))
    cte_look = float(np.clip((look_bev[0] - car_bev[0]) / float(half_w_bev), -1.0, 1.0))

    # Do cong uoc luong tu sai khac huong doan gan vs doan xa
    mid_idx = len(ordered) // 2
    v1 = ordered[mid_idx] - ordered[0]
    v2 = ordered[-1] - ordered[mid_idx]
    a1 = np.arctan2(v1[0], max(1.0, -v1[1]))
    a2 = np.arctan2(v2[0], max(1.0, -v2[1]))
    curvature = float(np.clip((a2 - a1) / (np.pi / 2.0), -1.0, 1.0))

    dx = float(look_bev[0] - car_bev[0])
    dy = float(max(10.0, car_bev[1] - look_bev[1]))
    steer_deg = float(np.degrees(np.arctan2(dx, dy)))

    # Chieu nguoc duong tam BEV ve anh goc de ve mask ground truth centerline
    pts_img = cv2.perspectiveTransform(
        ordered.reshape(-1, 1, 2), H_INV
    ).reshape(-1, 2)
    look_img = cv2.perspectiveTransform(
        look_bev.reshape(1, 1, 2), H_INV
    ).reshape(2)

    valid = pts_img[
        (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) &
        (pts_img[:, 1] >= roi_y) & (pts_img[:, 1] < h)
    ]
    if len(valid) >= 2:
        curve = valid.astype(np.int32)
        cv2.polylines(
            center_mask, [curve], False, 255,
            thickness=CENTERLINE_THICKNESS_PX, lineType=cv2.LINE_AA,
        )
        # Giu centerline nam trong vung ROI
        center_mask[:roi_y, :] = 0

    info['cte'] = round(cte, 4)
    info['cte_lookahead'] = round(cte_look, 4)
    info['curvature'] = round(curvature, 4)
    info['steer_deg'] = round(steer_deg, 2)
    info['lookahead_img'] = [round(float(look_img[0]), 1), round(float(look_img[1]), 1)]
    info['points_img'] = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in valid]
    return center_mask, info


def select_diverse_frames(zfile, img_names, target_count):
    """Loc bo frame dung yen / trung lap, chon deu ~target_count frame da dang."""
    sorted_names = sorted(img_names)
    non_static = []
    prev_small = None

    for name in sorted_names:
        data = zfile.read(name)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        small = cv2.cvtColor(cv2.resize(img, (80, 60)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_small is None:
            non_static.append(name)
            prev_small = small
        else:
            diff = float(np.mean(np.abs(small - prev_small)))
            # Bo qua frame gan nhu dung yen (diff < 2.5)
            if diff >= 2.5:
                non_static.append(name)
                prev_small = small

    if len(non_static) <= target_count:
        return non_static

    indices = np.linspace(0, len(non_static) - 1, target_count).astype(int)
    return [non_static[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description='Pre-label Option B dataset (3 kenh)')
    parser.add_argument('--zip', required=True, help='Duong dan toi dataset_raw.zip')
    parser.add_argument('--out', default='data/lane_option_b', help='Thu muc xuat dataset')
    parser.add_argument('--count', type=int, default=200, help='So frame da dang can trich loc')
    parser.add_argument('--all', action='store_true', help='Pre-label TOAN BO anh trong zip (khong loc bo)')
    parser.add_argument('--half-w-bev', type=float, default=DEFAULT_HALF_W_BEV)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    img_dir = os.path.join(out_dir, 'images')
    road_dir = os.path.join(out_dir, 'masks', 'road')
    white_dir = os.path.join(out_dir, 'masks', 'white')
    center_dir = os.path.join(out_dir, 'masks', 'centerline')
    for d in (img_dir, road_dir, white_dir, center_dir):
        if not os.path.isdir(d):
            os.makedirs(d)

    with zipfile.ZipFile(args.zip, 'r') as zf:
        img_names = [
            n for n in zf.namelist()
            if n.lower().endswith(('.jpg', '.jpeg', '.png'))
            and '.ipynb_checkpoints' not in n
        ]
        print('Tong so anh trong zip:', len(img_names))
        if args.all:
            selected = sorted(img_names)
            print('Che do --all: pre-label toan bo %d anh.' % len(selected))
        else:
            selected = select_diverse_frames(zf, img_names, args.count)
            print('Da loc %d frame da dang (bo frame dung yen/trung).' % len(selected))

        frames_meta = []
        mode_counts = {}
        for idx, zname in enumerate(selected):
            fname = os.path.basename(zname)
            stem = os.path.splitext(fname)[0]
            raw_bytes = zf.read(zname)
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            cv2.imwrite(os.path.join(img_dir, fname), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            road_mask, white_mask = segment_road_and_white(img)
            center_mask, info = compute_centerline_from_masks(
                road_mask, white_mask, half_w_bev=args.half_w_bev
            )

            cv2.imwrite(os.path.join(road_dir, stem + '.png'), road_mask)
            cv2.imwrite(os.path.join(white_dir, stem + '.png'), white_mask)
            cv2.imwrite(os.path.join(center_dir, stem + '.png'), center_mask)

            mode_counts[info['mode']] = mode_counts.get(info['mode'], 0) + 1
            frames_meta.append({
                'id': stem,
                'image': fname,
                'status': 'pending',
                'mode': info['mode'],
                'cte': info['cte'],
                'cte_lookahead': info['cte_lookahead'],
                'curvature': info['curvature'],
                'steer_deg': info['steer_deg'],
                'lookahead_img': info['lookahead_img'],
                'half_w_bev': float(args.half_w_bev),
            })
            if (idx + 1) % 50 == 0 or (idx + 1) == len(selected):
                print('  Da pre-label %d / %d' % (idx + 1, len(selected)))

    meta = {
        'version': 'option_b_v1',
        'classes': ['drivable_road', 'white_line', 'centerline'],
        'img_width': IMG_W,
        'img_height': IMG_H,
        'roi_top_frac': ROI_TOP_FRAC,
        'bev_width': BEV_W,
        'bev_height': BEV_H,
        'bev_src': BEV_SRC.tolist(),
        'bev_dst': BEV_DST.tolist(),
        'default_half_w_bev': float(args.half_w_bev),
        'centerline_thickness_px': CENTERLINE_THICKNESS_PX,
        'frames': frames_meta,
    }
    meta_path = os.path.join(out_dir, 'metadata.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)

    print('Hoan tat pre-label tai:', out_dir)
    print('Thong ke mode centerline:', mode_counts)


if __name__ == '__main__':
    main()
