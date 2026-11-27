# -*- coding: utf-8 -*-
r"""Sinh lai mask centerline cho cac frame '1_line_parallel' (khuc cua) cua Option B.

VAN DE (do tren 1039 frame da duyet, BEV 500x360 cua metadata):
  - Frame '2_lines'        : khoang cach centerline -> mep road (vach trang) ~ 55-57 px
                             = dung nua be rong lan (road width ~116 px).
  - Frame '1_line_parallel': khoang cach do lai ~ 70-77 px -> centerline bi day
                             LECH VAO PHIA TRONG CUA ~15-20 px (~30% nua lan) va co
                             doan "moc" chay doc mep tam nhin camera.
  => Model hoc duong tam CAT CUA -> xe an vao vach trang phia trong o cac cua gat.

CACH SUA: chi sinh lai centerline cho frame '1_line_parallel' bang hinh hoc:
  centerline = cac pixel road co khoang cach toi "tuong" (vung KHONG phai road
  trong tam nhin) = HALF, voi HALF do truc tiep tu cac frame '2_lines' (cung
  thuoc do distance transform). Bien tam nhin camera KHONG bi tinh la tuong.
  Mask road/white giu nguyen (da dung, IoU road 0.99).

Mask cu duoc sao luu sang masks/centerline_before_fix/ (chi lan chay dau).

Chay:
  .\.venv_cuda\Scripts\python.exe tools/relabel_centerline_option_b.py --data data/lane_option_b
  (them --dry-run de chi in thong ke, --preview reports/relabel_preview.jpg de xem anh)
"""

from __future__ import print_function

import argparse
import json
import os
import shutil

import cv2
import numpy as np

THICKNESS = 14


def _load_geom(meta):
    src = np.float32(meta['bev_src'])
    dst = np.float32(meta['bev_dst'])
    H = cv2.getPerspectiveTransform(src, dst)
    Hi = cv2.getPerspectiveTransform(dst, src)
    bw, bh = int(meta['bev_width']), int(meta['bev_height'])
    img_w, img_h = int(meta['img_width']), int(meta['img_height'])
    roi_y = int(img_h * float(meta['roi_top_frac']))
    vis = cv2.warpPerspective(np.full((img_h, img_w), 255, np.uint8), H, (bw, bh),
                              flags=cv2.INTER_NEAREST)
    # Bo vien tam nhin: bien camera cat KHONG phai mep duong.
    vis = cv2.erode(vis, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return H, Hi, bw, bh, roi_y, vis


def _bev(mask, H, bw, bh):
    return cv2.warpPerspective(mask, H, (bw, bh), flags=cv2.INTER_NEAREST) > 127


def _wall_dist(road_b, vis):
    wall = vis & ~road_b
    return cv2.distanceTransform((~wall).astype(np.uint8), cv2.DIST_L2, 5), wall


def _center_dist_stats(center_b, dist):
    ys, xs = np.nonzero(center_b)
    if len(ys) < 20:
        return None
    return float(np.median(dist[ys, xs]))


def measure_half(meta, data_dir, H, bw, bh, vis):
    vals = []
    for f in meta['frames']:
        if f.get('status') != 'approved' or f.get('mode') != '2_lines':
            continue
        r = cv2.imread(os.path.join(data_dir, 'masks', 'road', f['id'] + '.png'), 0)
        c = cv2.imread(os.path.join(data_dir, 'masks', 'centerline', f['id'] + '.png'), 0)
        rb = _bev(r, H, bw, bh)
        dist, wall = _wall_dist(rb, vis)
        if wall.sum() < 200:
            continue
        d = _center_dist_stats(_bev(c, H, bw, bh), dist)
        if d is not None:
            vals.append(d)
    return float(np.median(vals)), len(vals)


def _ego_component(road_b, bw, bh):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(road_b.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    near = lbl[bh - 25:bh, bw // 2 - 30:bw // 2 + 30].ravel()
    near = near[near > 0]
    if len(near) == 0:
        return None
    return lbl == np.bincount(near).argmax()


def trace_centerline(road_b, vis, half, bw, bh, tol=3.0):
    """Tra ve polyline (N,2) trong BEV cua duong tam cach tuong `half` px."""
    ego = _ego_component(road_b, bw, bh)
    if ego is None:
        return None
    dist, wall = _wall_dist(ego, vis)
    if wall.sum() < 120:
        return None
    band = (ego & vis & (np.abs(dist - half) <= tol)).astype(np.uint8) * 255
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    car = np.array([bw / 2.0, bh - 2.0], np.float32)
    best, best_score = 0, -1e9
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 60:
            continue
        ys, xs = np.nonzero(lbl == i)
        d = float(np.min(np.hypot(xs - car[0], ys - car[1])))
        score = area - 2.0 * d
        if score > best_score:
            best, best_score = i, score
    if best == 0:
        return None
    ys, xs = np.nonzero(lbl == best)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    # Bat dau tu diem gan xe nhat, di theo vanh khuyen va SNAP ve pixel band
    # (tranh lech day cung khi lay trung binh tren cung cong).
    cur = pts[int(np.argmin(np.hypot(pts[:, 0] - car[0], pts[:, 1] - car[1])))]
    path = [cur.copy()]
    visited = np.zeros(len(pts), bool)
    for _ in range(120):
        d = np.hypot(pts[:, 0] - cur[0], pts[:, 1] - cur[1])
        visited |= d < 8.0
        ring = (~visited) & (d >= 8.0) & (d <= 16.0)
        if not ring.any():
            break
        m = pts[ring].mean(0)
        cand = np.flatnonzero(ring)
        cur = pts[cand[int(np.argmin(np.hypot(pts[cand, 0] - m[0], pts[cand, 1] - m[1])))]]
        path.append(cur.copy())
    if len(path) < 4:
        return None
    path = np.array(path, np.float32)
    k = np.array([0.25, 0.5, 0.25], np.float32)
    sm = np.stack([np.convolve(np.pad(path[:, j], 1, mode='edge'), k, mode='valid') for j in range(2)], 1)
    return sm


def render_mask(path_bev, Hi, img_w, img_h, roi_y):
    out = np.zeros((img_h, img_w), np.uint8)
    pts = cv2.perspectiveTransform(path_bev.reshape(-1, 1, 2), Hi).reshape(-1, 2)
    ok = (pts[:, 0] > -50) & (pts[:, 0] < img_w + 50) & (pts[:, 1] >= roi_y - 5) & (pts[:, 1] < img_h + 50)
    pts = pts[ok]
    if len(pts) < 2:
        return None
    cv2.polylines(out, [np.rint(pts).astype(np.int32)], False, 255, THICKNESS, cv2.LINE_AA)
    out[:roi_y] = 0
    return (out > 127).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default='data/lane_option_b')
    ap.add_argument('--half', type=float, default=None, help='Ghi de HALF (BEV px); mac dinh do tu frame 2_lines')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--preview', default=None)
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data)
    meta_path = os.path.join(data_dir, 'metadata.json')
    with open(meta_path, 'r') as fh:
        meta = json.load(fh)
    H, Hi, bw, bh, roi_y, vis = _load_geom(meta)
    img_w, img_h = int(meta['img_width']), int(meta['img_height'])

    half, n_ref = measure_half(meta, data_dir, H, bw, bh, vis)
    if args.half is not None:
        half = float(args.half)
    print('HALF (centerline -> mep road, BEV px) do tu %d frame 2_lines = %.1f' % (n_ref, half))

    cl_dir = os.path.join(data_dir, 'masks', 'centerline')
    bak_dir = os.path.join(data_dir, 'masks', 'centerline_before_fix')
    if not args.dry_run and not os.path.isdir(bak_dir):
        shutil.copytree(cl_dir, bak_dir)
        print('Da sao luu mask cu ->', bak_dir)

    before, after, fixed, failed = [], [], 0, []
    previews = []
    for i, f in enumerate(meta['frames']):
        if f.get('mode') != '1_line_parallel':
            continue
        fid = f['id']
        r = cv2.imread(os.path.join(data_dir, 'masks', 'road', fid + '.png'), 0)
        c_old = cv2.imread(os.path.join(bak_dir if os.path.isdir(bak_dir) else cl_dir, fid + '.png'), 0)
        rb = _bev(r, H, bw, bh)
        dist, _ = _wall_dist(rb, vis)
        d0 = _center_dist_stats(_bev(c_old, H, bw, bh), dist)
        path = trace_centerline(rb, vis, half, bw, bh)
        new = render_mask(path, Hi, img_w, img_h, roi_y) if path is not None else None
        if new is None or np.count_nonzero(new) < 300:
            failed.append(fid)
            continue
        d1 = _center_dist_stats(_bev(new, H, bw, bh), dist)
        if d0 is not None:
            before.append(d0)
        if d1 is not None:
            after.append(d1)
        fixed += 1
        if not args.dry_run:
            cv2.imwrite(os.path.join(cl_dir, fid + '.png'), new)
            f['centerline_fix'] = 'geom_half_%.0f' % half
        if args.preview and len(previews) < 12 and i % 15 == 0:
            img = cv2.imread(os.path.join(data_dir, 'images', f['image']))
            ov = img.copy()
            ov[c_old > 127] = (0, 0, 255)
            ov[new > 127] = (0, 255, 255)
            cv2.putText(ov, '%s old=red new=yellow' % fid, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            previews.append(cv2.resize(ov, (400, 300)))

    print('Frame 1_line_parallel: sua=%d, giu nguyen (khong du hinh hoc)=%d' % (fixed, len(failed)))
    if before:
        print('  Khoang cach centerline->mep TRUOC: median=%.1f px' % np.median(before))
    if after:
        print('  Khoang cach centerline->mep SAU  : median=%.1f px (muc tieu %.1f)' % (np.median(after), half))
    if failed:
        print('  Frame khong sinh lai duoc (giu label cu, nen xem lai tay):', ', '.join(failed[:30]))

    if not args.dry_run:
        meta['centerline_fix'] = {'half_bev_px': round(half, 1), 'fixed': fixed, 'failed': failed}
        tmp = meta_path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp, meta_path)
    if args.preview and previews:
        while len(previews) % 3:
            previews.append(np.zeros_like(previews[0]))
        grid = np.vstack([np.hstack(previews[k:k + 3]) for k in range(0, len(previews), 3)])
        d = os.path.dirname(os.path.abspath(args.preview))
        if not os.path.isdir(d):
            os.makedirs(d)
        cv2.imwrite(args.preview, grid)
        print('Anh preview:', args.preview)


if __name__ == '__main__':
    main()
