# -*- coding: utf-8 -*-
"""Thiet lap moi truong & build TensorRT FP16 engine sau khi `git pull` tren Jetson Nano.

Chay tren Jetson Nano (Python 3.6, JetPack 4.5.1, TensorRT 7.1.3):
    python3 tools/setup_jetson_models.py
    python3 tools/setup_jetson_models.py --model resse
    python3 tools/setup_jetson_models.py --force

Chuc nang:
  1. Kiem tra goi Python can thiet (cv2, numpy, yaml, tensorrt, pycuda, ipywidgets).
  2. Kiem tra `trtexec` va tu dong build ca 2 engine FP16 Option B:
       - models/lane_option_b_resse.onnx    -> models/lane_option_b_resse.engine
       - models/lane_option_b_fastscnn.onnx -> models/lane_option_b_fastscnn.engine
  3. Chay thu suy dien 1 frame gia de xac nhan engine nap OK va do latency.
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRTEXEC_CANDIDATES = [
    '/usr/src/tensorrt/bin/trtexec',
    'trtexec',
]

MODELS = {
    'fast': {
        'label': 'Option B #1: LaneTiny-Fast (uu tien toc do >= 25 FPS, 56K params, 0 SE)',
        'onnx': 'models/lane_option_b_fast.onnx',
        'engine': 'models/lane_option_b_fast.engine',
        'config': 'configs/cnn_option_b_fast.yaml',
    },
    'resse': {
        'label': 'Option B #4: LaneTiny-ResSE (can bang, 157K params)',
        'onnx': 'models/lane_option_b_resse.onnx',
        'engine': 'models/lane_option_b_resse.engine',
        'config': 'configs/cnn_option_b.yaml',
    },
    'fastscnn': {
        'label': 'Option B #5: FastSCNN-Lane (du phong / A-B test, 331K params)',
        'onnx': 'models/lane_option_b_fastscnn.onnx',
        'engine': 'models/lane_option_b_fastscnn.engine',
        'config': 'configs/cnn_option_b_fastscnn.yaml',
    },
}


def find_trtexec():
    for path in TRTEXEC_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def check_python_env():
    print('=== BUOC 1: KIEM TRA MOI TRUONG PYTHON ===')
    print('Python :', sys.version.split()[0])
    ok_all = True
    checks = [
        ('cv2', 'OpenCV (co san trong JetPack, KHONG pip install de)', True),
        ('numpy', 'NumPy (co san trong JetPack)', True),
        ('yaml', 'PyYAML -> cai bang: pip3 install --user PyYAML', True),
        ('tensorrt', 'TensorRT (co san theo JetPack 4.5.1)', True),
        ('pycuda.driver',
         'PyCUDA -> cai dung cach (KHONG sudo):\n'
         '    export PATH=/usr/local/cuda/bin:$PATH\n'
         '    export CPATH=/usr/local/cuda/include:$CPATH\n'
         '    export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH\n'
         '    pip3 install --user --no-cache-dir pycuda', True),
        ('ipywidgets', 'ipywidgets (de mo giao dien Speed Track trong JupyterLab)', False),
    ]
    for mod, hint, required in checks:
        try:
            __import__(mod)
            print('  [ OK ] %s' % mod)
        except ImportError as exc:
            tag = 'THIEU' if required else 'CANH BAO'
            print('  [%s] %s (%s)\n         %s' % (tag, mod, exc, hint))
            if required:
                ok_all = False
    return ok_all


def build_one_engine(trtexec, onnx_rel, engine_rel, workspace_mb, force):
    onnx_path = os.path.join(ROOT, onnx_rel)
    engine_path = os.path.join(ROOT, engine_rel)
    if not os.path.isfile(onnx_path):
        print('  [LOI] Khong tim thay file ONNX: %s' % onnx_rel)
        return False

    if (not force and os.path.isfile(engine_path)
            and os.path.getmtime(engine_path) >= os.path.getmtime(onnx_path)):
        size_kb = os.path.getsize(engine_path) / 1024.0
        print('  [BO QUA] %s da ton tai va moi hon ONNX (%.1f KB). '
              'Dung --force de build lai.' % (engine_rel, size_kb))
        return True

    cmd = [
        trtexec,
        '--onnx=%s' % onnx_path,
        '--fp16',
        '--workspace=%d' % int(workspace_mb),
        '--saveEngine=%s' % engine_path,
    ]
    print('  Dang build FP16 engine (mat khoang 1-3 phut tren Jetson Nano)...')
    print('  $ %s' % ' '.join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT)
    rc = proc.wait()
    dt = time.time() - t0
    if rc != 0 or not os.path.isfile(engine_path):
        print('  [LOI] trtexec that bai (ma %d sau %.1fs).' % (rc, dt))
        return False
    size_kb = os.path.getsize(engine_path) / 1024.0
    print('  [ OK ] Da tao %s (%.1f KB trong %.1fs)' % (engine_rel, size_kb, dt))
    return True


def verify_engine(config_rel):
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    import numpy as np
    from jetracer_baseline.config import load_config
    from jetracer_baseline.perception.lane_cnn import CnnLaneDetector

    cfg = load_config(
        os.path.join(ROOT, 'configs', 'default.yaml'),
        [os.path.join(ROOT, config_rel)],
    )
    det = CnnLaneDetector(cfg)
    try:
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        inf_times = []
        tot_times = []
        for _ in range(30):
            det.process(dummy)
            inf_times.append(det.last_infer_ms)
            tot_times.append(det.last_total_ms)
        inf_p50 = float(np.percentile(np.array(inf_times), 50))
        inf_p95 = float(np.percentile(np.array(inf_times), 95))
        tot_p50 = float(np.percentile(np.array(tot_times), 50))
        tot_p95 = float(np.percentile(np.array(tot_times), 95))
        fps_p50 = 1000.0 / max(tot_p50, 1e-6)
        has_3ch = det.last_white_mask is not None and det.last_road_mask is not None
        print('  [ OK ] Nap TensorRT OK | 3 kenh=%s | infer p50=%.2fms p95=%.2fms | '
              'tong p50=%.2fms p95=%.2fms (~%.1f FPS)'
              % ('CO' if has_3ch else 'KHONG', inf_p50, inf_p95, tot_p50, tot_p95, fps_p50))
        return has_3ch
    finally:
        det.close()


def main():
    ap = argparse.ArgumentParser(
        description='Kiem tra Jetson Nano va build TensorRT FP16 engine cho Option B.')
    ap.add_argument('--model', choices=['all', 'fast', 'resse', 'fastscnn'],
                    default='all', help='Chon model de build (mac dinh: all)')
    ap.add_argument('--workspace', type=int, default=256,
                    help='Bo nho workspace TensorRT (MB, mac dinh: 256)')
    ap.add_argument('--force', action='store_true',
                    help='Build lai du file .engine da ton tai')
    ap.add_argument('--check-only', action='store_true',
                    help='Chi kiem tra moi truong va file ONNX, khong build')
    args = ap.parse_args()

    env_ok = check_python_env()
    trtexec = find_trtexec()
    print('\n=== BUOC 2: KIEM TRA TRTEXEC & FILE ONNX ===')
    print('trtexec:', trtexec if trtexec else 'KHONG TIM THAY (/usr/src/tensorrt/bin/trtexec)')

    selected = ['fast', 'resse', 'fastscnn'] if args.model == 'all' else [args.model]
    for key in selected:
        info = MODELS[key]
        onnx_path = os.path.join(ROOT, info['onnx'])
        exists = os.path.isfile(onnx_path)
        size_kb = (os.path.getsize(onnx_path) / 1024.0) if exists else 0.0
        print('  [%s] %s (%.1f KB)' % (' OK ' if exists else 'THIEU', info['onnx'], size_kb))

    if args.check_only:
        return 0 if (env_ok and trtexec) else 1

    if not trtexec:
        print('\n[DUNG] Khong tim thay trtexec. Script nay phai chay TREN JETSON NANO.')
        return 2

    all_ok = True
    print('\n=== BUOC 3: BUILD TENSORRT FP16 ENGINE ===')
    for key in selected:
        info = MODELS[key]
        print('\n--- %s ---' % info['label'])
        ok = build_one_engine(trtexec, info['onnx'], info['engine'],
                              args.workspace, args.force)
        if not ok:
            all_ok = False
            continue
        if env_ok:
            try:
                verify_engine(info['config'])
            except Exception as exc:
                print('  [LOI] Kiem tra engine that bai: %s' % exc)
                all_ok = False

    print('\n=== HOAN TAT ===')
    if all_ok:
        print('San sang chay Speed Track! Cac buoc tiep theo:')
        print('  1. Do latency & kiem tra dau CTE (day xe sang trai -> cte duong):')
        print('       python3 tools/bench_lane_cnn.py --override configs/cnn_option_b_fast.yaml')
        print('       python3 tools/bench_lane_cnn.py --override configs/cnn_option_b_fast.yaml --watch --source csi')
        print('  2. Mo giao dien JupyterLab (tune_lane.ipynb):')
        print('       jupyter lab --ip=0.0.0.0 --no-browser')
        print('  3. Hoac chay thang bang CLI (khong ton CPU ve UI -> FPS cao nhat):')
        print('       python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia '
              '--override configs/cnn_option_b_fast.yaml --record')
        return 0
    print('Co buoc bi loi - xem chi tiet o tren.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
