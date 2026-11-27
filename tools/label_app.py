# -*- coding: utf-8 -*-
"""Local Web Server cho app HTML kiem tra va sua label Option B (3 kenh).

Cach chay:
  cd D:\\Competition\\Jetson_Race_SS2\\JetsonRacer
  python tools/label_app.py --data data/lane_option_b --port 8765

Mo trinh duyet tai: http://localhost:8765
Khong can cai them thu vien ngoai (chi dung http.server + cv2 + numpy).
"""

from __future__ import print_function

import argparse
import base64
import json
import os
import sys
import threading

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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from prelabel_option_b import (  # noqa: E402
    DEFAULT_HALF_W_BEV,
    compute_centerline_from_masks,
)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _decode_png_b64(data_url):
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError('Khong decode duoc PNG mask')
    if img.ndim == 3:
        # Neu canvas gui RGBA, lay alpha hoac kenh sang nhat
        if img.shape[2] == 4:
            mask = (img[:, :, 3] > 127).astype(np.uint8) * 255
        else:
            mask = (np.max(img, axis=2) > 127).astype(np.uint8) * 255
    else:
        mask = (img > 127).astype(np.uint8) * 255
    return mask


def _encode_png_b64(mask):
    ok, buf = cv2.imencode('.png', mask)
    if not ok:
        raise RuntimeError('Khong encode duoc PNG')
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return 'data:image/png;base64,' + b64


class LabelStore(object):
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.meta_path = os.path.join(self.data_dir, 'metadata.json')
        self.lock = threading.Lock()
        if not os.path.isfile(self.meta_path):
            raise IOError('Khong thay metadata.json tai: %s' % self.meta_path)
        with open(self.meta_path, 'r') as fh:
            self.meta = json.load(fh)
        self.frame_map = dict((f['id'], f) for f in self.meta.get('frames', []))

    def save_meta(self):
        tmp = self.meta_path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.meta, fh, indent=2)
        if os.path.isfile(self.meta_path):
            os.replace(tmp, self.meta_path)
        else:
            os.rename(tmp, self.meta_path)

    def mask_path(self, layer, frame_id):
        return os.path.join(self.data_dir, 'masks', layer, frame_id + '.png')

    def image_path(self, fname):
        return os.path.join(self.data_dir, 'images', fname)

    def save_frame_masks(self, frame_id, road_mask, white_mask, center_mask,
                         status=None, half_w_bev=None, info=None):
        with self.lock:
            if frame_id not in self.frame_map:
                raise KeyError('Khong thay frame_id: %s' % frame_id)
            cv2.imwrite(self.mask_path('road', frame_id), road_mask)
            cv2.imwrite(self.mask_path('white', frame_id), white_mask)
            cv2.imwrite(self.mask_path('centerline', frame_id), center_mask)
            rec = self.frame_map[frame_id]
            if status is not None:
                rec['status'] = status
            if half_w_bev is not None:
                rec['half_w_bev'] = float(half_w_bev)
            if info is not None:
                rec['mode'] = info['mode']
                rec['cte'] = info['cte']
                rec['cte_lookahead'] = info['cte_lookahead']
                rec['curvature'] = info['curvature']
                rec['steer_deg'] = info['steer_deg']
                rec['lookahead_img'] = info['lookahead_img']
            self.save_meta()
            return dict(rec)

    def set_status(self, frame_id, status):
        with self.lock:
            if frame_id not in self.frame_map:
                raise KeyError('Khong thay frame_id: %s' % frame_id)
            rec = self.frame_map[frame_id]
            rec['status'] = status
            self.save_meta()
            return dict(rec)


def make_handler(store, html_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Giam log nhieu tren console
            pass

        def _send_json(self, obj, code=200):
            payload = json.dumps(obj).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def _send_file(self, path, content_type):
            if not os.path.isfile(path):
                self.send_error(404, 'Not found')
                return
            with open(path, 'rb') as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self):
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length)
            return json.loads(raw.decode('utf-8'))

        def do_GET(self):
            parsed = urlparse(self.path)
            p = parsed.path

            if p == '/' or p == '/index.html':
                self._send_file(html_path, 'text/html; charset=utf-8')
                return

            if p == '/api/metadata':
                with store.lock:
                    self._send_json(store.meta)
                return

            if p.startswith('/images/'):
                fname = os.path.basename(p[len('/images/'):])
                self._send_file(store.image_path(fname), 'image/jpeg')
                return

            if p.startswith('/masks/'):
                parts = p[len('/masks/'):].split('/')
                if len(parts) == 2 and parts[0] in ('road', 'white', 'centerline'):
                    layer = parts[0]
                    fname = os.path.basename(parts[1])
                    frame_id = os.path.splitext(fname)[0]
                    self._send_file(store.mask_path(layer, frame_id), 'image/png')
                    return

            self.send_error(404, 'Not found')

        def do_POST(self):
            parsed = urlparse(self.path)
            p = parsed.path
            try:
                body = self._read_json_body()

                if p == '/api/recompute':
                    road = _decode_png_b64(body['road'])
                    white = _decode_png_b64(body['white'])
                    half_w = float(body.get('half_w_bev', DEFAULT_HALF_W_BEV))
                    center, info = compute_centerline_from_masks(
                        road, white, half_w_bev=half_w
                    )
                    self._send_json({
                        'ok': True,
                        'centerline': _encode_png_b64(center),
                        'info': info,
                    })
                    return

                if p == '/api/save':
                    frame_id = body['id']
                    road = _decode_png_b64(body['road'])
                    white = _decode_png_b64(body['white'])
                    center = _decode_png_b64(body['centerline'])
                    status = body.get('status')
                    half_w = float(body.get('half_w_bev', DEFAULT_HALF_W_BEV))
                    # Cap nhat lai metrics tu road/white hien tai (giu nguyen mask center nguoi dung da ve)
                    _, info = compute_centerline_from_masks(
                        road, white, half_w_bev=half_w
                    )
                    rec = store.save_frame_masks(
                        frame_id, road, white, center,
                        status=status, half_w_bev=half_w, info=info,
                    )
                    self._send_json({'ok': True, 'frame': rec})
                    return

                if p == '/api/status':
                    rec = store.set_status(body['id'], body['status'])
                    self._send_json({'ok': True, 'frame': rec})
                    return

                self.send_error(404, 'Unknown endpoint')
            except Exception as exc:
                self._send_json({'ok': False, 'error': str(exc)}, code=400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description='App HTML sua & kiem tra label Option B')
    parser.add_argument('--data', default='data/lane_option_b', help='Thu muc dataset Option B')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    html_path = os.path.join(TOOLS_DIR, 'label_app.html')
    if not os.path.isfile(html_path):
        raise IOError('Khong thay file giao dien: %s' % html_path)

    store = LabelStore(args.data)
    handler_cls = make_handler(store, html_path)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print('=== JetRacer Option B Label Studio ===')
    print('Dataset : %s (%d frames)' % (store.data_dir, len(store.meta.get('frames', []))))
    print('URL     : http://%s:%d' % (args.host, args.port))
    print('Nhan Ctrl+C de dung server.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nDa dung server.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
