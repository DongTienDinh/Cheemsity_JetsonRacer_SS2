# -*- coding: utf-8 -*-
"""Bam duong tam bang CNN chay tren TensorRT. Thay the DUNG BUOC TAO MASK.

Doi lai cua `lane.py` (CV co dien) chi la nguon mask:

    lane.py      anh -> nguong HSV -> mask VET SON  -\
                                                      >-- warp -> band -> fit -> cte -> PID
    lane_cnn.py  anh -> TensorRT   -> mask DUONG TAM -/

Tra ve dung `LaneResult` cua lane.py nen `pipeline.py`, PID, FSM, driver khong
phai sua mot dong nao.

VI SAO HAU XU LY O DAY NGAN HON lane.py NHIEU:
lane.py phai gan loc vien lane, chon cum theo anchor, chong nhay - vi mask mau
cua no lan lon ca vien lane cung mau. Mask CNN chi co DUNG MOT dai lien tuc,
khong vien khong dom khong khoang trong, nen chi con: chia band -> lay cum manh
nhat -> fit. Do la loi cua viec dat nhan la duong lien tuc thay vi cac vet son.

CHUOI TIEN/HAU XU LY PHAI GIONG HET LUC TRAIN.
Cac hang so hinh hoc duoi day sao chep tu `training/lane_post.py`. Sua mot ben
ma khong sua ben kia thi model KHONG BAO LOI GI - no chi te di. Doi luon la doi
CA HAI, va export lai ONNX.

Co y KHONG doc `lane.warp_src` tu config: khoa do dung de tune duong CV. Ai do
chinh no cho CV se lam lech am tham hinh hoc cua CNN. Muon doi rieng cho CNN thi
dung khoa `lane.cnn.*`.

Python 3.6 (JetPack 4.5.1, TensorRT 7.1.3 - DA XAC NHAN TREN XE).
Can: tensorrt (co san theo JetPack), pycuda (cai them).
Cai pycuda PHAI xuat bien CUDA truoc, khong thi loi 'cuda.h: No such file':
    export PATH=/usr/local/cuda/bin:$PATH
    export CPATH=/usr/local/cuda/include:$CPATH
    export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
    pip3 install --user --no-cache-dir pycuda      # KHONG dung sudo
"""

from __future__ import print_function

import os
import time

import cv2
import numpy as np

from .lane import LaneResult

# --- Sao chep tu training/lane_post.py. DOI THI DOI CA HAI. ----------------
PROC_W, PROC_H = 320, 240
ROI_TOP = 0.55
ROI_Y0 = int(PROC_H * ROI_TOP)
IN_W, IN_H = 256, 128            # dau vao model
SEG_W, SEG_H = 128, 64           # dau ra seg (stride 2)
WARP_SRC = [[0.10, 1.00], [0.90, 1.00], [0.62, 0.58], [0.38, 0.58]]
WARP_DST_MARGIN = 0.25
N_BANDS = 10
BAND_MIN_PIXELS = 10
MIN_BANDS = 2
MAX_RUN_FRAC = 0.60
LOOKAHEAD = 0.6
# ---------------------------------------------------------------------------


def _build_warp(w, h, src_pts, margin):
    src = np.float32([[p[0] * w, p[1] * h] for p in src_pts])
    dst = np.float32([[margin * w, h], [(1.0 - margin) * w, h],
                      [(1.0 - margin) * w, 0.0], [margin * w, 0.0]])
    return cv2.getPerspectiveTransform(src, dst)


class TensorRTEngine(object):
    """Boc engine TensorRT. Ho tro ca API binding (TRT 8.0) va tensor (TRT >=8.5).

    Vi sao ho tro ca hai: xe chay TRT 7.1.3 dung API binding, nhung may dev hoac
    ban JetPack khac co the la 8.5+ noi API do da bo. Doan if/else nay re hon
    nhieu so voi mot buoi debug vi doi image.
    """

    def __init__(self, engine_path):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            # CO Y KHONG dung `pycuda.autoinit`. autoinit tao context va day len
            # NGAN XEP CUA DUNG THREAD DANG IMPORT. Detector duoc tao o thread
            # chinh, con vong camera cua giao dien chay o THREAD NEN -> thread do
            # khong co context nao, moi loi goi CUDA deu that bai.
            # Trieu chung khong phai "bao loi ro rang" ma la: panel dung im o
            # khung hinh dau tien, mask va cte khong doi du camera van cap nhat -
            # vi `_loop` bat exception roi break, chi con anh cuoi cung tren man.
            # Cach dung dung: giu primary context, push/pop quanh moi lan suy dien.
            cuda.init()
        except ImportError as exc:
            raise ImportError(
                '%s.\n'
                'lane.mode=cnn CHI chay duoc tren Jetson. tensorrt co san theo\n'
                'JetPack; pycuda phai cai them va PHAI xuat bien CUDA truoc,\n'
                'neu khong se loi "cuda.h: No such file or directory":\n'
                '  export PATH=/usr/local/cuda/bin:$PATH\n'
                '  export CPATH=/usr/local/cuda/include:$CPATH\n'
                '  export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH\n'
                '  pip3 install --user --no-cache-dir pycuda   (KHONG dung sudo:\n'
                '  sudo xoa sach cac bien vua xuat)\n'
                'Tren laptop: dat lane.cnn.backend=onnx de chay bang ONNXRuntime,\n'
                'hoac lane.cnn.engine rong de chi dung hau xu ly.'
                % exc)
        if not os.path.isfile(engine_path):
            raise IOError(
                'Khong thay engine: %s\n'
                'Build TREN CHINH XE:\n'
                '  /usr/src/tensorrt/bin/trtexec --onnx=lane_tiny.onnx '
                '--fp16 --workspace=256 --saveEngine=%s'
                % (engine_path, engine_path))

        import threading
        self._cuda = cuda
        self._ctx = cuda.Device(0).retain_primary_context()
        self._closed = False
        self.engine = None
        self.context = None
        self.stream = None
        self.host, self.dev, self.names, self.shapes = [], [], [], []
        self.in_idx, self.out_idx = [], []
        # TensorRT execution context KHONG an toan da luong. Giao dien co ca
        # thread camera lan thread dieu khien -> phai khoa.
        self._lock = threading.Lock()

        logger = trt.Logger(trt.Logger.WARNING)
        self._ctx.push()
        try:
            try:
                self._build(trt, cuda, engine_path, logger)
            finally:
                self._ctx.pop()
        except BaseException:
            self.close()
            raise

    def _build(self, trt, cuda, engine_path, logger):
        with open(engine_path, 'rb') as fh:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(
                'TensorRT khong doc duoc %s. Gan nhu chac chan engine nay build '
                'tren may/phien ban khac - engine KHONG chuyen may duoc.'
                % engine_path)
        self.context = self.engine.create_execution_context()
        self._new_api = hasattr(self.engine, 'num_io_tensors')

        # THU TU BINDING CUA ENGINE KHONG PHAI THU TU KHAI BAO TRONG ONNX.
        # Do binding nao la vao/ra bang API cua engine, va tra ket qua theo TEN.
        # Ban dau file nay gia dinh outs[0] = seg, outs[1] = reg -> tren xe
        # TensorRT tra reg truoc va no gay "cannot reshape array of size 3 into
        # shape (64,128)". Loi chi lo ra khi co engine that, khong cach nao bat
        # duoc tren laptop.
        self.host, self.dev, self.names, self.shapes = [], [], [], []
        self.in_idx, self.out_idx = [], []
        if self._new_api:
            for i in range(self.engine.num_io_tensors):
                nm = self.engine.get_tensor_name(i)
                is_in = (self.engine.get_tensor_mode(nm) ==
                         trt.TensorIOMode.INPUT)
                self._alloc(nm, tuple(self.engine.get_tensor_shape(nm)),
                            trt.nptype(self.engine.get_tensor_dtype(nm)), is_in)
        else:
            for i in range(self.engine.num_bindings):
                self._alloc(self.engine.get_binding_name(i),
                            tuple(self.engine.get_binding_shape(i)),
                            trt.nptype(self.engine.get_binding_dtype(i)),
                            self.engine.binding_is_input(i))
        if not self.in_idx or not self.out_idx:
            raise RuntimeError('Engine khong co du input/output: %s' % self.names)
        self.stream = cuda.Stream()
        self.bindings = [int(d) for d in self.dev]

    def _alloc(self, name, shape, dtype, is_input):
        idx = len(self.host)
        host = self._cuda.pagelocked_empty(int(np.prod(shape)), dtype)
        self.host.append(host)
        self.dev.append(self._cuda.mem_alloc(host.nbytes))
        self.names.append(name)
        self.shapes.append(shape)
        (self.in_idx if is_input else self.out_idx).append(idx)

    def infer(self, x):
        """x: float32 (1,3,H,W) lien tuc -> dict {ten: mang}.

        Push/pop context moi lan goi de chay duoc tu BAT KY thread nao.
        """
        with self._lock:
            if self._closed or self._ctx is None:
                raise RuntimeError('TensorRTEngine da dong')
            self._ctx.push()
            try:
                return self._infer_locked(x)
            finally:
                self._ctx.pop()

    def close(self):
        """Giai phong buffer GPU/TRT TRONG primary context roi moi detach.

        Neu de Python GC pha engine/buffer NGOAI context (vi du `del _det`
        trong notebook truoc `launch()`), PyCUDA canh bao dead context va ro ri
        bo nho tren Jetson Nano 4GB.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            ctx = self._ctx
            self._ctx = None
            if ctx is None:
                return
            try:
                ctx.push()
                try:
                    self.context = None
                    self.engine = None
                    self.stream = None
                    self.dev = []
                    self.host = []
                    self.bindings = []
                finally:
                    ctx.pop()
            except Exception:
                pass
            try:
                ctx.detach()
            except Exception:
                pass

    def _infer_locked(self, x):
        cuda = self._cuda
        i0 = self.in_idx[0]
        np.copyto(self.host[i0], x.ravel())
        cuda.memcpy_htod_async(self.dev[i0], self.host[i0], self.stream)
        if self._new_api:
            for nm, d in zip(self.names, self.dev):
                self.context.set_tensor_address(nm, int(d))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings,
                                          stream_handle=self.stream.handle)
        for i in self.out_idx:
            cuda.memcpy_dtoh_async(self.host[i], self.dev[i], self.stream)
        self.stream.synchronize()
        # `.copy()`: host buffer la pagelocked va bi GHI DE o lan infer sau.
        # Tra ve view thi ket qua co the doi duoi tay nguoi goi.
        return dict((self.names[i],
                     self.host[i].reshape(self.shapes[i]).copy())
                    for i in self.out_idx)


class OnnxEngine(object):
    """Chay bang ONNXRuntime thay TensorRT. CHAM HON NHIEU - khong dung de thi.

    Ly do co mat: cho phep chay TOAN BO duong deploy (preprocess -> model ->
    hau xu ly -> cte) tren laptop, khong can Jetson. Nho no, phan duy nhat chua
    duoc kiem chung khi len xe chi con dung mot thu: TensorRT. Ngoai ra no la
    luoi an toan neu trtexec khong build duoc vao phut chot.
    """

    def __init__(self, onnx_path):
        import onnxruntime as ort
        if not os.path.isfile(onnx_path):
            raise IOError('Khong thay file ONNX: %s' % onnx_path)
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            onnx_path, opt, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def infer(self, x):
        outs = self.sess.run(None, {self.input_name: x})
        return dict(zip(self.out_names, outs))


class CnnLaneDetector(object):
    """Cung giao dien voi LaneDetector: .process(frame_bgr) -> LaneResult."""

    def __init__(self, cfg):
        c = cfg.get
        engine = c('lane.cnn.engine', 'models/lane_tiny.engine')
        # 'trt' = TensorRT (tren xe). 'onnx' = ONNXRuntime (kiem chung tren PC).
        self.backend = str(c('lane.cnn.backend', 'trt'))
        self.roi_top = float(c('lane.cnn.roi_top', ROI_TOP))
        self.engine_path = engine
        self.update_config(cfg)
        self.M = _build_warp(PROC_W, PROC_H,
                             c('lane.cnn.warp_src', WARP_SRC),
                             float(c('lane.cnn.warp_dst_margin', WARP_DST_MARGIN)))
        self.roi_y0 = int(PROC_H * self.roi_top)
        roi_h = max(1, PROC_H - self.roi_y0)
        # Gop resize (SEG_W x SEG_H -> PROC_W x roi_h) + offset roi_y0 + warpPerspective
        # thanh MOT ma tran homography duy nhat -> bo 2 buoc resize trung gian va cap phat canvas.
        seg_to_proc = np.array([
            [PROC_W / float(SEG_W), 0.0, 0.0],
            [0.0, roi_h / float(SEG_H), float(self.roi_y0)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self._M_seg = self.M.dot(seg_to_proc)
        # Bang tra cuu BEV tinh san (per-pixel LUT) cho mask 64x128: thay the
        # hoan toan cv2.warpPerspective(320x240) + np.count_nonzero(76800 px)
        # bang phep loc sparse tren ~200-400 pixel centerline (~15-20x nhanh hon
        # tren Jetson Nano 5W, giu nguyen do chinh xac CTE/Lookahead).
        self._build_bev_lut()
        # Buffer tai su dung giua cac frame de giam cap phat RAM / GC tren Jetson Nano
        self._in_buf = np.empty((1, 3, IN_H, IN_W), dtype=np.float32)
        self._has_cv_dnn = hasattr(cv2, 'dnn') and hasattr(cv2.dnn, 'blobFromImage')
        self._canvas = np.zeros((PROC_H, PROC_W), dtype=np.uint8)
        self._xs_proc = np.arange(PROC_W, dtype=np.float32)
        self._xs_seg = np.arange(SEG_W, dtype=np.float32)
        self._hot_pad = np.zeros(PROC_W + 2, dtype=np.int8)
        # `engine: ''` -> chi dung duoc hau xu ly, khong suy dien. Duong nay ton
        # tai de doi chieu hau xu ly cua XE voi training/lane_post.py NGAY TREN
        # LAPTOP - neu hai ben lech nhau thi moi con so do o PC deu vo nghia.
        if not engine:
            self.engine = None
        elif self.backend == 'onnx':
            self.engine = OnnxEngine(engine)
        else:
            self.engine = TensorRTEngine(engine)

        self.last_infer_ms = 0.0
        self.last_total_ms = 0.0
        self.last_mask = None        # mask ROI 256x128 dung de fit (centerline hoac fallback)
        self.last_road_mask = None   # Option B Ch0 (64x128) neu model 3 kenh
        self.last_white_mask = None  # Option B Ch1 (64x128) neu model 3 kenh
        self.last_center_mask = None # Option B Ch2 (64x128) neu model 3 kenh
        self.last_left_line = False
        self.last_right_line = False
        self.last_two_lines_ok = False
        self.last_used_fallback = False
        self.last_bev = None         # cung mask do sau khi warp sang BEV
        self.n_disagree = 0
        self.n_fallback = 0
        self._reset_state()
        if self.engine is not None:
            self._warmup()

    def update_config(self, cfg):
        """Doc lai cac tham so RE tu config. KHONG dung toi engine TensorRT.

        Ton tai vi giao dien tune goi `rebuild()` moi lan keo slider. Neu de no
        dung lai ca detector thi moi lan keo se nap lai engine: treo giao dien
        vai giay, va cap phat lai bo nho GPU ma khong giai phong cai cu.
        Nhung tham so duoi day deu chi la so - doi tuc thi, khong ton gi.
        """
        c = cfg.get
        self.alpha = float(c('lane.smooth_alpha', 0.6))
        self.min_bands = int(c('lane.cnn.min_bands', MIN_BANDS))
        self.band_min_pixels = int(c('lane.cnn.band_min_pixels', BAND_MIN_PIXELS))
        # Diem ngam xa: truoc day la hang so nen slider khong co tac dung gi -
        # nguoi tune keo mai ma khong hieu sao xe khong doi.
        self.lookahead = float(c('lane.cnn.lookahead', LOOKAHEAD))
        # Nguong bat dong y giua seg head va reg head. Vuot nguong = hai duong
        # tinh doc lap trong model cho ket qua khac nhau -> khong tin duoc nua,
        # bao found=False de FSM ha ga / roi ve CV.
        self.reg_disagree = float(c('lane.cnn.reg_disagree', 0.35))
        self.check_reg = bool(c('lane.cnn.check_reg', True))
        # Cac che do du phong Option B tu mat duong Ch0 va bien trang Ch1.
        self.road_fallback = bool(c('lane.cnn.road_fallback', False))
        self.single_line_fallback = bool(c('lane.cnn.single_line_fallback', False))
        self.single_line_offset_px = float(c('lane.cnn.single_line_offset_px', 108.0))

    def _reset_state(self):
        self._cte = 0.0
        self._look = 0.0
        self._curv = 0.0
        self._init = False

    def reset(self):
        self._reset_state()

    def close(self):
        engine = getattr(self, 'engine', None)
        if engine is not None and hasattr(engine, 'close'):
            engine.close()

    def _warmup(self, n=3):
        """Lan suy dien dau tien cua TensorRT ton hang tram ms. Dot no o day,
        khong phai o frame dau tien cua luot thi."""
        x = np.zeros((1, 3, IN_H, IN_W), np.float32)
        for _ in range(n):
            self.engine.infer(x)

    def _build_bev_lut(self):
        """Tinh truoc bang tra cuu tu tung pixel 64x128 sang band BEV (0..9) va x_bev.

        Chieu nguoc moi pixel (xb, yb) cua anh BEV 320x240 qua M_seg^-1 ve pixel
        (xs, ys) cua mask 64x128 -> dem dung dien tich pixel BEV va tong toa do x
        cho tung pixel 64x128. Tuong duong toan hoc voi warpPerspective(INTER_NEAREST)
        nhung khi chay chi dung ~300 pixel active thay vi 76,800 pixel.
        """
        valid = cv2.warpPerspective(np.ones((SEG_H, SEG_W), np.uint8), self._M_seg,
                                    (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)
        self._edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # Bo vien cua anh BEV va bien do camera cat khoi cac moc tinh khoang cach.
        self._bev_visible = cv2.erode(valid, self._edge_kernel, borderType=cv2.BORDER_CONSTANT,
                                      borderValue=0) > 0
        m_inv = np.linalg.inv(self._M_seg)
        yy, xx = np.mgrid[0:PROC_H, 0:PROC_W]
        denom = m_inv[2, 0] * xx + m_inv[2, 1] * yy + m_inv[2, 2]
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        sx = np.rint((m_inv[0, 0] * xx + m_inv[0, 1] * yy + m_inv[0, 2]) / denom).astype(np.int32)
        sy = np.rint((m_inv[1, 0] * xx + m_inv[1, 1] * yy + m_inv[1, 2]) / denom).astype(np.int32)
        valid = (sx >= 0) & (sx < SEG_W) & (sy >= 0) & (sy < SEG_H)
        band_h = max(1, PROC_H // N_BANDS)
        band_id = np.clip((PROC_H - 1 - yy) // band_h, 0, N_BANDS - 1)
        flat_s = (sy[valid] * SEG_W + sx[valid]).astype(np.int32)
        b_v = band_id[valid].astype(np.int32)
        x_v = xx[valid].astype(np.float64)

        n_pix = SEG_H * SEG_W
        pix_w = np.bincount(flat_s, minlength=n_pix).astype(np.float32)
        pix_wx = np.bincount(flat_s, weights=x_v, minlength=n_pix).astype(np.float32)
        pix_wb = np.bincount(flat_s, weights=b_v, minlength=n_pix).astype(np.float32)
        pix_b = np.full(n_pix, -1, dtype=np.int8)
        has = pix_w > 0.0
        pix_b[has] = np.clip(
            np.rint(pix_wb[has] / pix_w[has]).astype(np.int32),
            0, N_BANDS - 1).astype(np.int8)
        self._lut_b = pix_b
        self._lut_w = pix_w
        self._lut_wx = pix_wx
        self._band_ts = (np.arange(N_BANDS, dtype=np.float64) + 0.5) / float(N_BANDS)

    @staticmethod
    def _fit_poly_fast(ts, xs):
        """Fit da thuc bac 2 (hoac bac 1 neu 2 diem) bang he phuong trinh chuan 3x3."""
        n = len(ts)
        if n < 3:
            s0 = float(n)
            s1 = float(ts.sum())
            s2 = float((ts * ts).sum())
            y0 = float(xs.sum())
            y1 = float((ts * xs).sum())
            d = s0 * s2 - s1 * s1
            if abs(d) < 1e-12:
                return np.array([0.0, 0.0, y0 / max(s0, 1.0)])
            c1 = (s0 * y1 - s1 * y0) / d
            c2 = (y0 - c1 * s1) / s0
            return np.array([0.0, c1, c2])
        t2 = ts * ts
        s0 = float(n)
        s1 = float(ts.sum())
        s2 = float(t2.sum())
        s3 = float((t2 * ts).sum())
        s4 = float((t2 * t2).sum())
        y0 = float(xs.sum())
        y1 = float((ts * xs).sum())
        y2 = float((t2 * xs).sum())
        a_mat = np.array([[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]], dtype=np.float64)
        b_vec = np.array([y2, y1, y0], dtype=np.float64)
        try:
            return np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            return np.polyfit(ts, xs, 2)

    def _mask_to_fit_lut(self, seg2_or_mask):
        """Fit duong tam truc tiep tu mang 64x128 (logits float32 hoac mask uint8) KHONG can warpPerspective."""
        idx = np.flatnonzero(seg2_or_mask.ravel() > 0)
        if len(idx) == 0:
            return None, 0.0, 0, 0
        b = self._lut_b[idx]
        v = b >= 0
        if not v.any():
            return None, 0.0, 0, 0
        b = b[v]
        iv = idx[v]
        w_band = np.bincount(b, weights=self._lut_w[iv], minlength=N_BANDS)
        ok = w_band >= float(self.band_min_pixels)
        n_b = int(ok.sum())
        npix = int(w_band[ok].sum()) if n_b > 0 else 0
        if n_b < self.min_bands:
            return None, 0.0, n_b, npix
        wx_band = np.bincount(b, weights=self._lut_wx[iv], minlength=N_BANDS)
        ts = self._band_ts[ok]
        xs = wx_band[ok] / w_band[ok]
        co = self._fit_poly_fast(ts, xs)
        return co, float(ts[-1]), n_b, npix

    # ------------------------------------------------------------ tien xu ly
    def preprocess(self, frame_bgr):
        """Crop hang [roi_top*H, H] -> resize 256x128 -> BGR float32 0..255 NCHW.

        Dung cv2.dnn.blobFromImage (C++ OpenCV NEON: gop resize INTER_LINEAR +
        chuyen HWC uint8 -> NCHW float32 trong 1 buoc, nhanh hon nhieu so voi
        NumPy stride-3 tren ARM Cortex-A57). Chuan hoa nam TRONG graph ONNX.
        """
        h = frame_bgr.shape[0]
        roi = frame_bgr[int(h * self.roi_top):, :]
        if self._has_cv_dnn:
            return cv2.dnn.blobFromImage(
                roi, 1.0, (IN_W, IN_H), (0.0, 0.0, 0.0),
                swapRB=False, crop=False)
        small = cv2.resize(roi, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        buf = self._in_buf
        buf[0, 0] = small[:, :, 0]
        buf[0, 1] = small[:, :, 1]
        buf[0, 2] = small[:, :, 2]
        return buf

    # ------------------------------------------------------------- hau xu ly
    def _band_centre_from_cols(self, cols):
        if int(cols.sum()) < self.band_min_pixels:
            return None, 0
        hot_pad = self._hot_pad
        hot_pad[1:-1] = (cols > 0)
        edges = np.diff(hot_pad)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        max_run = MAX_RUN_FRAC * PROC_W
        xs_proc = self._xs_proc
        best_x, best_n = None, 0.0
        for s, e in zip(starts, ends):
            if (e - s) > max_run:
                continue
            w = cols[s:e]
            n = float(w.sum())
            if n <= best_n:
                continue
            best_x, best_n = float(np.dot(w, xs_proc[s:e]) / n), n
        if best_x is None or best_n < self.band_min_pixels:
            return None, 0
        return best_x, int(best_n)

    def _band_centre(self, band):
        return self._band_centre_from_cols(np.count_nonzero(band, axis=0))

    def mask_to_fit(self, mask, compute_bev=True):
        """mask ROI (64x128 hoac 128x256, 0/255) -> (coeff, t_max, n_bands, n_pixels)."""
        if mask.shape[0] == SEG_H and mask.shape[1] == SEG_W:
            co, t_max, n_bands, npix = self._mask_to_fit_lut(mask)
            if compute_bev:
                self.last_bev = cv2.warpPerspective(
                    mask, self._M_seg, (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)
            else:
                self.last_bev = None
            return co, t_max, n_bands, npix

        roi_h = PROC_H - self.roi_y0
        m = cv2.resize(mask, (PROC_W, roi_h), interpolation=cv2.INTER_NEAREST)
        canvas = self._canvas
        canvas[self.roi_y0:, :] = m
        bev = cv2.warpPerspective(
            canvas, self.M, (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)
        self.last_bev = bev

        band_h = max(1, PROC_H // N_BANDS)
        pts, npix = [], 0
        if PROC_H == N_BANDS * band_h:
            cols_all = np.count_nonzero(
                bev.reshape(N_BANDS, band_h, PROC_W), axis=1)
            for b in range(N_BANDS):
                cx, n = self._band_centre_from_cols(cols_all[N_BANDS - 1 - b])
                if cx is None:
                    continue
                pts.append(((b + 0.5) / float(N_BANDS), cx))
                npix += n
        else:
            for b in range(N_BANDS):
                y1 = PROC_H - b * band_h
                y0 = max(0, y1 - band_h)
                if y1 <= 0:
                    break
                cx, n = self._band_centre(bev[y0:y1, :])
                if cx is None:
                    continue
                pts.append(((b + 0.5) / float(N_BANDS), cx))
                npix += n
        if len(pts) < self.min_bands:
            return None, 0.0, len(pts), npix
        ts = np.array([p[0] for p in pts], np.float64)
        xs = np.array([p[1] for p in pts], np.float64)
        co = self._fit_poly_fast(ts, xs)
        return co, float(ts.max()), len(pts), npix

    @staticmethod
    def _pick_outputs(outs):
        """Lay seg/reg tu dict output. Uu tien TEN, du phong theo KICH THUOC.

        Du phong ton tai vi mot so ban TensorRT doi ten binding khi build engine.
        Ho tro ca model 1 kenh cu (64x128 = 8192) va Option B 3 kenh
        (3x64x128 = 24576: Ch0=road, Ch1=white_line, Ch2=centerline).
        """
        seg, reg = outs.get('seg'), outs.get('reg')
        if seg is None or reg is None:
            for v in outs.values():
                if v.size in (SEG_H * SEG_W, 3 * SEG_H * SEG_W):
                    seg = v
                elif v.size == 3:
                    reg = v
        if seg is None:
            raise RuntimeError(
                'Khong tim thay dau ra seg trong engine. Cac dau ra: %s'
                % dict((k, v.shape) for k, v in outs.items()))
        return seg, (None if reg is None else reg.ravel())

    @staticmethod
    def _sorted_median(xs):
        """Trung vi chinh xac cua mang 1D da sap xep (flatnonzero) trong O(1)."""
        n = len(xs)
        mid = n >> 1
        if n & 1:
            return float(xs[mid])
        return 0.5 * (float(xs[mid - 1]) + float(xs[mid]))

    @staticmethod
    def _check_two_lines(white_s, center_s, road_s):
        """Kiem tra da nhan du 2 line trang ben trai + ben phai cua lan duong chua.

        Vector hoa bang NumPy thay vi vong lap Python 48 hang (~3-5x nhanh hon tren ARM).
        """
        if white_s is None:
            return False, False, False
        y0 = SEG_H // 4
        wb = white_s[y0:] > 0
        w_any = wb.any(axis=1)
        if not w_any.any():
            return False, False, False
        w_first = np.argmax(wb, axis=1).astype(np.float32)
        w_last = (SEG_W - 1 - np.argmax(wb[:, ::-1], axis=1)).astype(np.float32)
        ref_cx = np.where(
            (w_last - w_first) >= 14.0,
            0.5 * (w_first + w_last),
            SEG_W * 0.5,
        )
        xs_seg = np.arange(SEG_W, dtype=np.float32)
        if road_s is not None:
            rb = road_s[y0:] > 0
            r_cnt = rb.sum(axis=1)
            r_ok = r_cnt >= 4
            if r_ok.any():
                ref_cx[r_ok] = (rb[r_ok] * xs_seg).sum(axis=1) / r_cnt[r_ok]
        if center_s is not None:
            cb = center_s[y0:] > 0
            c_cnt = cb.sum(axis=1)
            c_ok = c_cnt >= 1
            if c_ok.any():
                ref_cx[c_ok] = (cb[c_ok] * xs_seg).sum(axis=1) / c_cnt[c_ok]
        left_rows = int(np.count_nonzero(w_any & (w_first < (ref_cx - 4.0))))
        right_rows = int(np.count_nonzero(w_any & (w_last > (ref_cx + 4.0))))
        has_left = left_rows >= 4
        has_right = right_rows >= 4
        return has_left, has_right, (has_left and has_right)

    @staticmethod
    def _road_fallback_mask(road_s, white_s):
        """Suy dai tam duong 64x128 tu mat duong (Ch0-Ch1) hoac giua 2 line trang (Ch1)."""
        clean = None
        if road_s is not None:
            if white_s is not None:
                clean = cv2.bitwise_and(road_s, cv2.bitwise_not(white_s))
            else:
                clean = road_s
        fb = np.zeros((SEG_H, SEG_W), np.uint8)
        med = CnnLaneDetector._sorted_median
        for y in range(SEG_H):
            cx = None
            if clean is not None:
                xs = np.flatnonzero(clean[y])
                if len(xs) >= 4:
                    cx = int(med(xs))
            if cx is None and white_s is not None:
                w_xs = np.flatnonzero(white_s[y])
                if len(w_xs) >= 2 and (int(w_xs[-1]) - int(w_xs[0])) >= 14:
                    cx = int(0.5 * (int(w_xs[0]) + int(w_xs[-1])))
            if cx is not None:
                x0 = max(0, cx - 2)
                x1 = min(SEG_W, cx + 3)
                fb[y, x0:x1] = 255
        return fb

    def _ego_road_bev(self, road_logits):
        road = cv2.warpPerspective((road_logits > 0.0).astype(np.uint8), self._M_seg,
                                   (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)
        _, road_ids = cv2.connectedComponents(road, connectivity=8)
        half = PROC_W // 2
        near = road_ids[-30:, half - 12:half + 12]
        road_counts = np.bincount(near.ravel())
        road_counts[0] = 0
        if road_counts.max() < 40:
            return None
        ego_road = (road_ids == road_counts.argmax()).astype(np.uint8)
        # Lap lo trong road truoc khi tim mep de loai vet trang do phan chieu.
        outlines = cv2.findContours(ego_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.drawContours(ego_road, outlines, -1, 1, cv2.FILLED)
        return ego_road

    def _white_road_edge_bev(self, white_logits, ego_road):
        white = cv2.warpPerspective((white_logits > 0.405465).astype(np.uint8), self._M_seg,
                                    (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)  # sigmoid > 0.60
        rim = cv2.morphologyEx(ego_road, cv2.MORPH_GRADIENT, self._edge_kernel) > 0
        rim &= self._bev_visible
        _, line_ids, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
        touching = np.bincount(line_ids[rim], minlength=len(stats))
        area = stats[:, cv2.CC_STAT_AREA]
        span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        supported = (area >= 25) & (span >= 24) & (touching >= 20) & (touching >= 0.18 * area)
        lines = np.flatnonzero(supported[1:]) + 1
        if len(lines) != 1:
            return None
        return (line_ids == lines[0]) & rim

    def _fit_inward_offset(self, ego_road, outside):
        # Duong cach deu o phia trong mat duong, ke ca khi mep cua gan nam ngang.
        distance = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 5)
        center = (ego_road > 0) & (np.abs(distance - self.single_line_offset_px) <= 4.0)
        _, center_ids, stats, _ = cv2.connectedComponentsWithStats(center.astype(np.uint8), connectivity=8)
        height = stats[:, cv2.CC_STAT_HEIGHT]
        bottom = stats[:, cv2.CC_STAT_TOP] + height
        supported = (stats[:, cv2.CC_STAT_AREA] >= 70) & (height >= 48) & (bottom >= 0.65 * PROC_H)
        half = PROC_W // 2
        best_id, best_dist = None, 75 ** 2
        for idx in np.flatnonzero(supported[1:]) + 1:
            ys, xs = np.where(center_ids == idx)
            dist = np.min((xs - half) ** 2 + (ys - PROC_H) ** 2)
            if dist < best_dist:
                best_id, best_dist = idx, dist
        if best_id is None:
            return None

        bev = (center_ids == best_id).astype(np.uint8) * 255
        ts, xs, npix = [], [], 0
        for t, band in zip(self._band_ts, np.array_split(bev[::-1], N_BANDS)):
            x, n = self._band_centre(band)
            if x is not None:
                ts.append(t)
                xs.append(x)
                npix += n
        if len(ts) < max(3, self.min_bands):
            return None
        co = self._fit_poly_fast(np.asarray(ts), np.asarray(xs))
        return co, ts[-1], len(ts), npix, bev

    def _fit_values(self, co, t_max):
        half = PROC_W / 2.0
        t_look = min(self.lookahead, t_max)
        cte = (co[2] - half) / half
        look = ((co[0] * t_look + co[1]) * t_look + co[2] - half) / half
        curvature = co[0] * t_max * t_max / half
        return tuple(float(np.clip(v, -1.0, 1.0)) for v in (cte, look, curvature))

    def _fit_recovery(self, ego_road, edge, reg_look):
        if edge is None:
            return None
        fit = self._fit_inward_offset(ego_road, edge)
        if fit is None:
            return None
        co, t_max = fit[:2]
        t_look = min(self.lookahead, t_max)
        x = int(round(float(np.polyval(co, t_look))))
        y = int(np.clip(round((1.0 - t_look) * PROC_H), 0, PROC_H - 1))
        if not (0 <= x < PROC_W and ego_road[y, x]):
            return None
        values = self._fit_values(co, t_max)
        if values[1] * reg_look <= 0.0 or abs(values[1] - reg_look) > self.reg_disagree:
            return None
        return fit, values

    def _recover_single_line(self, seg3, reg, raw, cte_disagree):
        if reg is None or len(reg) < 2:
            return None
        reg_look = float(reg[1])
        if not np.isfinite(reg_look) or abs(reg_look) < 0.20:
            return None
        look_disagree = (raw is not None and self.check_reg and raw[1] * reg_look <= 0.0
                         and abs(raw[1] - reg_look) > self.reg_disagree)
        if raw is not None and not (cte_disagree or look_disagree):
            return None
        ego_road = self._ego_road_bev(seg3[0])
        if ego_road is None:
            return None
        outside = self._bev_visible & (ego_road == 0)
        if np.count_nonzero(outside) >= 70:
            recovery = self._fit_recovery(ego_road, outside, reg_look)
            if recovery is not None:
                return recovery
        # Chi tham khao phan cua vach trang thuc su nam sat mep Ch0.
        white_edge = self._white_road_edge_bev(seg3[1], ego_road)
        return self._fit_recovery(ego_road, white_edge, reg_look)

    # ------------------------------------------------------------------ chay
    def process(self, frame_bgr, need_aux=True, compute_bev=False):
        """Suy dien va fit duong tam.

        - `need_aux=False`: khi Ch2 du tin cay, bo qua chuyen uint8 Ch0/Ch1
          va _check_two_lines tren duong nong.
        - `compute_bev=False`: chi chieu BEV khi can suy lai tu mat duong.
        """
        if self.engine is None:
            raise RuntimeError(
                'lane.cnn.engine dang de rong - chi dung duoc hau xu ly. '
                'Dat duong dan toi file .engine da build tren xe.')
        t0 = time.perf_counter()
        x = self.preprocess(frame_bgr)

        t1 = time.perf_counter()
        outs = self.engine.infer(x)
        self.last_infer_ms = (time.perf_counter() - t1) * 1000.0

        seg, reg = self._pick_outputs(outs)
        self.last_used_fallback = False
        seg3 = None
        # Nguong 0 tren logits == sigmoid > 0.5. Model co y khong co Sigmoid.
        if seg.size == 3 * SEG_H * SEG_W:
            seg3 = seg.reshape(3, SEG_H, SEG_W)
            if need_aux or compute_bev:
                seg_bin = (seg3 > 0).astype(np.uint8) * 255
                road_s = seg_bin[0]
                white_s = seg_bin[1]
                center_s = seg_bin[2]
                self.last_road_mask = road_s
                self.last_white_mask = white_s
                self.last_center_mask = center_s
                if need_aux:
                    hl, hr, two_ok = self._check_two_lines(white_s, center_s, road_s)
                    self.last_left_line = hl
                    self.last_right_line = hr
                    self.last_two_lines_ok = two_ok
                self.last_mask = center_s
                co, t_max, n_bands, npix = self._mask_to_fit_lut(center_s)
                if compute_bev:
                    self.last_bev = cv2.warpPerspective(
                        center_s, self._M_seg, (PROC_W, PROC_H),
                        flags=cv2.INTER_NEAREST)
                else:
                    self.last_bev = None
            else:
                co, t_max, n_bands, npix = self._mask_to_fit_lut(seg3[2])
                self.last_bev = None

            if co is None and self.road_fallback:
                if not (need_aux or compute_bev):
                    road_s = (seg3[0] > 0).astype(np.uint8) * 255
                    white_s = (seg3[1] > 0).astype(np.uint8) * 255
                    self.last_road_mask = road_s
                    self.last_white_mask = white_s
                fb_mask = self._road_fallback_mask(
                    self.last_road_mask, self.last_white_mask)
                co_fb, t_max_fb, n_bands_fb, npix_fb = self._mask_to_fit_lut(fb_mask)
                if co_fb is not None:
                    co, t_max, n_bands, npix = co_fb, t_max_fb, n_bands_fb, npix_fb
                    self.last_mask = fb_mask
                    if compute_bev:
                        self.last_bev = cv2.warpPerspective(
                            fb_mask, self._M_seg, (PROC_W, PROC_H),
                            flags=cv2.INTER_NEAREST)
                    self.last_used_fallback = True
                    self.n_fallback += 1
        else:
            seg1 = seg.reshape(SEG_H, SEG_W)
            self.last_road_mask = None
            self.last_white_mask = None
            self.last_center_mask = None
            self.last_left_line = False
            self.last_right_line = False
            self.last_two_lines_ok = False
            if compute_bev or need_aux:
                raw_mask_s = (seg1 > 0).astype(np.uint8) * 255
                self.last_mask = raw_mask_s
                co, t_max, n_bands, npix = self._mask_to_fit_lut(raw_mask_s)
                if compute_bev:
                    self.last_bev = cv2.warpPerspective(
                        raw_mask_s, self._M_seg, (PROC_W, PROC_H),
                        flags=cv2.INTER_NEAREST)
                else:
                    self.last_bev = None
            else:
                co, t_max, n_bands, npix = self._mask_to_fit_lut(seg1)
                self.last_bev = None

        raw = self._fit_values(co, t_max) if co is not None else None
        cte_disagree = (raw is not None and self.check_reg and reg is not None
                        and len(reg) >= 1 and abs(float(reg[0]) - raw[0]) > self.reg_disagree)
        if cte_disagree:
            self.n_disagree += 1

        if self.single_line_fallback and seg3 is not None:
            recovery = self._recover_single_line(seg3, reg, raw, cte_disagree)
            if recovery is not None:
                (co, _, n_bands, npix, fb_bev), raw = recovery
                cte_disagree = False
                if not self.last_used_fallback:
                    self.n_fallback += 1
                self.last_used_fallback = True
                if need_aux or compute_bev:
                    self.last_mask = cv2.warpPerspective(
                        fb_bev, self._M_seg, (SEG_W, SEG_H),
                        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                if compute_bev:
                    self.last_bev = fb_bev

        if raw is None or cte_disagree:
            self.last_total_ms = (time.perf_counter() - t0) * 1000.0
            return LaneResult(False, self._cte, self._curv, npix, self.last_bev,
                              cte_lookahead=self._look, n_bands=n_bands)

        raw_cte, raw_look, raw_curv = raw
        if not self._init:
            self._cte, self._look, self._curv = raw_cte, raw_look, raw_curv
            self._init = True
        else:
            a = self.alpha
            self._cte = a * raw_cte + (1.0 - a) * self._cte
            self._look = a * raw_look + (1.0 - a) * self._look
            self._curv = a * raw_curv + (1.0 - a) * self._curv

        self.last_total_ms = (time.perf_counter() - t0) * 1000.0
        return LaneResult(True, self._cte, self._curv, npix, self.last_bev,
                          cte_lookahead=self._look, n_bands=n_bands, fit=co)

    def debug_process(self, frame_bgr):
        res = self.process(frame_bgr, need_aux=True, compute_bev=True)
        return {
            'small': cv2.resize(frame_bgr, (PROC_W, PROC_H)),
            'binary': self.last_mask,
            'masked': self.last_mask,
            'warped': self.last_bev,
            'roi_y': (self.roi_y0, PROC_H),
            'result': res,
            'infer_ms': self.last_infer_ms,
            'total_ms': self.last_total_ms,
        }
