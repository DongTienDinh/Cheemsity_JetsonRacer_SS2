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

import ctypes
import ctypes.util
import os
import subprocess
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


_TRTEXEC_CANDIDATES = (
    '/usr/src/tensorrt/bin/trtexec',
    'trtexec',
)
_CUDART_CANDIDATES = (
    '/usr/local/cuda/lib64/libcudart.so',
    '/usr/local/cuda-10.2/lib64/libcudart.so',
    'libcudart.so',
    'libcudart.so.10.2',
)


def _find_trtexec():
    for path in _TRTEXEC_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _auto_build_trt_engine(onnx_path, engine_path, workspace_mb=256):
    """Tu dong build file .engine FP16 tu .onnx bang trtexec co san tren Jetson."""
    trtexec = _find_trtexec()
    if not trtexec or not os.path.isfile(onnx_path):
        return False
    out_dir = os.path.dirname(os.path.abspath(engine_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    cmd = [
        trtexec,
        '--onnx=%s' % onnx_path,
        '--fp16',
        '--workspace=%d' % int(workspace_mb),
        '--saveEngine=%s' % engine_path,
    ]
    print('[TRT] Khong thay/khac ban %s -> tu dong build FP16 tu %s (mat ~1-2 phut)...'
          % (engine_path, onnx_path))
    rc = subprocess.call(cmd)
    return rc == 0 and os.path.isfile(engine_path)


class _CudartRuntime(object):
    """Boc CUDA Runtime C API (`libcudart.so`) qua ctypes — KHONG can cai `pycuda`.

    `libcudart.so` co san 100% tren moi Jetson Nano JetPack -> sang Jetson moi
    chi can pull code la chay duoc TensorRT FP16 ngay, khong can bien dich pycuda.
    """

    HTOD = 1
    DTOH = 2

    def __init__(self, lib=None):
        if lib is None:
            lib = self._load_lib()
        self._lib = lib
        self._lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self._lib.cudaSetDevice.restype = ctypes.c_int
        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaMalloc.restype = ctypes.c_int
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int
        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.cudaStreamCreate.restype = ctypes.c_int
        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int
        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int

    @staticmethod
    def _load_lib():
        found = ctypes.util.find_library('cudart')
        cands = list(_CUDART_CANDIDATES) + ([found] if found else [])
        last_err = None
        for name in cands:
            try:
                return ctypes.CDLL(name)
            except OSError as exc:
                last_err = exc
        raise ImportError('Khong nap duoc libcudart.so: %s' % last_err)

    def _check(self, err, fn_name):
        if int(err) != 0:
            raise RuntimeError('CUDA Runtime %s that bai (ma loi %d)' % (fn_name, int(err)))

    def set_device(self, dev=0):
        self._check(self._lib.cudaSetDevice(int(dev)), 'cudaSetDevice')

    def init_context(self, dev=0):
        """Chon GPU va ep khoi tao primary CUDA context bang `cudaFree(0)`."""
        self.set_device(dev)
        self._check(self._lib.cudaFree(ctypes.c_void_p(0)), 'cudaFree(0)')

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), int(nbytes)), 'cudaMalloc')
        return int(ptr.value)

    def free(self, ptr_int):
        if ptr_int:
            self._lib.cudaFree(ctypes.c_void_p(int(ptr_int)))

    def create_stream(self):
        stream = ctypes.c_void_p()
        self._check(self._lib.cudaStreamCreate(ctypes.byref(stream)), 'cudaStreamCreate')
        return int(stream.value)

    def destroy_stream(self, stream_int):
        if stream_int:
            self._lib.cudaStreamDestroy(ctypes.c_void_p(int(stream_int)))

    def sync_stream(self, stream_int):
        self._check(
            self._lib.cudaStreamSynchronize(ctypes.c_void_p(int(stream_int))),
            'cudaStreamSynchronize')

    def memcpy_htod_async(self, dst_dev_int, host_arr, stream_int):
        self._check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(int(dst_dev_int)),
                ctypes.c_void_p(int(host_arr.ctypes.data)),
                int(host_arr.nbytes),
                self.HTOD,
                ctypes.c_void_p(int(stream_int)),
            ),
            'cudaMemcpyAsync(HtoD)')

    def memcpy_dtoh_async(self, host_arr, src_dev_int, stream_int):
        self._check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(int(host_arr.ctypes.data)),
                ctypes.c_void_p(int(src_dev_int)),
                int(host_arr.nbytes),
                self.DTOH,
                ctypes.c_void_p(int(stream_int)),
            ),
            'cudaMemcpyAsync(DtoH)')


class TensorRTEngine(object):
    """Boc engine TensorRT. Ho tro ca API binding (TRT 7/8.0) va tensor (TRT >=8.5).

    Tu dong dung `pycuda` neu co, hoac chuyen sang `libcudart.so` (ctypes co san trong
    JetPack) khi Jetson chua cai `pycuda` -> pull code la chay duoc ngay!
    """

    def __init__(self, engine_path):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                '%s.\n'
                'lane.mode=cnn CHI chay duoc tren Jetson (tensorrt co san theo JetPack).\n'
                'Tren laptop: dat lane.cnn.backend=onnx de chay bang ONNXRuntime,\n'
                'hoac lane.cnn.engine rong de chi dung hau xu ly.'
                % exc)

        cuda = None
        cudart = None
        try:
            import pycuda.driver as cuda
            # CO Y KHONG dung `pycuda.autoinit` (tranh loi context giua cac thread).
            cuda.init()
        except ImportError as pycuda_exc:
            try:
                cudart = _CudartRuntime()
                cudart.init_context(0)
                print('[TRT] Khong co pycuda -> dung truc tiep libcudart.so (ctypes, khong can cai them).')
            except Exception as rt_exc:
                raise ImportError(
                    'Khong co pycuda (%s) va khong nap duoc libcudart.so (%s).\n'
                    'Tren laptop: dat lane.cnn.backend=onnx hoac de trong lane.cnn.engine.'
                    % (pycuda_exc, rt_exc))

        onnx_sibling = os.path.splitext(engine_path)[0] + '.onnx'
        if not os.path.isfile(engine_path):
            if not _auto_build_trt_engine(onnx_sibling, engine_path):
                raise IOError(
                    'Khong thay engine: %s\n'
                    'Build TREN CHINH XE:\n'
                    '  python3 tools/setup_jetson_models.py --model road_fast'
                    % engine_path)

        import threading
        self._cuda = cuda
        self._cudart = cudart
        self._ctx = cuda.Device(0).retain_primary_context() if cuda is not None else None
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
        if self._ctx is not None:
            self._ctx.push()
            try:
                try:
                    self._build(trt, engine_path, onnx_sibling, logger)
                finally:
                    self._ctx.pop()
            except BaseException:
                self.close()
                raise
        else:
            try:
                self._cudart.set_device(0)
                self._build(trt, engine_path, onnx_sibling, logger)
            except BaseException:
                self.close()
                raise

    def _build(self, trt, engine_path, onnx_sibling, logger):
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as fh:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None and _auto_build_trt_engine(onnx_sibling, engine_path):
            with open(engine_path, 'rb') as fh:
                self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(
                'TensorRT khong doc duoc %s. Gan nhu chac chan engine nay build '
                'tren may/phien ban khac - chay `python3 tools/setup_jetson_models.py --force`.'
                % engine_path)
        self.context = self.engine.create_execution_context()
        self._new_api = hasattr(self.engine, 'num_io_tensors')

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
        if self._cuda is not None:
            self.stream = self._cuda.Stream()
        else:
            self.stream = self._cudart.create_stream()
        self.bindings = [int(d) for d in self.dev]

    def _alloc(self, name, shape, dtype, is_input):
        idx = len(self.host)
        n_elems = int(np.prod(shape))
        if self._cuda is not None:
            host = self._cuda.pagelocked_empty(n_elems, dtype)
            dev = self._cuda.mem_alloc(host.nbytes)
        else:
            host = np.empty(n_elems, dtype=dtype)
            dev = self._cudart.malloc(host.nbytes)
        self.host.append(host)
        self.dev.append(dev)
        self.names.append(name)
        self.shapes.append(shape)
        (self.in_idx if is_input else self.out_idx).append(idx)

    def infer(self, x):
        """x: float32 (1,3,H,W) lien tuc -> dict {ten: mang}.

        An toan da luong cho ca PyCUDA lan libcudart.so (ctypes).
        """
        with self._lock:
            if self._closed or (self._ctx is None and self._cudart is None):
                raise RuntimeError('TensorRTEngine da dong')
            if self._ctx is not None:
                self._ctx.push()
                try:
                    return self._infer_locked(x)
                finally:
                    self._ctx.pop()
            self._cudart.set_device(0)
            return self._infer_locked(x)

    def close(self):
        """Giai phong buffer GPU/TRT an toan cho ca PyCUDA lan libcudart."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            ctx = self._ctx
            self._ctx = None
            if ctx is not None:
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
                return
            if self._cudart is not None:
                try:
                    self._cudart.set_device(0)
                except Exception:
                    pass
                self.context = None
                self.engine = None
                for ptr in self.dev:
                    try:
                        self._cudart.free(ptr)
                    except Exception:
                        pass
                if self.stream:
                    try:
                        self._cudart.destroy_stream(self.stream)
                    except Exception:
                        pass
                self.stream = None
                self.dev = []
                self.host = []
                self.bindings = []
                self._cudart = None

    def _infer_locked(self, x):
        i0 = self.in_idx[0]
        np.copyto(self.host[i0], x.ravel())
        if self._cuda is not None:
            cuda = self._cuda
            cuda.memcpy_htod_async(self.dev[i0], self.host[i0], self.stream)
            stream_handle = self.stream.handle
            if self._new_api:
                for nm, d in zip(self.names, self.dev):
                    self.context.set_tensor_address(nm, int(d))
                self.context.execute_async_v3(stream_handle=stream_handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings,
                                              stream_handle=stream_handle)
            for i in self.out_idx:
                cuda.memcpy_dtoh_async(self.host[i], self.dev[i], self.stream)
            self.stream.synchronize()
        else:
            rt = self._cudart
            rt.memcpy_htod_async(self.dev[i0], self.host[i0], self.stream)
            stream_handle = int(self.stream)
            if self._new_api:
                for nm, d in zip(self.names, self.dev):
                    self.context.set_tensor_address(nm, int(d))
                self.context.execute_async_v3(stream_handle=stream_handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings,
                                              stream_handle=stream_handle)
            for i in self.out_idx:
                rt.memcpy_dtoh_async(self.host[i], self.dev[i], self.stream)
            rt.sync_stream(self.stream)
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
        self.last_road_logits = None # Option B Ch0 float32 logits (64x128) cho Pure Pursuit
        self.last_used_fallback = False
        self.last_road_turn = 0
        self.last_edge_guard = 0.0
        self._ego_on_center = False
        self.road_guided = False
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
        # Nua be rong mat duong den do thuc te trong BEV 320x240 la ~70-78 px
        # (mac dinh cu 108.0 day duong tam lech sau vao trong cua +34 px!).
        self.single_line_offset_px = float(c('lane.cnn.single_line_offset_px', 74.0))
        # Khoang cach an toan toi thieu tu duong tam Ch2 toi mep vach trang (BEV px).
        self.center_edge_margin_px = float(c('lane.cnn.center_edge_margin_px', 32.0))
        # Dem an toan chong cham vach trang o vung gan dau xe (BEV px & he so day).
        self.edge_guard_px = float(c('lane.cnn.edge_guard_px', 42.0))
        self.edge_guard_gain = float(c('lane.cnn.edge_guard_gain', 0.45))
        self.road_only = bool(c('lane.cnn.road_only', False))

    def _reset_state(self):
        self._cte = 0.0
        self._look = 0.0
        self._curv = 0.0
        self._init = False
        self._road_start_streak = 0

    @property
    def road_start_ready(self):
        return self._road_start_streak >= 5

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

    def _ego_road_bev(self, road_logits):
        road = cv2.warpPerspective((road_logits > 0.0).astype(np.uint8), self._M_seg,
                                   (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)
        _, road_ids = cv2.connectedComponents(road, connectivity=8)
        half = PROC_W // 2
        near = road_ids[-30:, half - 12:half + 12]
        road_counts = np.bincount(near.ravel())
        road_counts[0] = 0
        if road_counts.max() >= 40:
            self._ego_on_center = True
            best_id = int(road_counts.argmax())
        else:
            # Xe da lech sat/de len vach trang: mo rong tim mat duong den o dai
            # day anh de van biet phia nao la duong va keo xe quay lai.
            self._ego_on_center = False
            wide = road_ids[-48:, :]
            w_counts = np.bincount(wide.ravel())
            w_counts[0] = 0
            if w_counts.max() < 120:
                return None
            best_id, best_dist = None, 1e9
            for cid in np.flatnonzero(w_counts >= 120):
                xs = np.where(wide == cid)[1]
                d = abs(float(xs.mean()) - half)
                if d < best_dist:
                    best_id, best_dist = int(cid), d
            if best_id is None:
                return None
        ego_road = (road_ids == best_id).astype(np.uint8)
        # Lap lo trong road truoc khi tim mep de loai vet trang do phan chieu / vach START.
        outlines = cv2.findContours(ego_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.drawContours(ego_road, outlines, -1, 1, cv2.FILLED)
        return ego_road

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

    def _fit_midpoints(self, ego_road, preview):
        """Lay trung diem tren nhung hang co du hai mep road thuc su thay duoc."""
        road = ego_road > 0
        width = road.sum(axis=1)
        left = road.argmax(axis=1)
        right = PROC_W - 1 - road[:, ::-1].argmax(axis=1)
        rows = np.arange(PROC_H)
        inside_view = (left >= 3) & (right < PROC_W - 3)
        inside_view &= (self._bev_visible[rows, np.maximum(0, left - 3)] &
                        self._bev_visible[rows, np.minimum(PROC_W - 1, right + 3)])
        # Mot hang giao ca hai nhanh cua chu U co the co nhieu khoang road.
        valid = inside_view & (width >= 26) & (width == right - left + 1)
        band_h = PROC_H // N_BANDS
        ts, xs, npix = [], [], 0
        for b, t in enumerate(self._band_ts):
            hi = PROC_H - b * band_h
            indices = np.flatnonzero(valid[hi - band_h:hi]) + hi - band_h
            if len(indices) >= 9:
                ts.append(t)
                xs.append(float(np.median((left[indices] + right[indices]) * 0.5)))
                npix += int(len(indices))
        if len(ts) < max(3, self.min_bands) or ts[0] > 0.45:
            return None
        co = self._fit_poly_fast(np.asarray(ts), np.asarray(xs))
        bev = None
        if preview:
            bev = np.zeros_like(ego_road)
            points = np.column_stack((xs, (1.0 - np.asarray(ts)) * PROC_H))
            cv2.polylines(bev, [np.rint(points).astype(np.int32)], False, 255, 3)
        return co, ts[-1], len(ts), npix, bev

    def _road_fit_values(self, ego_road, fit):
        if fit is None:
            return None
        co, t_max = fit[:2]
        t_look = min(self.lookahead, t_max)
        x = int(round(float(np.polyval(co, t_look))))
        y = int(np.clip(round((1.0 - t_look) * PROC_H), 0, PROC_H - 1))
        if 0 <= x < PROC_W and ego_road[y, x]:
            return self._fit_values(co, t_max)
        return None

    def _fit_road_guide(self, ego_road, preview):
        if ego_road is None:
            return None, None, False
        fit = self._fit_midpoints(ego_road, preview)
        raw = self._road_fit_values(ego_road, fit)
        if raw is not None:
            return fit, raw, True
        outside = self._bev_visible & (ego_road == 0)
        if np.count_nonzero(outside) < 70:
            return None, None, False
        fit = self._fit_inward_offset(ego_road, outside)
        raw = self._road_fit_values(ego_road, fit)
        if raw is not None:
            # Mot mep o xa cho huong ngam, khong du du lieu tinh CTE sat dau xe.
            raw = (0.0, raw[1], raw[2])
        return fit, raw, False

    def _road_turn_direction(self, fit, raw):
        if raw is None:
            return 0
        co, t_max = fit[:2]
        t_far = min(self.lookahead, t_max)
        t_near = max(0.0, t_far - 0.35)
        half = PROC_W / 2.0
        lateral = (np.polyval(co, t_far) - np.polyval(co, t_near)) / half
        bend = abs(co[0] * t_far * t_far / half)
        # Lech sang mot ben chua du: duong con phai doi huong va co do cong.
        if abs(raw[1]) < 0.22 or bend < 0.05 or abs(lateral) < 0.12 or lateral * raw[1] <= 0.0:
            return 0
        return 1 if lateral > 0.0 else -1

    def _point_near_outside(self, ego_road, outside, x, y, margin_px):
        if x < 0 or x >= PROC_W or y < 0 or y >= PROC_H:
            return True
        if not ego_road[y, x]:
            return True
        r = int(round(margin_px))
        if r <= 0 or outside is None:
            return False
        y0, y1 = max(0, y - r), min(PROC_H, y + r + 1)
        x0, x1 = max(0, x - r), min(PROC_W, x + r + 1)
        patch = outside[y0:y1, x0:x1]
        if not patch.any():
            return False
        py, px = np.nonzero(patch)
        dy = (py + y0) - y
        dx = (px + x0) - x
        return bool(np.min(dx * dx + dy * dy) < r * r)

    def _center_edge_unsafe(self, ego_road, outside, co, t_max):
        """True neu duong tam Ch2 chay ra ngoai mat duong hoac ap sat vach trang."""
        if ego_road is None or co is None:
            return False
        t_look = min(self.lookahead, t_max)
        x_look = int(round(float(np.polyval(co, t_look))))
        y_look = int(np.clip(round((1.0 - t_look) * PROC_H), 0, PROC_H - 1))
        if self._point_near_outside(ego_road, outside, x_look, y_look, self.center_edge_margin_px):
            return True
        t_near = min(0.25, t_max)
        x_near = int(round(float(np.polyval(co, t_near))))
        y_near = int(np.clip(round((1.0 - t_near) * PROC_H), 0, PROC_H - 1))
        return not (0 <= x_near < PROC_W and ego_road[y_near, x_near])

    def _apply_edge_guard(self, ego_road, outside, raw):
        """Dem ao day xe khoi vach trang khi dau xe ap sat bien mat duong den."""
        self.last_edge_guard = 0.0
        if (ego_road is None or raw is None or
                self.edge_guard_px <= 0.0 or self.edge_guard_gain <= 0.0):
            return raw
        half = PROC_W // 2
        near_road = ego_road[168:224, :]
        # Truong hop khan: mui xe da cham/vuot ra ngoai mat duong den -> keo manh ve phia mat duong
        center_fill = float(np.count_nonzero(near_road[:, half - 8:half + 8])) / float(56 * 16)
        if center_fill < 0.25:
            xs_road = np.where(near_road > 0)[1]
            if len(xs_road) < 80:
                return raw
            direction = 1.0 if float(np.median(xs_road)) >= half else -1.0
            push = direction * self.edge_guard_gain
            self.last_edge_guard = push
            new_cte = float(np.clip(raw[0] + push, -1.0, 1.0))
            new_look = raw[1]
            if new_look * direction < 0.25:
                new_look = float(np.clip(new_look + 0.6 * push, -1.0, 1.0))
            return (new_cte, new_look, raw[2])

        if outside is None:
            return raw
        out_near = outside[168:224, :]
        left_out = out_near[:, :half]
        right_out = out_near[:, half:]
        has_l = left_out.any(axis=1)
        has_r = right_out.any(axis=1)
        d_left = float(np.median(left_out[has_l, ::-1].argmax(axis=1) + 1)) if has_l.any() else 999.0
        d_right = float(np.median(right_out[has_r].argmax(axis=1) + 1)) if has_r.any() else 999.0

        push = 0.0
        if d_right < self.edge_guard_px and (d_right + 10.0) < d_left:
            push = -self.edge_guard_gain * (self.edge_guard_px - d_right) / self.edge_guard_px
        elif d_left < self.edge_guard_px and (d_left + 10.0) < d_right:
            push = self.edge_guard_gain * (self.edge_guard_px - d_left) / self.edge_guard_px
        if push == 0.0:
            return raw

        self.last_edge_guard = float(push)
        new_cte = float(np.clip(raw[0] + push, -1.0, 1.0))
        new_look = raw[1]
        if raw[1] * push <= 0.0 or abs(raw[1]) < abs(push):
            new_look = float(np.clip(raw[1] + 0.5 * push, -1.0, 1.0))
        return (new_cte, new_look, raw[2])

    def _cte_disagrees(self, raw, reg):
        return (raw is not None and self.check_reg and reg is not None and len(reg) >= 1
                and abs(float(reg[0]) - raw[0]) > self.reg_disagree)

    def _process_road_guided(self, seg3, reg, preview, t0):
        road_logits, center_logits = seg3[0], seg3[2]
        self.last_road_logits = road_logits
        self.last_road_mask = self.last_mask = self.last_bev = None
        self.last_edge_guard = 0.0
        if getattr(self, 'road_only', False):
            # Che do chi dung Ch0 (cho RoadPurePursuitController): bo qua toan bo
            # hau xu ly Ch2/fallback de toi uu FPS tren Jetson Nano.
            if preview:
                self.last_road_mask = (road_logits > 0.0).astype(np.uint8) * 255
            self.last_total_ms = (time.perf_counter() - t0) * 1000.0
            return LaneResult(True, 0.0, 0.0, int(np.count_nonzero(road_logits > 0.0)),
                              None, cte_lookahead=0.0, n_bands=10)
        co, t_max, n_bands, npix = self._mask_to_fit_lut(center_logits)
        raw = self._fit_values(co, t_max) if co is not None else None
        if self._cte_disagrees(raw, reg):
            self.n_disagree += 1
            raw = None

        ego_road = self._ego_road_bev(road_logits)
        outside = (self._bev_visible & (ego_road == 0)) if ego_road is not None else None
        # Vung Ch0 phai nam ngay duoi mui xe va keo dai den dai thu ba phia truoc.
        band_h = PROC_H // N_BANDS
        road_seen = (ego_road is not None and self._ego_on_center and
                     np.count_nonzero(ego_road[-3 * band_h:-2 * band_h]) >= 10 * band_h)
        self._road_start_streak = self._road_start_streak + 1 if road_seen else 0

        guide, guide_raw, two_edges = self._fit_road_guide(ego_road, preview)
        self.last_road_turn = self._road_turn_direction(guide, guide_raw)
        wrong_way = (self.last_road_turn and raw is not None and
                     raw[1] * self.last_road_turn <= 0.0 and
                     abs(raw[1] - guide_raw[1]) >= 0.25)
        edge_unsafe = (raw is not None and guide_raw is not None and
                       self._center_edge_unsafe(ego_road, outside, co, t_max))
        two_edge_drift = (two_edges and raw is not None and guide_raw is not None and
                          (abs(raw[0] - guide_raw[0]) > 0.28 or abs(raw[1] - guide_raw[1]) > 0.30))
        if wrong_way or edge_unsafe or two_edge_drift or (raw is None and (self.last_road_turn or two_edges or guide_raw is not None)):
            if guide is not None and guide_raw is not None:
                co, _, n_bands, npix, self.last_bev = guide
                # Khong de feed-forward quay nguoc huong mep xanh da xac nhan.
                curv = 0.0 if guide_raw[2] * self.last_road_turn < 0.0 else guide_raw[2]
                raw = (guide_raw[0], guide_raw[1], curv)
                self.last_used_fallback = True
                self.n_fallback += 1

        raw = self._apply_edge_guard(ego_road, outside, raw)

        if preview:
            self.last_road_mask = (road_logits > 0.0).astype(np.uint8) * 255
            if self.last_used_fallback:
                self.last_mask = cv2.warpPerspective(
                    self.last_bev, self._M_seg, (SEG_W, SEG_H),
                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            elif raw is not None:
                self.last_mask = (center_logits > 0.0).astype(np.uint8) * 255
                self.last_bev = cv2.warpPerspective(
                    self.last_mask, self._M_seg, (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)
        else:
            self.last_bev = None
        return self._finish_frame(raw, co, n_bands, npix, t0)

    # ------------------------------------------------------------------ chay
    def process(self, frame_bgr, need_aux=True, compute_bev=False):
        """Suy tam tu Ch0 va Ch2 cua Option B, hoac mask tam 1 kenh cu."""
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
        self.last_road_turn = 0
        if getattr(self, 'road_only', False):
            self.road_guided = True
            if seg.size == 3 * SEG_H * SEG_W:
                road_logits = seg.reshape(3, SEG_H, SEG_W)[0]
            else:
                road_logits = seg.reshape(SEG_H, SEG_W)
            self.last_road_logits = road_logits
            self.last_mask = None
            self.last_bev = None
            self.last_road_mask = (
                (road_logits > 0.0).astype(np.uint8) * 255 if compute_bev else None)
            self.last_total_ms = (time.perf_counter() - t0) * 1000.0
            return LaneResult(True, 0.0, 0.0, int(np.count_nonzero(road_logits > 0.0)),
                              None, cte_lookahead=0.0, n_bands=10)
        self.road_guided = seg.size == 3 * SEG_H * SEG_W
        if self.road_guided:
            return self._process_road_guided(seg.reshape(3, SEG_H, SEG_W), reg, compute_bev, t0)

        self._road_start_streak = 0
        self.last_road_mask = None
        self.last_bev = None
        center = seg.reshape(SEG_H, SEG_W)
        if compute_bev or need_aux:
            center = (center > 0).astype(np.uint8) * 255
            self.last_mask = center
        if compute_bev:
            self.last_bev = cv2.warpPerspective(
                center, self._M_seg, (PROC_W, PROC_H), flags=cv2.INTER_NEAREST)
        co, t_max, n_bands, npix = self._mask_to_fit_lut(center)
        raw = self._fit_values(co, t_max) if co is not None else None
        if self._cte_disagrees(raw, reg):
            self.n_disagree += 1
            raw = None
        return self._finish_frame(raw, co, n_bands, npix, t0)

    def _finish_frame(self, raw, co, n_bands, npix, t0):
        if raw is None:
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
