# -*- coding: utf-8 -*-
"""Smoke test chay duoc tren laptop, khong can xe, khong can model.

    python -m pytest tests/ -q      (hoac)      python tests/test_smoke.py

Muc dich: bat loi tich hop TRUOC khi mang code len xe. Gio chay tren xe la tai
nguyen khan hiem nhat (5 xe / 10 doi) - khong duoc dung de debug loi cu phap.
"""

import os
import csv
import io
import shutil
import sys
import tempfile
import threading
import time
import types

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from jetracer_baseline import fsm as fsm_mod                     # noqa: E402
from jetracer_baseline.camera import (                          # noqa: E402
    CSISource, LatestFrameGrabber, ShadedSource, SyntheticSource,
    _build_raw_source, build_source, gstreamer_pipeline,
    shading_applied_at_source)
from jetracer_baseline.config import load_config                 # noqa: E402
from jetracer_baseline.control.corner import (                   # noqa: E402
    CORNER, STRAIGHT, CornerController)
from jetracer_baseline.control.pid import PID                    # noqa: E402
from jetracer_baseline.control import driver as driver_mod       # noqa: E402
from jetracer_baseline.perception.lane import LaneDetector       # noqa: E402
from jetracer_baseline.perception.shading import (               # noqa: E402
    ShadingCorrector, fit_coefficients, measure_ratio_profile)
from jetracer_baseline.perception.signs import SignTracker       # noqa: E402
from jetracer_baseline.perception.stopline import StoplineDetector  # noqa: E402
from jetracer_baseline.pipeline import Runner                    # noqa: E402
from jetracer_baseline.tuning_ui import (                        # noqa: E402
    MANUAL_PARAMS, MODE_AUTO, MODE_MANUAL, MODE_STOP, ControllerShaper,
    LaneTuningEngine, LaneTuningUI, ProcessingFpsMeter, RollingStats,
    _dict_diff)
from jetracer_baseline.manual_collection import (                # noqa: E402
    CSV_FIELDS, DatasetSessionWriter, ManualDriveCollector, apply_deadzone,
    shape_steering, shape_throttle, slew_towards)

CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')


def _cfg(**overrides):
    cfg = load_config(CONFIG)
    for key, value in overrides.items():
        cfg.set(key.replace('__', '.'), value)
    return cfg


def test_config_loads():
    cfg = load_config(CONFIG)
    assert cfg.get('control.v_max') > 0
    assert cfg.get('pipeline.control_hz') >= 20
    assert cfg.get('control.driver.implementation') == 'waveshare_single_pca'
    assert cfg.get('control.driver.i2c_address') == 0x40
    assert cfg.get('control.driver.steering_channel') == 0
    assert cfg.get('control.driver.throttle_channel') == 1
    assert cfg.get('control.driver.command_timeout_s') > 0
    assert cfg.get('control.driver.pwm_frequency') is None
    assert abs(cfg.get('control.driver.steering_gain')) <= 0.65
    assert cfg.get('control.driver.pulse_width_range') == [750, 2250]
    assert cfg.get('control.driver.steering_output_min') == -0.40
    assert cfg.get('control.driver.steering_output_max') == 0.40
    assert 0 < cfg.get('manual.min_throttle') < cfg.get('manual.max_throttle')
    assert cfg.get('manual.test_throttle') >= cfg.get('manual.min_throttle')
    assert 0 < cfg.get('manual.max_steering') <= 0.60
    assert cfg.get('manual.steering_slew_rate') > 0
    assert 10 <= cfg.get('manual.video_fps') <= cfg.get('camera.fps')
    assert cfg.get('control.steer_max') <= 0.60
    assert cfg.get('fsm.turn_steer') <= cfg.get('control.steer_max')
    # Doc override phai ghi de dung, khong lam mat khoa khac
    cfg2 = load_config(CONFIG, [os.path.join(ROOT, 'configs', 'fast.yaml')])
    # Profile nhanh phai nhanh hon o DOAN THANG. Khong kiem v_max: v_max chi la
    # tran an toan, khong phai ga muc tieu (xem control/corner.py).
    assert cfg2.get('control.v_straight') > cfg.get('control.v_straight')
    assert cfg2.get('control.v_max') >= cfg2.get('control.v_straight')
    # Va KHONG duoc nhanh hon trong cua: cua sa ban rat hep.
    assert cfg2.get('control.v_corner') <= cfg2.get('control.v_straight')
    # Vao cua nhanh hon thi phai nhan ra cua som hon
    assert cfg2.get('control.curve_enter') <= cfg.get('control.curve_enter')
    assert cfg2.get('lane.roi_top') == cfg.get('lane.roi_top')


def test_pid_sign_and_limit():
    pid = PID(kp=1.0, ki=0.0, kd=0.0, out_limit=0.5)
    assert pid.step(0.3, 0.033) > 0        # lech phai -> lai phai
    pid.reset()
    assert pid.step(-0.3, 0.033) < 0       # lech trai -> lai trai
    pid.reset()
    assert abs(pid.step(10.0, 0.033)) <= 0.5   # ton trong out_limit


def test_manual_collection_deadzone_and_writer():
    assert apply_deadzone(0.03, 0.06) == 0.0
    assert apply_deadzone(-0.03, 0.06) == 0.0
    assert 0.45 < apply_deadzone(0.50, 0.06) < 0.50
    assert shape_throttle(0.03, 0.06, 0.12, 0.30) == 0.0
    assert abs(shape_throttle(1.0, 0.06, 0.12, 0.30) - 0.30) < 1e-9
    assert -0.22 < shape_throttle(-0.50, 0.06, 0.12, 0.30) < -0.19
    assert shape_steering(0.03, 0.06, 0.85, 0.35) == 0.0
    assert abs(shape_steering(1.0, 0.06, 0.85, 0.35) - 0.85) < 1e-9
    assert 0.25 < shape_steering(0.50, 0.06, 0.85, 0.35) < 0.32
    assert abs(slew_towards(0.0, 1.0, 2.0, 0.1) - 0.2) < 1e-9

    temp_root = tempfile.mkdtemp(prefix='jetracer_collect_')
    writer = None
    try:
        writer = DatasetSessionWriter(
            temp_root, 'test sang', metadata={'purpose': 'smoke'})
        frame = np.full((24, 32, 3), 80, dtype=np.uint8)
        path = writer.write(
            frame, camera_frame_id=7,
            timestamp_unix=1000.25, timestamp_monotonic=20.5,
            steering_raw=0.4, throttle_raw=-0.6,
            steering_cmd=0.36, throttle_cmd=0.12,
            deadman_pressed=True, controller_connected=True)
        writer.close()
        assert os.path.exists(path)
        assert os.path.exists(writer.metadata_path)
        with open(writer.csv_path, 'r') as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]['camera_frame_id'] == '7'
        assert rows[0]['deadman_pressed'] == '1'
        assert rows[0]['image_file'] == 'images/frame_000000.jpg'
    finally:
        if writer is not None:
            writer.close()
        shutil.rmtree(temp_root)


def test_csi_open_without_frame_falls_back_to_usb():
    """Bat loi Jetson hay gap: isOpened=True nhung nvargus khong tra frame."""
    import jetracer_baseline.camera as camera_mod

    frame = np.full((12, 16, 3), 123, dtype=np.uint8)

    class FakeCapture(object):
        def __init__(self, opened, reads):
            self.opened = opened
            self.reads = list(reads)
            self.released = False

        def isOpened(self):
            return self.opened

        def read(self):
            if not self.reads:
                return False, None
            return self.reads.pop(0)

        def set(self, _key, _value):
            return True

        def release(self):
            self.released = True

    csi_cap = FakeCapture(True, [(False, None), (False, None)])
    usb_cap = FakeCapture(True, [(True, frame)])
    original = camera_mod.cv2.VideoCapture

    def fake_video_capture(source, *args):
        if isinstance(source, str):
            return csi_cap
        return usb_cap

    camera_mod.cv2.VideoCapture = fake_video_capture
    source = None
    try:
        source = CSISource(
            640, 480, 30, allow_usb_fallback=True,
            startup_attempts=2, startup_delay_s=0.0,
            csi_open_attempts=1, csi_retry_delay_s=0.0)
        ok, result = source.read()
        assert ok
        assert result.shape == frame.shape
        assert source.backend == 'usb-v4l2-index-0'
        assert csi_cap.released
        assert source.startup_notes
    finally:
        if source is not None:
            source.release()
        camera_mod.cv2.VideoCapture = original


def test_csi_open_without_frame_reopens_argus_before_usb():
    """CaptureSession loi tam thoi: release va mo lai CSI truoc khi fallback."""
    import jetracer_baseline.camera as camera_mod

    frame = np.full((12, 16, 3), 77, dtype=np.uint8)

    class FakeCapture(object):
        def __init__(self, reads):
            self.reads = list(reads)
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if not self.reads:
                return False, None
            return self.reads.pop(0)

        def release(self):
            self.released = True

    first = FakeCapture([(False, None)])
    second = FakeCapture([(True, frame)])
    captures = [first, second]
    original = camera_mod.cv2.VideoCapture

    def fake_video_capture(_source, *_args):
        return captures.pop(0)

    camera_mod.cv2.VideoCapture = fake_video_capture
    source = None
    try:
        source = CSISource(
            640, 480, 30, allow_usb_fallback=True,
            startup_attempts=1, startup_delay_s=0.0,
            csi_open_attempts=2, csi_retry_delay_s=0.0)
        ok, result = source.read()
        assert ok
        assert result.shape == frame.shape
        assert source.backend == 'csi-gstreamer'
        assert first.released
        assert any('mo lai lan 2/2' in note for note in source.startup_notes)
        assert not captures
    finally:
        if source is not None:
            source.release()
        camera_mod.cv2.VideoCapture = original


def test_csi_close_during_startup_drops_late_capture_and_keeps_lease():
    import jetracer_baseline.camera as camera_mod

    original_capture = camera_mod.cv2.VideoCapture
    original_lease = camera_mod.CameraLease
    for pause_at in ('read', 'reopen'):
        entered, unblock = threading.Event(), threading.Event()
        captures, results = [], []

        class Capture(object):
            released = False

            def isOpened(self):
                return True

            def read(self):
                if pause_at == 'read':
                    entered.set()
                    unblock.wait(2.0)
                return False, None

            def release(self):
                self.released = True

        class Lease(object):
            released = False

            def acquire(self):
                pass

            def release(self):
                self.released = True

        def open_mock(*_args):
            cap = Capture()
            captures.append(cap)
            if pause_at == 'reopen' and len(captures) == 2:
                entered.set()
                unblock.wait(2.0)
            return cap

        camera_mod.cv2.VideoCapture = open_mock
        camera_mod.CameraLease = Lease
        source = thread = None
        try:
            source = CSISource(320, 240, 30, startup_attempts=1,
                               startup_delay_s=0.0, csi_open_attempts=2,
                               csi_retry_delay_s=0.0, allow_usb_fallback=False)
            thread = threading.Thread(target=lambda: results.append(source.read()))
            thread.start()
            assert entered.wait(1.0)
            source.release()
            assert source.cap is None and not source._lease.released
            unblock.set()
            thread.join(2.0)
            assert results == [(False, None)]
            assert len(captures) == (1 if pause_at == 'read' else 2)
            assert all(cap.released for cap in captures)
            assert source.cap is None and source._lease.released
        finally:
            unblock.set()
            if thread is not None:
                thread.join(2.0)
            if source is not None:
                source.release()
            camera_mod.cv2.VideoCapture = original_capture
            camera_mod.CameraLease = original_lease


def test_jetson_gstreamer_pipeline_has_low_latency_caps():
    pipeline = gstreamer_pipeline(
        640, 480, 30, flip_method=2,
        capture_width=1280, capture_height=720, sensor_id=0)
    assert 'nvarguscamerasrc' in pipeline
    assert 'sensor-id=0' in pipeline
    assert 'memory:NVMM' in pipeline
    assert 'width=(int)1280, height=(int)720' in pipeline
    assert 'format=(string)NV12' in pipeline
    assert 'nvvidconv flip-method=2' in pipeline
    assert 'format=(string)BGR' in pipeline
    assert 'drop=true' in pipeline
    assert 'max-buffers=1' in pipeline
    assert 'sync=false' in pipeline


def test_nvidia_driver_finds_sibling_repo_without_install():
    """Jetson co ~/jetracer repo nhung chua setup.py install van phai import."""
    temp_root = tempfile.mkdtemp(prefix='nvidia_jetracer_repo_')
    package = os.path.join(temp_root, 'jetracer')
    old_env = os.environ.get('JETRACER_NVIDIA_ROOT')
    old_path = list(sys.path)
    old_package = sys.modules.get('jetracer')
    old_module = sys.modules.get('jetracer.nvidia_racecar')
    try:
        os.makedirs(package)
        with open(os.path.join(package, '__init__.py'), 'w') as fh:
            fh.write('')
        with open(os.path.join(package, 'nvidia_racecar.py'), 'w') as fh:
            fh.write('class NvidiaRacecar(object):\n    pass\n')
        os.environ['JETRACER_NVIDIA_ROOT'] = temp_root
        sys.modules.pop('jetracer.nvidia_racecar', None)
        sys.modules.pop('jetracer', None)

        cls, module_path = driver_mod._load_nvidia_racecar()
        assert cls.__name__ == 'NvidiaRacecar'
        assert os.path.abspath(temp_root) in os.path.abspath(module_path)
    finally:
        sys.path[:] = old_path
        sys.modules.pop('jetracer.nvidia_racecar', None)
        sys.modules.pop('jetracer', None)
        if old_package is not None:
            sys.modules['jetracer'] = old_package
        if old_module is not None:
            sys.modules['jetracer.nvidia_racecar'] = old_module
        if old_env is None:
            os.environ.pop('JETRACER_NVIDIA_ROOT', None)
        else:
            os.environ['JETRACER_NVIDIA_ROOT'] = old_env
        shutil.rmtree(temp_root)


def test_nvidia_driver_passes_only_supported_i2c_addresses():
    """Ho tro ca driver chuan (mot dia chi) va image cu (hai dia chi)."""
    class StandardDriver(object):
        i2c_address = 0x40

        @classmethod
        def class_traits(cls):
            return {'i2c_address': object()}

    class LegacyDriver(object):
        i2c_address = 0x40
        i2c_address2 = 0x60

        @classmethod
        def class_traits(cls):
            return {'i2c_address': object(), 'i2c_address2': object()}

    standard = driver_mod._nvidia_constructor_kwargs(
        StandardDriver, i2c_address=0x40, i2c_address2=0x40)
    legacy = driver_mod._nvidia_constructor_kwargs(
        LegacyDriver, i2c_address=0x40, i2c_address2=0x40)

    assert standard == {'i2c_address': 0x40}
    assert legacy == {'i2c_address': 0x40, 'i2c_address2': 0x40}


def test_nvidia_driver_wraps_wrong_0x60_with_actionable_error():
    class BrokenDriver(object):
        i2c_address = 0x40
        i2c_address2 = 0x60

        @classmethod
        def class_traits(cls):
            return {'i2c_address': object(), 'i2c_address2': object()}

        def __init__(self, **_kwargs):
            raise ValueError('No I2C device at address: 0x60')

    original_loader = driver_mod._load_nvidia_racecar
    try:
        driver_mod._load_nvidia_racecar = lambda: (
            BrokenDriver, '/fake/jetracer/nvidia_racecar.py')
        try:
            driver_mod.NvidiaJetRacerDriver(
                i2c_address=0x40, i2c_address2=0x40)
            assert False, 'Phai bao loi khi driver van truy cap 0x60'
        except RuntimeError as exc:
            message = str(exc)
            assert 'i2c_address2=0x40' in message
            assert 'sai bien the driver' in message
            assert 'i2cdetect -y -r 1' in message
    finally:
        driver_mod._load_nvidia_racecar = original_loader


def test_waveshare_direct_driver_stops_and_watchdog_cuts_throttle():
    """Driver 0x40 phai cat ga truc tiep va khong giu lenh cu qua timeout."""
    class FakeServo(object):
        def __init__(self):
            self.history = []
            self.pulse_range = None

        @property
        def throttle(self):
            return self.history[-1] if self.history else None

        @throttle.setter
        def throttle(self, value):
            value = float(value)
            if value < -1.0 or value > 1.0:
                raise ValueError('Out of range')
            self.history.append(value)

        def set_pulse_width_range(self, low, high):
            self.pulse_range = (low, high)

    class FakeChannels(object):
        def __init__(self):
            self.items = [FakeServo() for _ in range(16)]

        def __getitem__(self, index):
            return self.items[index]

    class FakePCA(object):
        def __init__(self):
            self.frequency = None

    class FakeServoKit(object):
        instances = []

        def __init__(self, channels, address):
            self.channels = channels
            self.address = address
            self._pca = FakePCA()
            self.continuous_servo = FakeChannels()
            self.__class__.instances.append(self)

    original_loader = driver_mod._load_servo_kit
    drv = None
    try:
        driver_mod._load_servo_kit = lambda: FakeServoKit
        drv = driver_mod.NvidiaJetRacerDriver(
            implementation='waveshare_single_pca',
            i2c_address=0x40, pwm_frequency=None,
            steering_channel=0, throttle_channel=1,
            steering_gain=-0.65, steering_offset=0.0,
            throttle_gain=0.8, pulse_width_range=(750, 2250),
            command_timeout_s=0.12,
            steering_output_min=-0.40, steering_output_max=0.40)

        kit = FakeServoKit.instances[-1]
        steering = kit.continuous_servo[0]
        throttle = kit.continuous_servo[1]
        assert kit.address == 0x40
        assert kit._pca.frequency is None
        assert steering.pulse_range == (750, 2250)

        drv.set(0.50, 0.25)
        assert abs(steering.throttle - (-0.325)) < 1e-9
        assert abs(throttle.throttle - 0.20) < 1e-9
        drv.set(1.0, 0.0)
        assert abs(steering.throttle - (-0.40)) < 1e-9
        assert abs(drv.last_steering_output - (-0.40)) < 1e-9
        drv.stop()
        assert throttle.throttle == 0.0

        drv.set(0.0, 0.25)
        time.sleep(0.25)
        assert throttle.throttle == 0.0
    finally:
        if drv is not None:
            drv.close()
        driver_mod._load_servo_kit = original_loader


def test_hardware_preflight_initializes_driver_not_only_imports():
    """Preflight phai mo driver that va ghi neutral truoc khi bao OK."""
    from tools import check_hardware

    events = []

    class FakeDriver(object):
        def stop(self):
            events.append('stop')

        def close(self):
            events.append('close')

    original_probe = driver_mod.probe
    original_build = driver_mod.build_driver
    try:
        driver_mod.probe = lambda: {
            'nvidia': 'OK (chi moi truong) - fake',
            'dryrun': 'OK - fake',
        }
        driver_mod.build_driver = lambda kind, cfg: FakeDriver()
        assert check_hardware.check_driver('nvidia', _cfg())
        assert events == ['stop', 'close']
    finally:
        driver_mod.probe = original_probe
        driver_mod.build_driver = original_build


def test_power_mode_parser_and_shutdown_probe_helpers():
    from tools import check_hardware, diagnose_shutdown

    assert check_hardware._is_5w_mode('NV Power Mode: 5W\n1\n')
    assert check_hardware._is_5w_mode('MODE_ID: 1\n')
    assert not check_hardware._is_5w_mode('NV Power Mode: MAXN\n0\n')
    assert isinstance(diagnose_shutdown._uptime_s(), float)
    assert diagnose_shutdown._uptime_s() >= 0.0


def test_latest_grabber_surfaces_background_camera_error():
    """Exception trong thread camera phai hien ra UI, khong duoc chet im lang."""

    class BrokenSource(object):
        def __init__(self):
            self.released = False

        def read(self):
            raise RuntimeError('nvargus test failure')

        def release(self):
            self.released = True

    source = BrokenSource()
    grabber = LatestFrameGrabber(source, wait_timeout_s=0.2).start()
    try:
        frame, frame_id = grabber.read()
        assert frame is None
        assert frame_id == 0
        assert grabber.eof
        assert grabber.error is not None
        assert 'nvargus test failure' in str(grabber.error)
    finally:
        grabber.stop()
    assert source.released


def test_manual_collector_preview_control_and_recording():
    """Integration test UI-camera-control-recorder, khong can ipywidgets that."""

    class FakeWidget(object):
        def __init__(self, value=None, description='', **kwargs):
            self.value = value
            self.description = description
            self.disabled = False
            self._observers = []
            for key, item in kwargs.items():
                setattr(self, key, item)

        def observe(self, callback, names=None):
            # Khong mo phong reactivity that (khong bao gio goi callback khi
            # .value doi) - chi de cac widget dong bo (slider <-> o nhap so)
            # trong ManualDriveCollector khoi tao duoc ma khong AttributeError.
            self._observers.append(callback)

    class FakeLayout(object):
        def __init__(self, **kwargs):
            self.options = kwargs

    class FakeButton(FakeWidget):
        def __init__(self, *args, **kwargs):
            FakeWidget.__init__(self, *args, **kwargs)
            self._callbacks = []

        def on_click(self, callback):
            self._callbacks.append(callback)

    class FakeOutput(FakeWidget):
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return False

        def clear_output(self):
            pass

    class FakeAxis(object):
        def __init__(self):
            self.value = 0.0

    class FakeControllerButton(object):
        def __init__(self):
            self.value = 0.0
            self.pressed = False

    class FakeController(FakeWidget):
        def __init__(self, index=0):
            FakeWidget.__init__(self)
            self.index = index
            self.connected = True
            self.name = 'fake-gamepad'
            self.axes = [FakeAxis() for _ in range(4)]
            self.buttons = [FakeControllerButton() for _ in range(10)]

    class FakeBox(FakeWidget):
        def __init__(self, children):
            FakeWidget.__init__(self)
            self.children = children

    fake_widgets = types.ModuleType('ipywidgets.widgets')
    fake_widgets.Controller = FakeController
    fake_widgets.Image = FakeWidget
    fake_widgets.Output = FakeOutput
    fake_widgets.HTML = FakeWidget
    fake_widgets.Text = FakeWidget
    fake_widgets.BoundedIntText = FakeWidget
    fake_widgets.BoundedFloatText = FakeWidget
    fake_widgets.Checkbox = FakeWidget
    fake_widgets.FloatSlider = FakeWidget
    fake_widgets.Button = FakeButton
    fake_widgets.Layout = FakeLayout
    fake_widgets.HBox = FakeBox
    fake_widgets.VBox = FakeBox
    fake_widgets.Label = FakeWidget
    fake_package = types.ModuleType('ipywidgets')
    fake_package.widgets = fake_widgets

    old_package = sys.modules.get('ipywidgets')
    old_widgets = sys.modules.get('ipywidgets.widgets')
    sys.modules['ipywidgets'] = fake_package
    sys.modules['ipywidgets.widgets'] = fake_widgets

    temp_root = tempfile.mkdtemp(prefix='jetracer_ui_collect_')
    collector = None
    try:
        collector = ManualDriveCollector(
            config_path=CONFIG, out_root=temp_root,
            source_kind='synthetic', driver_kind='dryrun')
        collector._ensure_control_thread()
        collector._on_open_camera()

        deadline = time.time() + 2.0
        while time.time() < deadline and not collector.preview.value:
            time.sleep(0.02)
        assert collector._grabber is not None
        assert collector.preview.value

        # ARM lai tren mat dat khong phu thuoc checkbox danh rieng cho actuator.
        collector._on_arm()
        assert collector._armed
        assert not collector.use_deadman.value
        assert not collector.wheels_lifted.value
        assert type(collector._driver).__name__ == 'DryRunDriver'

        collector.controller.buttons[4].pressed = True
        collector.controller.buttons[4].value = 1.0
        collector.controller.axes[2].value = 0.50
        collector.controller.axes[1].value = -1.00
        time.sleep(0.15)
        assert 0.15 < collector._steering_cmd < 0.22
        assert collector._throttle_cmd > 0.15
        assert collector._deadman_pressed

        collector._on_start_recording()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if collector._writer is not None and collector._writer.count >= 2:
                break
            time.sleep(0.02)
        collector._on_stop_recording()
        assert collector.last_session_dir is not None
        labels = os.path.join(collector.last_session_dir, 'labels.csv')
        video = os.path.join(collector.last_session_dir, 'drive.avi')
        video_sidecar = os.path.join(
            collector.last_session_dir, 'drive.sidecar.csv')
        metadata_path = os.path.join(
            collector.last_session_dir, 'metadata.json')
        assert os.path.exists(labels)
        assert os.path.exists(video) and os.path.getsize(video) > 0
        assert os.path.exists(video_sidecar)
        with open(labels, 'r') as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 2
        assert float(rows[-1]['throttle_cmd']) > 0.15
        with open(video_sidecar, 'r') as fh:
            video_rows = list(csv.DictReader(fh))
        assert len(video_rows) > 0
        assert 'steering_cmd' in video_rows[0]
        with open(metadata_path, 'r') as fh:
            metadata = __import__('json').load(fh)
        assert metadata['video_frames'] == len(video_rows)
        assert metadata['video_error'] is None
        cap = cv2.VideoCapture(video)
        ok, saved_frame = cap.read()
        cap.release()
        assert ok and saved_frame is not None

        # Mo phong control loop dang ghi ga thi emergency den. Lock phai dam
        # bao lenh cuoi cung sau khi set bi treo duoc nha ra van la (0, 0).
        collector.controller.axes[1].value = 0.0
        time.sleep(0.08)

        class BlockingDriver(object):
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()
                self.blocked_once = False
                self.last = (0.0, 0.0)

            def set(self, steering, throttle):
                if throttle > 0.0 and not self.blocked_once:
                    self.blocked_once = True
                    self.started.set()
                    self.release.wait(1.0)
                self.last = (steering, throttle)

            def stop(self):
                self.last = (0.0, 0.0)

            def close(self):
                self.stop()

        blocking = BlockingDriver()
        with collector._driver_io_lock:
            collector._driver = blocking
        collector.controller.axes[1].value = -1.0
        assert blocking.started.wait(1.0)
        emergency = threading.Thread(target=collector._on_emergency)
        emergency.start()
        time.sleep(0.03)
        blocking.release.set()
        emergency.join(timeout=1.0)
        assert not emergency.is_alive()
        assert not collector._armed
        assert blocking.last == (0.0, 0.0)

        # Checkbox ke banh chi thuoc bai test, bo tick khong duoc cat lai that.
        collector._armed = True
        blocking.last = (0.2, 0.2)
        collector._on_wheels_lifted_change({'new': False})
        assert collector._armed
        assert blocking.last == (0.2, 0.2)
    finally:
        if collector is not None:
            collector.close()
        shutil.rmtree(temp_root)
        if old_package is None:
            del sys.modules['ipywidgets']
        else:
            sys.modules['ipywidgets'] = old_package
        if old_widgets is None:
            del sys.modules['ipywidgets.widgets']
        else:
            sys.modules['ipywidgets.widgets'] = old_widgets


def test_lane_detector_tracks_synthetic_road():
    """CTE phai BAM theo tam duong, khong chi nam trong [-1, 1].

    Ban test cu chi assert `-1 <= cte <= 1` va `found >= 30/40` - mot detector
    tra ve hang so 0.0 van pass. Nguon tong hop uon theo sin nen co ground truth
    thuc su; kiem tra tuong quan moi bat duoc loi bam nham vach.
    """
    # Nguon tong hop mo phong sa ban THI: vach trang dut khuc tren lane toi.
    cfg = _cfg(lane__line_color='white')
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=120)

    got, truth, found = [], [], 0
    for i in range(120):
        ok, frame = src.read()
        assert ok
        res = det.process(frame)
        assert -1.0 <= res.cte <= 1.0
        if res.found:
            found += 1
        got.append(res.cte)
        # SyntheticSource: center = w/2 + sin(i*0.05 + depth*1.2) * w*0.18
        # tai day anh (depth = 1) -> lech chuan hoa = sin(i*0.05 + 1.2) * 0.36
        truth.append(np.sin(i * 0.05 + 1.2) * 0.36)

    assert found >= 110, 'chi bat duoc vach o %d/120 frame' % found

    got = np.array(got)
    truth = np.array(truth)
    assert got.std() > 0.02, 'cte gan nhu khong doi - detector khong bam gi ca'
    corr = float(np.corrcoef(got, truth)[0, 1])
    assert corr > 0.9, 'cte khong bam tam duong (tuong quan chi %.3f)' % corr


def test_lane_detector_ignores_wrong_colour_line():
    """Doi mau vach ma detector van "tim thay" nghia la no dang bam nham thu khac."""
    cfg = _cfg(lane__line_color='red')      # nguon tong hop chi co vach TRANG
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=40)
    found = 0
    for _ in range(40):
        ok, frame = src.read()
        assert ok
        if det.process(frame).found:
            found += 1
    assert found <= 4, 'bat duoc vach do o %d/40 frame nhung anh khong co mau do' % found


def test_lane_detector_survives_dash_gaps():
    """Vach DUT: khong duoc bao mat vach chi vi dang o giua hai net."""
    cfg = _cfg(lane__line_color='white')
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=60)
    lost = 0
    for _ in range(60):
        ok, frame = src.read()
        assert ok
        if not det.process(frame).found:
            lost += 1
    assert lost <= 3, 'mat vach %d/60 frame tren duong lien mach' % lost


def test_lane_curvature_and_lookahead_do_not_saturate():
    """Do cong phai nam trong vung dung duoc, khong ghim +-1.

    Bam +-1 nghia la da thuc dang duoc NGOAI SUY ra ngoai vung nhin thay vach;
    khi do ga se tut ve v_min suot luot chay va xe bo het toc do.
    """
    cfg = _cfg(lane__line_color='white')
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=120)
    curvs, looks = [], []
    for _ in range(120):
        ok, frame = src.read()
        assert ok
        res = det.process(frame)
        if res.found:
            curvs.append(abs(res.curvature))
            looks.append(abs(res.cte_lookahead))
    assert curvs, 'khong frame nao tim thay vach'
    saturated = float(np.mean([c >= 0.999 for c in curvs]))
    assert saturated < 0.10, 'do cong bao hoa o %.0f%% frame' % (100 * saturated)
    assert float(np.mean([l >= 0.999 for l in looks])) < 0.10


def _synthetic_shaded_frame(w=160, h=120, rg_corner=1.6, bg_corner=1.2):
    """Anh xam trung tinh bi am do dan ve goc - gia lap dung lens shading that."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / 2.0
    base = 120.0
    img = np.zeros((h, w, 3), np.float64)
    img[:, :, 1] = base
    img[:, :, 2] = base * (1.0 + (rg_corner - 1.0) * r2)
    img[:, :, 0] = base * (1.0 + (bg_corner - 1.0) * r2)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_shading_corrector_flattens_measured_colour_cast():
    """Hieu chuan tren anh am do -> sau khi sua, R/G phai phang gan 1.0."""
    frame = _synthetic_shaded_frame()
    before = measure_ratio_profile(frame.astype(np.float64), n_bins=6)
    spread_before = max(p['rg'] for p in before) - min(p['rg'] for p in before)
    assert spread_before > 0.3, 'anh test phai co am mau ro ret'

    coeff_r, coeff_b = fit_coefficients(before)
    corrector = ShadingCorrector(coeff_r, coeff_b, enabled=True)
    after = measure_ratio_profile(
        corrector.apply(frame).astype(np.float64), n_bins=6)
    spread_after = max(p['rg'] for p in after) - min(p['rg'] for p in after)

    assert spread_after < 0.2 * spread_before, (
        'bien do R/G chi giam tu %.3f xuong %.3f' % (spread_before, spread_after))


def test_shading_is_off_unless_config_asks():
    """Mac dinh phai TAT. Sua mau am tham lam moi nguong da tune lech vo hinh."""
    from jetracer_baseline.config import Config

    frame = _synthetic_shaded_frame(w=40, h=40)
    off = ShadingCorrector.from_config(Config({}))
    assert off.enabled is False
    assert np.array_equal(off.apply(frame), frame)

    off2 = ShadingCorrector.from_config(
        Config({'camera': {'shading': {'enabled': False}}}))
    assert np.array_equal(off2.apply(frame), frame)

    try:
        ShadingCorrector.from_config(Config(
            {'camera': {'shading': {'enabled': True,
                                    'file': 'khong-ton-tai-abcxyz.yaml'}}}))
    except IOError:
        pass
    else:
        raise AssertionError('phai bao loi khi bat shading ma chua hieu chuan')


def test_shading_apply_resized_matches_apply():
    """Duong nhanh (resize truoc) phai cho cung ket qua voi duong thuong."""
    corrector = ShadingCorrector.from_config(_cfg())
    if not corrector.enabled:
        return
    big = _synthetic_shaded_frame(w=320, h=240)
    assert np.array_equal(corrector.apply(big),
                          corrector.apply_resized(big, (320, 240)))
    out = corrector.apply_resized(_synthetic_shaded_frame(w=640, h=480), (320, 240))
    assert out.shape == (240, 320, 3)


def test_shading_preserves_brightness_and_does_not_clip():
    """Chi doi mau, khong doi do sang -> khong duoc lam chay vung sang."""
    corrector = ShadingCorrector.from_config(_cfg())
    if not corrector.enabled:
        return
    frame = _synthetic_shaded_frame(w=160, h=120)
    out = corrector.apply(frame)
    mean_before, mean_after = float(frame.mean()), float(out.mean())
    assert abs(mean_after - mean_before) / mean_before < 0.15, (
        'do sang trung binh doi tu %.1f sang %.1f' % (mean_before, mean_after))
    assert float(np.mean(out >= 255)) < 0.01, 'qua nhieu pixel bi chay'


def test_tuning_engine_runs_without_ipywidgets():
    """Phan xu ly cua giao dien tune phai chay duoc ngoai Jupyter.

    Neu logic bi tron vao lop widget thi khong test duoc gi - ma logic moi la
    phan de sai. Test nay chinh la ly do LaneTuningEngine tach khoi LaneTuningUI.
    """
    engine = LaneTuningEngine(CONFIG)
    src = SyntheticSource(n_frames=30)
    engine.set_param('lane.line_color', 'white')

    for _ in range(30):
        ok, frame = src.read()
        assert ok
        result = engine.process(frame, 1.0 / 30.0)
        assert set(['proc', 'lane', 'steer', 'throttle']) <= set(result.keys())
        assert -1.0 <= result['steer'] <= 1.0
        assert 0.0 <= result['throttle'] <= engine.v_max + 1e-9

    summary = engine.stats.summary()
    assert summary is not None and summary['n'] == 30

    panel = engine.render_panel(result, fps=20.0, armed=False, width=640)
    assert panel.shape[1] == 640 and panel.ndim == 3
    assert engine.encode_jpeg(panel)


def test_processing_fps_meter_uses_completion_cadence():
    meter = ProcessingFpsMeter(window=60)
    for completed in (10.00, 10.04, 10.08):
        meter.tick(completed)
    assert abs(meter.fps - 25.0) < 1e-9
    assert abs(meter.mean_fps - 25.0) < 1e-9
    assert meter.current_fps(now=10.50, stale_after=0.25) == 0.0
    meter.reset()
    assert meter.frames == 0 and meter.fps == 0.0


def test_stopline_skips_resize_when_pipeline_frame_already_matches():
    detector = StoplineDetector(_cfg())
    frame = np.zeros((detector.ph, detector.pw, 3), np.uint8)
    original_resize = cv2.resize

    def forbidden_resize(*_args, **_kwargs):
        raise AssertionError('frame dung proc_size ma van resize lai')

    cv2.resize = forbidden_resize
    try:
        result = detector.process(frame)
    finally:
        cv2.resize = original_resize
    assert not result.found


def test_tuning_run_log_is_automatic_terminal_and_collision_safe(tmpdir=None):
    """Moi luot tao file rieng, co start/stop va t_rel thuc su tang."""
    workdir = tmpdir or tempfile.mkdtemp(prefix='tune_auto_log_')

    class Button(object):
        description = ''
        button_style = ''

    class Lane(object):
        cte = 0.1
        cte_lookahead = 0.2
        curvature = 0.01
        found = True
        n_bands = 5

    ui = LaneTuningUI.__new__(LaneTuningUI)
    ui.log_dir = workdir
    ui._logger = None
    ui._logger_lock = threading.Lock()
    ui._lifecycle_lock = threading.RLock()
    ui._state_lock = threading.Lock()
    ui._run_mode = MODE_STOP
    ui._run_started_mono = None
    ui._run_meter = ProcessingFpsMeter(60)
    ui._fps_stale_after = 0.25
    ui._last_run_summary = None
    ui.btn_log = Button()
    ui._armed = True
    ui._man_steer = -0.25
    ui._man_throttle = 0.15
    ui._log = lambda _message: None

    result = {
        'lane': Lane(), 'mode': 'THANG', 'steer': 0.1, 'throttle': 0.2,
        'servo_clipped': False,
    }
    paths = []
    try:
        for _ in range(2):
            assert ui._start_run_log(MODE_AUTO)
            # Frame bat dau truoc luc bam CHAY nhung hoan tat sau do khong
            # duoc lot vao session moi/tao t_rel am.
            ui._write_log_row(
                result, 6, time.time(), time.monotonic(), 4.0,
                pipeline_started_mono=ui._run_started_mono - 0.01)
            assert ui._run_meter.frames == 0
            time.sleep(0.01)
            ui._write_log_row(
                result, 7, time.time(), time.monotonic(), 4.0,
                pipeline_started_mono=ui._run_started_mono + 0.001)
            time.sleep(0.01)
            summary = ui._finish_run_log('stop')
            paths.append(summary['path'])
            with io.open(summary['path'], encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
            assert rows[0]['event'] == 'start'
            assert rows[-1]['event'] == 'stop'
            assert rows[-1]['state'] == 'FINISHED'
            assert float(rows[-1]['t_rel']) > 0.0
        assert paths[0] != paths[1]
        assert all(os.path.isfile(p) for p in paths)

        assert ui._start_run_log(MODE_MANUAL)
        ui._write_log_row(
            result, 8, time.time(), time.monotonic(), 4.0,
            pipeline_started_mono=ui._run_started_mono + 0.001)
        manual_summary = ui._finish_run_log('stop')
        with io.open(manual_summary['path'], encoding='utf-8') as fh:
            manual_rows = list(csv.DictReader(fh))
        assert manual_rows[1]['decision'] == 'MANUAL'
        assert manual_rows[1]['control_output'] == (
            'steer=-0.250;throttle=0.150')
        assert manual_rows[1]['throttle'] == '0.150'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_disarm_cannot_be_followed_by_stale_driver_set():
    """DUNG phai la action driver cuoi cung du control dang gui lenh."""
    entered = threading.Event()
    release_set = threading.Event()
    disarmed = threading.Event()
    actions = []

    class Driver(object):
        def set(self, _steer, _throttle):
            entered.set()
            release_set.wait(1.0)
            actions.append('set')

        def stop(self):
            actions.append('stop')

    class Engine(object):
        def stop_run(self):
            pass

    class Shaper(object):
        def reset(self):
            pass

    class Button(object):
        description = ''

    ui = LaneTuningUI.__new__(LaneTuningUI)
    ui._lifecycle_lock = threading.RLock()
    ui._engine_lock = threading.RLock()
    ui._driver_lock = threading.Lock()
    ui._state_lock = threading.Lock()
    ui._control_stop = threading.Event()
    ui.control_period = 0.01
    ui.mode = MODE_AUTO
    ui._armed = True
    ui._driver = Driver()
    ui._cv_steer = 0.2
    ui._cv_throttle = 0.3
    ui._cv_started_mono = time.monotonic()
    ui._run_started_mono = ui._cv_started_mono - 1.0
    ui._fps_stale_after = 0.25
    ui._camera_stopping = threading.Event()
    ui._stop_event = threading.Event()
    ui.engine = Engine()
    ui.shaper = Shaper()
    ui.btn_manual = Button()
    ui._sync_driver_limits = lambda: None
    ui._finish_run_log = lambda _reason: None
    ui._log = lambda _message: None

    control = threading.Thread(target=ui._control_loop)
    control.start()
    assert entered.wait(1.0)

    def stop_run():
        ui._disarm('stop')
        disarmed.set()

    stopper = threading.Thread(target=stop_run)
    stopper.start()
    time.sleep(0.02)
    assert not disarmed.is_set(), 'DUNG chen vao giua driver.set'
    ui._control_stop.set()
    release_set.set()
    control.join(1.0)
    stopper.join(1.0)

    assert disarmed.is_set()
    assert actions == ['set', 'stop'], actions
    assert ui._armed is False and ui.mode == MODE_STOP


def test_auto_control_rejects_command_from_pre_run_frame():
    commands = []

    class Driver(object):
        def set(self, steer, throttle):
            commands.append((steer, throttle))
            ui._control_stop.set()

    ui = LaneTuningUI.__new__(LaneTuningUI)
    ui._lifecycle_lock = threading.RLock()
    ui._driver_lock = threading.Lock()
    ui._state_lock = threading.Lock()
    ui._control_stop = threading.Event()
    ui.control_period = 0.001
    ui.mode = MODE_AUTO
    ui._armed = True
    ui._driver = Driver()
    ui._cv_steer = 0.8
    ui._cv_throttle = 0.9
    started = time.monotonic()
    ui._run_started_mono = started
    ui._cv_started_mono = started - 0.01
    ui._fps_stale_after = 0.25
    ui._camera_stopping = threading.Event()
    ui._stop_event = threading.Event()
    ui._sync_driver_limits = lambda: None
    ui._log = lambda _message: None

    ui._control_loop()
    assert commands == [(0.0, 0.0)], commands


def test_auto_control_disarms_when_pipeline_command_is_stale():
    actions = []
    reasons = []

    class Driver(object):
        def set(self, _steer, _throttle):
            actions.append('set')

        def stop(self):
            actions.append('stop')
            ui._control_stop.set()

    class Engine(object):
        def stop_run(self):
            pass

    class Shaper(object):
        def reset(self):
            pass

    class Button(object):
        description = ''

    ui = LaneTuningUI.__new__(LaneTuningUI)
    ui._lifecycle_lock = threading.RLock()
    ui._engine_lock = threading.RLock()
    ui._driver_lock = threading.Lock()
    ui._state_lock = threading.Lock()
    ui._control_stop = threading.Event()
    ui._camera_stopping = threading.Event()
    ui._stop_event = threading.Event()
    ui.control_period = 0.001
    ui.mode = MODE_AUTO
    ui._armed = True
    ui._driver = Driver()
    now = time.monotonic()
    ui._run_started_mono = now - 2.0
    ui._cv_started_mono = now - 1.0
    ui._fps_stale_after = 0.25
    ui._cv_steer = 0.8
    ui._cv_throttle = 0.9
    ui.engine = Engine()
    ui.shaper = Shaper()
    ui.btn_manual = Button()
    ui._sync_driver_limits = lambda: None
    ui._finish_run_log = lambda reason: reasons.append(reason)
    ui._log = lambda _message: None
    ui._set_status = lambda _message: None
    ui._metrics_html = lambda: ''
    ui.metrics = type('Metrics', (object,), {'value': ''})()
    ui._fps = 30.0

    ui._control_loop()
    assert actions == ['stop'], actions
    assert reasons == ['camera_stall']
    assert ui._camera_stopping.is_set() and ui._stop_event.is_set()
    assert not ui._armed and ui.mode == MODE_STOP


def test_ui_pipeline_exception_always_runs_finalizer():
    reasons = []
    ui = LaneTuningUI.__new__(LaneTuningUI)
    ui.status = type('Status', (object,), {'value': ''})()
    ui._log = lambda _message: None

    def fail(_grabber):
        raise RuntimeError('grabber copy failed')

    ui._process_loop = fail
    ui._finalize_camera_loop = (
        lambda _grabber, reason: reasons.append(reason))
    ui._loop(object())
    assert reasons == ['pipeline_error']


def test_log_summary_uses_full_run_throughput_not_mean_rolling_fps():
    from tools.analyze_log import summarise
    rows = []
    for i, t_rel in enumerate((0.00, 0.05, 0.10, 0.15)):
        rows.append({
            't_rel': str(t_rel), 'frame_id': str(i), 'fps': '30.0',
            'latency_ms': '4.0', 'sign_latency_ms': '', 'cte': '0.1',
            'state': 'RUNNING', 'event': '',
        })
    # FINISHED cua FSM van la mot frame pipeline that, khac marker dong file.
    rows.append({
        't_rel': '0.20', 'frame_id': '4', 'fps': '30.0',
        'latency_ms': '4.0', 'sign_latency_ms': '', 'cte': '0.1',
        'state': 'FINISHED', 'event': '',
        'control_output': 'steer=0;throttle=0',
        'lane_found': '1', 'n_bands': '5',
    })
    # Cached frame cu khong duoc tang frame count lan hai.
    rows.append(dict(rows[-1]))
    rows.append({
        't_rel': '0.25', 'frame_id': '5', 'fps': '30.0',
        'latency_ms': '0.0', 'sign_latency_ms': '', 'cte': '0.0',
        'state': 'FINISHED', 'event': 'stop', 'control_output': '',
        'lane_found': '', 'n_bands': '',
    })
    summary = summarise(rows)
    assert summary['frames'] == 5
    assert abs(summary['fps_mean'] - 20.0) < 1e-9


def test_log_analyzer_rejects_sidecar_schema(tmpdir=None):
    from tools.analyze_log import read_log
    workdir = tmpdir or tempfile.mkdtemp(prefix='analyze_sidecar_')
    try:
        path = os.path.join(workdir, 'run_speed_x.sidecar.csv')
        with io.open(path, 'w', encoding='utf-8') as fh:
            fh.write(u'timestamp,camera_backend\n1.0,csi\n')
        assert read_log(path) is None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_tuning_engine_applies_slider_changes_live():
    """Doi tham so phai co tac dung ngay, khong can tao lai engine.

    `dt` lon co chu dich: ga bi gioi han toc do doi (control/corner.py), nen do
    ga ngay sau mot tick nho se do gioi han do chu khong do gia tri slider.
    """
    engine = LaneTuningEngine(CONFIG)
    engine.set_param('lane.line_color', 'white')
    engine.set_param('control.slowdown', 0.0)
    engine.set_param('control.curve_enter', 5.0)   # ep luon o che do THANG
    engine.set_param('control.curve_exit', 4.0)
    engine.set_param('control.v_max', 0.60)
    engine.set_param('control.v_straight', 0.40)
    src = SyntheticSource(n_frames=5)
    ok, frame = src.read()
    assert ok
    for _ in range(5):
        result = engine.process(frame, 1.0)
    assert abs(result['throttle'] - 0.40) < 1e-6, result['throttle']

    engine.set_param('control.v_straight', 0.15)
    for _ in range(5):
        result = engine.process(frame, 1.0)
    assert abs(result['throttle'] - 0.15) < 1e-6, result['throttle']


def test_tuning_engine_saves_only_changed_keys(tmpdir=None):
    """LUU CONFIG phai ghi file override nho, khong de len default.yaml.

    Ghi de default.yaml se mat toan bo comment giai thich vi sao tung con so
    duoc chon - phan dat gia nhat cua file do.
    """
    workdir = tmpdir or tempfile.mkdtemp(prefix='tune_save_')
    try:
        engine = LaneTuningEngine(CONFIG)
        out = os.path.join(workdir, 'tuned.yaml')

        assert engine.save_overrides(out) is None, 'chua doi gi ma da ghi file'
        assert not os.path.exists(out)

        engine.set_param('control.pid.kp', 0.42)
        engine.set_param('lane.hsv_s_min', 111)
        diff = engine.save_overrides(out)

        assert diff == {'control': {'pid': {'kp': 0.42}},
                        'lane': {'hsv_s_min': 111}}, diff
        # File phai nap lai duoc va thuc su de len config goc
        cfg = load_config(CONFIG, [out])
        assert abs(float(cfg.get('control.pid.kp')) - 0.42) < 1e-9
        assert int(cfg.get('lane.hsv_s_min')) == 111
        assert cfg.get('control.v_max') == load_config(CONFIG).get('control.v_max')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_tuning_saved_thresholds_survive_colour_switch():
    """Nguong S/V da luu KHONG duoc de len preset khi doi mau vach.

    Neu UI ghi thang dai HSV cua mau do, sau nay doi sang `white` thi dai do cu
    van con hieu luc va detector khong bat duoc gi - loi im lang rat kho tim.
    """
    engine = LaneTuningEngine(CONFIG)
    engine.set_param('lane.hsv_s_min', 40)

    engine.set_param('lane.line_color', 'red')
    engine.rebuild()
    red_low = list(engine.lane.hsv_low_1)

    engine.set_param('lane.line_color', 'white')
    engine.rebuild()
    white_low = list(engine.lane.hsv_low_1)

    assert int(red_low[1]) == 40 and int(white_low[1]) == 40, 'S floor phai duoc ap'
    # Hue/V cua hai preset khac han nhau -> phai doi theo mau, khong bi dong bang
    assert int(red_low[2]) != int(white_low[2]), (
        'doi mau vach nhung dai HSV khong doi: %s vs %s' % (red_low, white_low))
    assert engine.lane.hsv_low_2 is None, 'preset trang khong co dai hue thu hai'


def test_tuning_engine_soft_start_ramps_throttle():
    """Bam CHAY khong duoc cho ga nhay thang len muc chay.

    Ga nhay tu 0 len v_max lam banh truot va nguoi bam khong kip phan ung neu
    chieu lai dang sai - dung luc nguy hiem nhat.
    """
    engine = LaneTuningEngine(CONFIG)
    engine.set_param('lane.line_color', 'white')
    # Tat gioi han toc do doi ga de test nay chi do RIENG soft-start.
    engine.set_param('control.throttle_rise_rate', 1000.0)
    engine.set_param('control.throttle_fall_rate', 1000.0)
    engine.soft_start_s = 1.0
    src = SyntheticSource(n_frames=5)
    ok, frame = src.read()
    assert ok

    # Chua chay -> preview hien ga day du, khong ramp
    for _ in range(3):
        full = engine.process(frame, 1.0)['throttle']
    assert full > 0.0

    t0 = 1000.0
    engine.start_run(now=t0)
    assert engine.running
    values = [engine.process(frame, 1.0, now=t0 + e)['throttle']
              for e in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert values[0] == 0.0, 'ga phai bat dau tu 0'
    for i in range(1, 4):
        assert values[i] > values[i - 1], 'ga phai tang dan: %r' % (values,)
    assert abs(values[3] - full) < 1e-6, 'sau soft_start_s phai dat ga day du'
    assert abs(values[4] - full) < 1e-6

    engine.stop_run()
    assert not engine.running
    assert abs(engine.process(frame, 1.0)['throttle'] - full) < 1e-6


def test_tuning_engine_soft_start_disabled_is_immediate():
    engine = LaneTuningEngine(CONFIG)
    engine.set_param('lane.line_color', 'white')
    engine.set_param('control.throttle_rise_rate', 1000.0)
    engine.soft_start_s = 0.0
    src = SyntheticSource(n_frames=3)
    ok, frame = src.read()
    assert ok
    for _ in range(3):
        full = engine.process(frame, 1.0)['throttle']
    engine.start_run(now=500.0)
    assert abs(engine.process(frame, 1.0, now=500.0)['throttle'] - full) < 1e-6


def _corner_cfg(**over):
    cfg = _cfg(control__v_straight=0.30, control__v_corner=0.12,
               control__curve_enter=0.25, control__curve_exit=0.15,
               control__v_min=0.05, control__v_max=0.35,
               control__slowdown=0.0, control__throttle_rise_rate=1000.0,
               control__throttle_fall_rate=1000.0)
    for key, value in over.items():
        cfg.set(key.replace('__', '.'), value)
    return cfg


def test_corner_controller_picks_two_speed_levels():
    """Doan thang di nhanh, khuc cua bo ga - day la muc dich cua module nay."""
    ctrl = CornerController(_corner_cfg())

    # Duong thang -> ga doan thang. dt lon de bo qua gioi han toc do doi ga.
    for _ in range(5):
        _steer, throttle, mode = ctrl.step(0.0, 0.0, 0.0, 1.0)
    assert mode == STRAIGHT
    assert abs(throttle - 0.30) < 1e-6, throttle

    # Cua gat -> ga cua
    for _ in range(5):
        _steer, throttle, mode = ctrl.step(0.0, 0.0, 0.6, 1.0)
    assert mode == CORNER
    assert abs(throttle - 0.12) < 1e-6, throttle


def test_corner_controller_hysteresis_prevents_flapping():
    """Mot nguong duy nhat se lam ga bat/tat lien tuc khi do cong dao quanh no."""
    ctrl = CornerController(_corner_cfg())
    assert ctrl.update_mode(0.30) == CORNER
    # Nam GIUA hai nguong -> phai GIU nguyen CUA, khong duoc nhay ve THANG
    assert ctrl.update_mode(0.20) == CORNER
    assert ctrl.update_mode(0.16) == CORNER
    assert ctrl.update_mode(0.10) == STRAIGHT
    # Va nguoc lai: giua hai nguong thi giu THANG
    assert ctrl.update_mode(0.20) == STRAIGHT
    assert ctrl.update_mode(0.26) == CORNER

    # Dem so lan doi che do tren mot chuoi dao quanh nguong
    ctrl2 = CornerController(_corner_cfg())
    modes = [ctrl2.update_mode(v) for v in
             [0.24, 0.26, 0.24, 0.26, 0.24, 0.26] * 5]
    switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
    assert switches <= 1, 'doi che do %d lan quanh nguong' % switches


def test_corner_controller_config_with_bad_hysteresis_is_repaired():
    """exit >= enter lam mat sach tac dung chong rung -> phai tu sua, khong im lang."""
    ctrl = CornerController(_corner_cfg(control__curve_exit=0.40))
    assert ctrl.curve_exit < ctrl.curve_enter


def test_confirmed_road_turn_boosts_steering_without_increasing_limits():
    cfg = _cfg(control__road_turn_steer_gain=1.2, control__corner_steer_gain=1.6,
               control__steer_max=0.6, control__corner_steer_max=0.6,
               control__v_corner=0.12, control__v_straight=0.25,
               control__v_max=0.30)
    ctrl = CornerController(cfg)
    straight = ctrl.step(0.25, 0.0, 0.0, 1.0)
    assert straight == (0.25, 0.25, STRAIGHT)
    right = ctrl.step(0.25, 0.0, 0.0, 1.0, road_turn=1)
    assert abs(right[0] - 0.48) < 1e-6 and right[1:] == (0.12, CORNER)
    left = ctrl.step(-0.25, 0.0, 0.0, 1.0, road_turn=-1)
    assert abs(left[0] + 0.48) < 1e-6
    assert abs(ctrl.step(-0.25, 0.0, 0.0, 1.0, road_turn=1)[0] + 0.40) < 1e-6
    assert ctrl.step(0.9, 0.0, 0.0, 1.0, road_turn=1)[0] == 0.60


def test_confirmed_turn_releases_steering_gradually_and_cancels_on_direction_change():
    ctrl = CornerController(_corner_cfg(control__road_turn_release_s=0.30,
                                        control__road_turn_steer_gain=1.2))
    for direction in (-1, 1):
        ctrl.reset()
        before, _, _ = ctrl.step(direction * 0.25, 0.0, 0.0, 0.10, road_turn=direction)
        released = [ctrl.step(0.0, 0.0, 0.0, 0.10)[0] for _ in range(3)]
        assert np.allclose(released, [before * 2.0 / 3.0, before / 3.0, 0.0])

    ctrl.step(0.25, 0.0, 0.0, 0.10, road_turn=1)
    assert ctrl.step(-0.05, 0.0, 0.0, 0.10)[0] == -0.05
    assert ctrl.step(0.0, 0.0, 0.0, 0.10)[0] == 0.0

    ctrl.step(0.25, 0.0, 0.0, 0.10, road_turn=1)
    assert ctrl.step(0.0, 0.0, 0.0, 0.10, lane_found=False)[0] == 0.0
    assert ctrl.step(0.0, 0.0, 0.0, 0.10)[0] == 0.0


def test_corner_controller_feedforward_steers_into_the_bend():
    """Cua hep can danh lai NGAY khi thay duong cong, khong doi CTE tang."""
    cfg = _corner_cfg(control__curve_feedforward=0.9,
                      control__corner_steer_gain=1.6,
                      control__steer_max=0.60)
    ctrl = CornerController(cfg)

    # CTE = 0 (xe dang dung tam vach) nhung phia truoc cong sang phai.
    # PID se ra 0; chi feed-forward moi sinh duoc lenh lai.
    steer, _t, mode = ctrl.step(0.0, 0.0, 0.5, 0.05)
    assert mode == CORNER
    assert steer > 0.3, 'feed-forward khong danh lai vao cua: %.3f' % steer

    ctrl.reset()
    steer_left, _t, _m = ctrl.step(0.0, 0.0, -0.5, 0.05)
    assert steer_left < -0.3, 'cua trai phai ra lai am: %.3f' % steer_left

    # Khong duoc vuot hard limit du nhan corner_steer_gain
    ctrl.reset()
    steer, _t, _m = ctrl.step(1.0, 1.0, 1.0, 0.05)
    assert abs(steer) <= 0.60 + 1e-9


def test_corner_controller_throttle_rate_limits_are_asymmetric():
    """Len ga tu tu (banh khong truot), bo ga nhanh (thay cua la buong ngay)."""
    cfg = _corner_cfg(control__throttle_rise_rate=0.8,
                      control__throttle_fall_rate=3.0)
    ctrl = CornerController(cfg)

    _s, t1, _m = ctrl.step(0.0, 0.0, 0.0, 0.1)      # muon 0.30, chi len 0.08
    assert abs(t1 - 0.08) < 1e-6, t1

    for _ in range(50):
        _s, t_top, _m = ctrl.step(0.0, 0.0, 0.0, 0.1)
    assert abs(t_top - 0.30) < 1e-6

    _s, t_down, _m = ctrl.step(0.0, 0.0, 0.9, 0.1)  # muon 0.12, duoc xuong 0.30
    assert abs(t_down - 0.0) < 1e-6 or t_down < t_top
    assert (t_top - t_down) > (t1 - 0.0), 'bo ga phai nhanh hon len ga'


def test_corner_controller_slows_down_when_lane_is_lost():
    """Mat vach thi dang lai theo gia tri cu - cang nhanh cang xa vach."""
    cfg = _corner_cfg()
    ctrl = CornerController(cfg)
    for _ in range(5):
        _s, fast, _m = ctrl.step(0.0, 0.0, 0.0, 1.0)
    for _ in range(5):
        _s, slow, _m = ctrl.step(0.0, 0.0, 0.0, 1.0, lane_found=False)
    assert slow < fast, 'mat vach ma ga khong giam: %.3f vs %.3f' % (slow, fast)


def test_controller_shaper_respects_deadzone_and_limits():
    """Tay cam khong duoc vuot gioi han da dat trong config."""
    shaper = ControllerShaper(_cfg())

    def settle(steer_raw, throttle_raw, **kw):
        shaper.reset()
        out = (0.0, 0.0)
        for _ in range(80):
            out = shaper.shape(steer_raw, throttle_raw, 1.0 / 30.0, **kw)
        return out

    assert settle(0.0, 0.0) == (0.0, 0.0)
    # Trong deadzone -> khong duoc sinh lenh nao
    steer, throttle = settle(0.05, 0.05)
    assert steer == 0.0 and throttle == 0.0

    steer, throttle = settle(1.0, 1.0)
    assert abs(steer - shaper.max_steering) < 1e-6
    assert abs(throttle - shaper.max_throttle) < 1e-6

    steer, _t = settle(-1.0, 0.0)
    assert abs(steer + shaper.max_steering) < 1e-6

    # Dao truc phai doi dau
    steer_norm, _t = settle(1.0, 0.0)
    steer_inv, _t = settle(1.0, 0.0, invert_steering=True)
    assert steer_norm > 0 > steer_inv


def test_controller_shaper_deadman_cuts_throttle_not_steering():
    """Nha dead-man phai cat GA ngay, nhung van cho lai de con be tranh."""
    shaper = ControllerShaper(_cfg())
    for _ in range(80):
        steer, throttle = shaper.shape(1.0, 1.0, 1.0 / 30.0, deadman_ok=True)
    assert throttle > 0.0

    for _ in range(80):
        steer, throttle = shaper.shape(1.0, 1.0, 1.0 / 30.0, deadman_ok=False)
    assert throttle == 0.0, 'nha dead-man ma van con ga: %.3f' % throttle
    assert abs(steer) > 0.0


def test_controller_shaper_slew_limits_sudden_stick_movement():
    """Gat can het co mot phat khong duoc thanh lenh nhay bac o servo."""
    shaper = ControllerShaper(_cfg())
    steer, _t = shaper.shape(1.0, 0.0, 1.0 / 30.0)
    assert abs(steer) < shaper.max_steering, (
        'lenh lai nhay thang len %.3f trong mot tick' % steer)
    rate = shaper.steering_slew_rate / 30.0
    assert abs(steer) <= rate + 1e-9


def test_dataset_writer_extra_fields_are_optional_and_appended():
    """Them cot phu KHONG duoc lam hong schema cua cac session da thu truoc do."""
    workdir = tempfile.mkdtemp(prefix='ds_extra_')
    try:
        frame = np.zeros((16, 16, 3), np.uint8)

        plain = DatasetSessionWriter(workdir, 'cu')
        plain.write(frame, 1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, False, False)
        plain.close()
        with io.open(plain.csv_path, encoding='utf-8') as fh:
            assert list(csv.DictReader(fh).fieldnames) == list(CSV_FIELDS)

        rich = DatasetSessionWriter(workdir, 'moi',
                                    extra_fields=['cv_steer', 'cte'])
        rich.write(frame, 2, 1.0, 1.0, 0.1, 0.2, 0.3, 0.4, True, True,
                   extra={'cv_steer': '-0.5', 'cte': '0.25'})
        rich.close()
        with io.open(rich.csv_path, encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        # Cot chuan phai giu nguyen THU TU, cot phu noi vao sau
        assert list(rows[0].keys())[:len(CSV_FIELDS)] == list(CSV_FIELDS)
        assert rows[0]['cv_steer'] == '-0.5' and rows[0]['cte'] == '0.25'
        assert rows[0]['steering_cmd'].startswith('0.3')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_dataset_writer_ignores_duplicate_extra_field_names():
    """Trung ten voi cot chuan se ghi de nhan train -> phai bi loai."""
    workdir = tempfile.mkdtemp(prefix='ds_dup_')
    try:
        writer = DatasetSessionWriter(
            workdir, 'dup', extra_fields=['steering_cmd', 'cte'])
        writer.write(np.zeros((8, 8, 3), np.uint8), 1, 1.0, 1.0,
                     0.0, 0.0, 0.77, 0.0, False, False,
                     extra={'steering_cmd': 'HONG', 'cte': '0.1'})
        writer.close()
        with io.open(writer.csv_path, encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]['steering_cmd'].startswith('0.77'), rows[0]['steering_cmd']
        assert rows[0]['cte'] == '0.1'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_ui_drive_modes_are_distinct():
    """Ba che do phai khac nhau - tron lan la nguon goc cua 'tuong tay ma tu chay'."""
    assert len({MODE_STOP, MODE_MANUAL, MODE_AUTO}) == 3


def test_manual_sliders_take_effect_without_restart():
    """Keo slider bo goc/toc do phai an ngay, khong phai mo lai giao dien."""
    cfg = _cfg()
    shaper = ControllerShaper(cfg)

    def settle(**kw):
        shaper.reset()
        out = (0.0, 0.0)
        for _ in range(120):
            out = shaper.shape(1.0, 1.0, 1.0 / 30.0, **kw)
        return out

    before_steer, before_throttle = settle()
    cfg.set('manual.max_steering', 1.0)
    cfg.set('manual.max_throttle', 0.45)
    shaper.reset_from_config(cfg)
    after_steer, after_throttle = settle()

    assert after_steer > before_steer and abs(after_steer - 1.0) < 1e-6
    assert after_throttle > before_throttle and abs(after_throttle - 0.45) < 1e-6


def test_manual_config_change_does_not_zero_running_command():
    """reset_from_config KHONG duoc dua lenh ve 0.

    Neu keo mot slider ma lenh ve 0 thi xe khuc ga giua chung moi lan chinh -
    khong the tune trong luc dang lai.
    """
    cfg = _cfg()
    shaper = ControllerShaper(cfg)
    for _ in range(120):
        shaper.shape(1.0, 1.0, 1.0 / 30.0)
    steer_before, throttle_before = shaper.command
    assert steer_before > 0 and throttle_before > 0

    cfg.set('manual.max_steering', 0.8)
    shaper.reset_from_config(cfg)
    assert shaper.command == (steer_before, throttle_before)


def test_every_manual_slider_has_a_config_value():
    """Slider khong doc duoc gia tri se hien giua thang do - vi du deadzone 0.15
    lam can gat gan nhu khong an."""
    cfg = _cfg()
    missing = [key for key, _l, _lo, _hi, _s in MANUAL_PARAMS
               if cfg.get(key) is None]
    assert not missing, 'thieu trong config: %s' % missing


def test_rolling_stats_window_and_flip_count():
    stats = RollingStats(window=5)
    for value in (0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1):
        stats.push(value, value, True, 4)
    assert stats.n == 5, 'cua so truot phai gioi han so mau'
    summary = stats.summary()
    assert summary['steer_flips'] == 4
    assert summary['loss_pct'] == 0.0

    stats.reset()
    assert stats.summary() is None


def test_dict_diff_ignores_unchanged_nested_keys():
    base = {'a': {'b': 1, 'c': 2}, 'd': 3}
    assert _dict_diff(base, {'a': {'b': 1, 'c': 2}, 'd': 3}) == {}
    assert _dict_diff(base, {'a': {'b': 9, 'c': 2}, 'd': 3}) == {'a': {'b': 9}}
    assert _dict_diff(base, {'a': {'b': 1, 'c': 2}, 'd': 3, 'e': 5}) == {'e': 5}


def test_shading_apply_returns_fresh_array_each_call():
    """apply() KHONG duoc tra buffer dung chung.

    recorder va dataset writer xep frame vao hang doi roi ghi dia o thread khac.
    Neu apply() tra ve cung mot buffer thi cac frame dang xep hang bi frame sau
    ghi de - anh luu ra dia se sai noi dung ma khong co loi nao bao.
    """
    corrector = ShadingCorrector.from_config(_cfg())
    if not corrector.enabled:
        return
    frame = _synthetic_shaded_frame(w=64, h=48)

    first = corrector.apply(frame)
    second = corrector.apply(frame)
    assert first is not second, 'apply() tra ve cung mot doi tuong buffer'
    assert np.array_equal(first, second)

    # Sua ket qua truoc khong duoc anh huong lan goi sau
    first[:] = 0
    third = corrector.apply(frame)
    assert np.array_equal(third, second)


def test_shading_at_source_wraps_and_cleans_every_frame():
    """apply_at=source -> MOI frame doc tu camera deu da sach, ke ca frame di
    vao anh dataset va video."""
    cfg = _cfg()
    cfg.set('camera.shading.enabled', True)
    cfg.set('camera.shading.apply_at', 'source')
    assert shading_applied_at_source(cfg)

    source = build_source(cfg, 'synthetic', n_frames=3)
    assert isinstance(source, ShadedSource)
    ok, frame = source.read()
    source.release()
    assert ok and frame is not None

    raw = _build_raw_source(cfg, 'synthetic', n_frames=3)
    assert not isinstance(raw, ShadedSource)
    ok, raw_frame = raw.read()
    raw.release()
    assert ok
    assert not np.array_equal(frame, raw_frame), (
        'nguon boc ShadedSource ma frame khong khac frame tho')


def test_shading_at_source_disables_pipeline_correction():
    """Sua hai lan se day gain len binh phuong: anh xam thanh xanh la va moi
    nguong mau da tune deu sai."""
    cfg = _cfg()
    cfg.set('camera.shading.enabled', True)

    cfg.set('camera.shading.apply_at', 'source')
    engine_src = LaneTuningEngine(CONFIG)
    engine_src.set_param('camera.shading.enabled', True)
    engine_src.set_param('camera.shading.apply_at', 'source')
    engine_src.rebuild()
    assert engine_src.shading.enabled is False, (
        'nguon da sua ma pipeline con sua nua -> sua hai lan')

    engine_pipe = LaneTuningEngine(CONFIG)
    engine_pipe.set_param('camera.shading.enabled', True)
    engine_pipe.set_param('camera.shading.apply_at', 'pipeline')
    engine_pipe.rebuild()
    assert engine_pipe.shading.enabled is True

    # Du sua o dau, he so ghi vao metadata phai luon la he so dang co hieu luc
    assert engine_src.shading_coeff_r == engine_pipe.shading_coeff_r


def test_shading_disabled_source_is_not_wrapped():
    cfg = _cfg()
    cfg.set('camera.shading.enabled', False)
    assert not shading_applied_at_source(cfg)
    source = build_source(cfg, 'synthetic', n_frames=2)
    assert not isinstance(source, ShadedSource)
    source.release()


def test_sign_tracker_needs_votes():
    """Mot frame nhieu KHONG duoc lam thay doi quyet dinh o nga tu."""
    cfg = _cfg(signs__vote_k=3, signs__vote_n=5)
    tracker = SignTracker(cfg)

    class D(object):
        def __init__(self, label, conf, area):
            self.label, self.confidence, self.area_ratio = label, conf, area

    assert tracker.update([D('turn_left', 0.9, 0.05)]) is None   # 1/3
    assert tracker.update([D('turn_left', 0.9, 0.05)]) is None   # 2/3
    out = tracker.update([D('turn_left', 0.9, 0.05)])            # 3/3 -> latch
    assert out is not None and out[0] == 'turn_left'


def test_runner_does_not_vote_same_cached_detection_more_than_once():
    """Lane 30 Hz khong duoc bien mot detection 10 Hz thanh ba vote gia."""
    runner = Runner.__new__(Runner)
    runner.tracker = SignTracker(_cfg(signs__vote_k=3, signs__vote_n=5))
    runner._last_detection_seq = None
    runner._stable_sign = None

    class D(object):
        label = 'turn_left'
        confidence = 0.9
        area_ratio = 0.05

    detection = [D()]
    assert runner._update_sign_tracker(detection, 1, time.time()) is None
    assert runner._update_sign_tracker(detection, 1, time.time()) is None
    assert runner._update_sign_tracker(detection, 1, time.time()) is None
    assert len(runner.tracker._history) == 1
    assert runner._update_sign_tracker(detection, 2, time.time()) is None
    stable = runner._update_sign_tracker(detection, 3, time.time())
    assert stable is not None and stable[0] == 'turn_left'


def test_green_light_needs_higher_confidence_than_red():
    """Bat doi xung an toan: bao nham xanh = vuot den do = HUY LUOT."""
    cfg = _cfg()
    tracker = SignTracker(cfg)

    class D(object):
        def __init__(self, label, conf, area):
            self.label, self.confidence, self.area_ratio = label, conf, area

    conf = 0.5   # nam giua conf_red (0.35) va conf_green (0.80)
    for _ in range(5):
        red = tracker.update([D('red_light', conf, 0.001)])
    assert red is not None and red[0] == 'red_light'   # do: chap nhan

    tracker2 = SignTracker(cfg)
    for _ in range(5):
        green = tracker2.update([D('green_light', conf, 0.001)])
    assert green is None                               # xanh: tu choi


def test_fsm_stops_on_red_and_does_not_creep():
    cfg = _cfg()
    fsm = fsm_mod.DecisionFSM(cfg, task='smartcity')
    fsm.start()

    class Lane(object):
        found, cte, curvature, n_pixels = True, 0.0, 0.0, 500

    class Stop(object):
        found, distance = True, 0.10   # da toi sat vach

    cmd = fsm.step(Lane(), stopline=Stop(), sign=('red_light', 0.9, time.time()))
    assert cmd.force_stop is True
    assert cmd.decision == 'stop'
    assert fsm.state == fsm_mod.S_WAIT_RED

    # Khong thay den nua -> VAN phai dung (nguyen tac: khong chac chan thi dung)
    cmd = fsm.step(Lane(), stopline=Stop(), sign=None)
    assert cmd.force_stop is True


def test_fsm_goes_on_green():
    cfg = _cfg(fsm__stop_hold_s=0.0)
    fsm = fsm_mod.DecisionFSM(cfg, task='smartcity')
    fsm.start()

    class Lane(object):
        found, cte, curvature, n_pixels = True, 0.0, 0.0, 500

    class Stop(object):
        found, distance = True, 0.10

    fsm.step(Lane(), stopline=Stop(), sign=('red_light', 0.9, time.time()))
    cmd = fsm.step(Lane(), stopline=Stop(), sign=('green_light', 0.9, time.time()))
    assert cmd.force_stop is False
    assert cmd.throttle_scale > 0


def test_fsm_mandatory_sign_turns():
    cfg = _cfg()
    fsm = fsm_mod.DecisionFSM(cfg, task='smartcity')
    fsm.start()

    class Lane(object):
        found, cte, curvature, n_pixels = True, 0.0, 0.0, 500

    cmd = fsm.step(Lane(), stopline=None, sign=('turn_left', 0.9, time.time()))
    assert cmd.decision == 'left'
    assert cmd.steer_override is not None and cmd.steer_override < 0


def test_full_pipeline_writes_valid_log(tmpdir=None):
    from jetracer_baseline.logging_csv import FIELDS

    cfg = _cfg()
    log_dir = os.path.join(ROOT, 'logs', 'test')
    runner = Runner(cfg, task='speed', driver_kind='dryrun',
                    source=SyntheticSource(n_frames=60), log_dir=log_dir,
                    verbose=False, sync=True, realtime=False)
    assert runner.worker is None, 'Speed Track khong duoc chay detector bien ngam'
    summary = runner.run(max_frames=60)

    assert summary['frames'] == 60
    with open(summary['log']) as fh:
        lines = fh.read().strip().split('\n')
    assert lines[0].split(',') == FIELDS
    assert len(lines) == 62                       # header + 60 frame + 1 dong ket
    # Moi dong du so cot -> parse duoc bang pandas/csv khi viet paper
    for line in lines[1:]:
        assert len(line.split(',')) == len(FIELDS)


def test_runner_constructor_failure_releases_camera():
    from jetracer_baseline import pipeline as pipeline_mod

    class Source(object):
        def __init__(self):
            self.released = False

        def read(self):
            return False, None

        def release(self):
            self.released = True

    source = Source()
    original = pipeline_mod.build_driver

    def fail_driver(_kind, _cfg_value):
        raise RuntimeError('driver init failed')

    pipeline_mod.build_driver = fail_driver
    try:
        try:
            Runner(_cfg(), task='speed', driver_kind='dryrun', source=source,
                   verbose=False, sync=True)
            raise AssertionError('Runner phai bao loi build_driver')
        except RuntimeError as exc:
            assert 'driver init failed' in str(exc)
    finally:
        pipeline_mod.build_driver = original
    assert source.released, 'constructor loi nhung camera/source chua release'


def test_live_runner_does_not_loop_forever_on_cached_eof_frame(tmpdir=None):
    workdir = tmpdir or tempfile.mkdtemp(prefix='runner_eof_')
    try:
        runner = Runner(
            _cfg(), task='speed', driver_kind='dryrun',
            source=SyntheticSource(n_frames=5), log_dir=workdir,
            verbose=False, sync=False, realtime=False)
        t0 = time.monotonic()
        summary = runner.run(max_seconds=2.0)
        assert time.monotonic() - t0 < 1.0
        with io.open(summary['log'], encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        assert rows[-1]['event'] == 'eof'
        assert rows[-1]['state'] == 'FINISHED'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_lane_cnn_uses_green_and_yellow_by_default():
    """Option B tu dung Ch0 va Ch2, model 1 kenh cu van hoat dong."""
    from jetracer_baseline.perception.lane_cnn import CnnLaneDetector, SEG_H, SEG_W

    for cfg_name, exp_engine in (
        ('cnn_option_b_fast.yaml', 'models/lane_option_b_fast.engine'),
        ('cnn_option_b.yaml', 'models/lane_option_b_resse.engine'),
        ('cnn_option_b_fastscnn.yaml', 'models/lane_option_b_fastscnn.engine'),
    ):
        cfg_opt = load_config(CONFIG, [os.path.join(ROOT, 'configs', cfg_name)])
        assert cfg_opt.get('lane.mode') == 'cnn'
        assert abs(float(cfg_opt.get('lane.cnn.roi_top')) - 0.34) < 1e-6
        assert cfg_opt.get('lane.cnn.engine') == exp_engine
        assert int(cfg_opt.get('camera.width')) == 320
        assert int(cfg_opt.get('camera.height')) == 240
        assert cfg_opt.get('camera.shading.apply_at') == 'pipeline'
        assert float(cfg_opt.get('tuning.run_preview_hz')) == 1.0

    class FakeEngine(object):
        def __init__(self):
            self.seg = None
            self.reg = None

        def infer(self, _x):
            return {'seg': self.seg, 'reg': self.reg}

    cfg = load_config(CONFIG, [os.path.join(ROOT, 'configs', 'cnn_option_b.yaml')])
    cfg.set('lane.cnn.engine', '')
    cfg.set('lane.cnn.road_guided', False)  # Config cu khong the bat lai Ch1.
    cfg.set('lane.cnn.check_reg', False)
    det = CnnLaneDetector(cfg)
    eng = FakeEngine()
    det.engine = eng
    frame = np.zeros((480, 640, 3), np.uint8)

    # 1. Option B 3 kenh: centerline (Ch2) lech phai -> cte duong, khong fallback
    seg3 = np.full((1, 3, SEG_H, SEG_W), -5.0, dtype=np.float32)
    seg3[0, 0, 10:64, 50:110] = 5.0
    seg3[0, 2, 10:64, 78:84] = 5.0
    eng.seg = seg3
    eng.reg = np.array([[0.25, 0.25, 0.0]], dtype=np.float32)
    res = det.process(frame, compute_bev=True)
    assert res.found
    assert res.cte > 0.05, 'centerline lech phai nhung cte=%.3f' % res.cte
    assert not det.last_used_fallback
    assert det.road_guided and det.last_road_mask is not None and det.last_mask is not None

    # Ch1 mot minh khong duoc sinh tam hoac mo cong khoi hanh.
    det.reset()
    seg3[0, 0] = -5.0
    seg3[0, 1] = 5.0
    seg3[0, 2] = -5.0
    for _ in range(5):
        assert not det.process(frame).found
    assert not det.road_start_ready

    # 3. Tuong thich nguoc model 1 kenh cu (64x128)
    det.reset()
    seg1 = np.full((1, 1, SEG_H, SEG_W), -5.0, dtype=np.float32)
    seg1[0, 0, 10:64, 61:67] = 5.0
    eng.seg = seg1
    eng.reg = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    res1 = det.process(frame)
    assert res1.found
    assert abs(res1.cte) < 0.15
    assert det.last_road_mask is None
    assert not det.road_guided


def test_lane_cnn_road_edges_guide_yellow_without_white():
    from jetracer_baseline.perception.lane_cnn import (
        CnnLaneDetector, PROC_H, PROC_W, SEG_H, SEG_W)

    cfg = load_config(CONFIG, [os.path.join(ROOT, 'configs', 'cnn_option_b_fast.yaml')])
    cfg.set('lane.cnn.engine', '')
    det = CnnLaneDetector(cfg)
    det._M_seg = np.diag([PROC_W / float(SEG_W), PROC_H / float(SEG_H), 1.0])
    det._build_bev_lut()
    frame = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)

    class FakeEngine(object):
        def infer(self, _x):
            return {'seg': self.seg, 'reg': self.reg}

    det.engine = FakeEngine()
    seg = np.full((3, SEG_H, SEG_W), -5.0, dtype=np.float32)
    for y in range(SEG_H):
        t = (SEG_H - 1 - y) / float(SEG_H - 1)
        edge = int(round(18 + 70.0 * t * t))
        seg[0, y, edge + 2:] = 5.0
    det.engine.seg = seg
    det.engine.reg = np.zeros(3, dtype=np.float32)
    before = det.process(frame, need_aux=False)
    assert before.found and before.cte_lookahead > 0.20
    assert det.last_road_turn == 1 and det.last_used_fallback

    seg[1, :, :9] = 10.0
    det.reset()
    after = det.process(frame, compute_bev=True)
    assert (after.found, after.cte, after.cte_lookahead, after.curvature) == (
        before.found, before.cte, before.cte_lookahead, before.curvature)
    assert det.road_guided
    assert after.debug.any() and det.last_mask.any()

    seg[2, :, 20:24] = 10.0
    det.engine.reg[0] = -0.65
    det.reset()
    assert det.process(frame).cte_lookahead > 0.20 and det.last_used_fallback

    seg[2] = -5.0
    for y in range(SEG_H):
        t = (SEG_H - 1 - y) / float(SEG_H - 1)
        x = int(round(64 + 48.0 * t * t))
        seg[2, y, x - 2:x + 3] = 5.0
    det.engine.reg[0] = 0.0
    det.reset()
    assert det.process(frame).found and not det.last_used_fallback and det.last_road_turn == 1

    # Ch0 phu kin tam nhin: vien anh khong the dung lam mep duong.
    seg[0] = 5.0
    seg[2] = -5.0
    det.reset()
    assert not det.process(frame).found


def test_speed_track_ui_option_b_panel_and_recent_run_gate():
    """Speed Track hien Ch0 va tam vang, khoi hanh theo Ch0."""
    from jetracer_baseline.perception.lane_cnn import (
        CnnLaneDetector, PROC_H, PROC_W, SEG_H, SEG_W)

    # 1. RollingStats.recent_loss_pct: 100 frame mat vach luc be xe + 20 frame
    # thay vach gan nhat -> recent_loss_pct phai la 0%, khong chan nut CHAY.
    st = RollingStats(window=200)
    for _ in range(100):
        st.push(0.0, 0.0, False, 0)
    for _ in range(20):
        st.push(0.02, 0.02, True, 6)
    assert st.summary()['loss_pct'] > 80.0
    assert st.recent_loss_pct(20) == 0.0

    cfg = load_config(CONFIG, [os.path.join(ROOT, 'configs', 'cnn_option_b.yaml')])
    cfg.set('lane.cnn.engine', '')
    cfg.set('lane.cnn.check_reg', False)
    engine = LaneTuningEngine(CONFIG)
    engine.cfg = cfg
    det = CnnLaneDetector(cfg)

    engine.lane = det
    engine._dirty = False
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    workdir = tempfile.mkdtemp(prefix='ui_green_start_gate_')
    try:
        class Btn(object):
            description = ''
            button_style = ''

        class DummyDrv(object):
            def stop(self):
                pass

        ui = LaneTuningUI.__new__(LaneTuningUI)
        ui._lifecycle_lock = threading.RLock()
        ui._engine_lock = threading.RLock()
        ui._driver_lock = threading.Lock()
        ui._state_lock = threading.Lock()
        ui._logger_lock = threading.Lock()
        ui._camera_stopping = threading.Event()
        ui._grabber = None
        ui._driver = DummyDrv()
        ui.driver_kind = 'dryrun'
        ui.mode = MODE_STOP
        ui._armed = False
        ui.engine = engine
        ui.log_dir = workdir
        ui._logger = None
        ui._run_mode = MODE_STOP
        ui._run_started_mono = None
        ui._run_meter = ProcessingFpsMeter(60)
        ui._fps_stale_after = 0.25
        ui._last_run_summary = None
        ui.btn_log = Btn()
        ui.btn_manual = Btn()
        ui.shaper = ControllerShaper(_cfg())
        ui._ensure_control_thread = lambda: None
        logs = []
        ui._log = lambda m: logs.append(m)
        ui._set_status = lambda _s: None

        # (a) Chua mo camera -> tu choi
        ui._on_run_locked()
        assert not ui._armed and any('MO CAMERA' in m for m in logs)

        ui._grabber = object()
        # CHAY duoc khi chi co Ch0, cho huong lai moi cap ga.
        det._M_seg = np.diag([PROC_W / float(SEG_W), PROC_H / float(SEG_H), 1.0])
        det._build_bev_lut()

        class FakeGreenEngine(object):
            road = False
            edges = False

            def infer(self, _x):
                seg = np.full((3, SEG_H, SEG_W), -5.0, dtype=np.float32)
                seg[1] = 5.0  # Ch1 nhieu van bi bo qua.
                if self.edges:
                    seg[0, :, 34:94] = 5.0
                elif self.road:
                    seg[0] = 5.0  # Ch0 kin tam nhin, chua co huong lai.
                return {'seg': seg, 'reg': np.zeros(3, dtype=np.float32)}

        det.engine = FakeGreenEngine()
        engine.stats.reset()
        for _ in range(5):
            result = engine.process(frame, 1.0 / 30.0)
        ui._on_run_locked()
        assert not ui._armed and not result['road_start_ready']

        det.engine.road = True
        for _ in range(4):
            result = engine.process(frame, 1.0 / 30.0)
        ui._on_run_locked()
        assert not ui._armed and not result['road_start_ready']
        result = engine.process(frame, 1.0 / 30.0)
        assert result['road_start_ready'] and not result['lane'].found
        assert result['_preview_road'].any()
        assert '_preview_white' not in result
        panel = engine.render_panel(result, fps=30.0, armed=False, driver_kind='dryrun')
        assert panel.shape[2] == 3
        ui._on_run_locked()
        assert ui._armed and ui.mode == MODE_AUTO
        now = engine._run_t0 + 2.0
        waiting = engine.process(frame, 1.0 / 30.0, now=now)
        assert waiting['waiting_for_path'] and waiting['throttle'] == 0.0
        det.engine.edges = True
        started = engine.process(frame, 1.0 / 30.0, now=now + 0.1)
        assert started['lane'].found and started['used_fallback'] and started['_preview_mask'].any()
        assert not started['waiting_for_path'] and started['ramp'] == 0.0
        running = engine.process(frame, 1.0 / 30.0, now=now + 1.2)
        assert running['throttle'] > 0.0
        ui._disarm_locked('stop')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_camera_open_timeout_does_not_block_ui_or_install_late_capture():
    from jetracer_baseline import tuning_ui as tuning_ui_mod

    ui = LaneTuningUI.__new__(LaneTuningUI)
    ui._thread = ui._grabber = ui._camera_open_ticket = None
    ui._camera_stopping = threading.Event()
    ui._lifecycle_lock = threading.RLock()
    ui._camera_open_timeout_s = 0.10
    ui.source_kind, ui.video_path = 'csi', None
    ui.engine = types.SimpleNamespace(cfg=None)
    ui._log = ui._set_status = lambda _msg: None
    timeout = threading.Event()
    ui._set_error_status = lambda _msg: timeout.set()
    opening, return_source, released = (threading.Event() for _ in range(3))
    calls = []

    def slow_camera(_cfg, _kind, video_path=None):
        calls.append(1)
        opening.set()
        return_source.wait(2.0)
        return types.SimpleNamespace(release=released.set)

    build_source = tuning_ui_mod.build_source
    tuning_ui_mod.build_source = slow_camera
    try:
        ui._on_open()
        assert opening.wait(1.0)
        assert timeout.wait(1.0)
        assert not released.is_set()
        ui._on_open()
        assert len(calls) == 1 and ui._grabber is None
    finally:
        return_source.set()
        tuning_ui_mod.build_source = build_source
    assert released.wait(1.0)
    assert ui._grabber is None


def test_run_start_survives_slow_driver_init_and_first_frame_latency():
    """Mo phong dung loi Jetson Nano 5W: driver init cham + inference ~100ms.

    Truoc khi sua:
      - _write_log_row() khoa _lifecycle_lock moi frame -> bi build_driver() treo
      - _fps_stale_after = 0.25s + do tu frame START -> vua bam CHAY da bao
        camera_stall -> _stop_event.set() -> camera dong bang, xe khong chay!
    Sau khi sua:
      - camera KHONG bi ngat, preview van song, xe tang ga va chay bam line,
        log CSV ghi du dong RUNNING va dong sach khi DUNG.
    """
    from jetracer_baseline import tuning_ui as tuning_ui_mod
    from jetracer_baseline.perception.lane_cnn import CnnLaneDetector, SEG_H, SEG_W

    class FakeWidget(object):
        def __init__(self, value=None, description='', **kwargs):
            self.value = value
            self.description = description
            self.disabled = False
            self._observers = []
            for key, item in kwargs.items():
                setattr(self, key, item)

        def observe(self, callback, names=None):
            self._observers.append(callback)

    class FakeLayout(object):
        def __init__(self, **kwargs):
            self.options = kwargs

    class FakeButton(FakeWidget):
        def __init__(self, *args, **kwargs):
            FakeWidget.__init__(self, *args, **kwargs)
            self._callbacks = []

        def on_click(self, callback):
            self._callbacks.append(callback)

    class FakeOutput(FakeWidget):
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return False

    class FakeController(FakeWidget):
        def __init__(self, index=0):
            FakeWidget.__init__(self)
            self.index = index
            self.connected = False
            self.axes = []
            self.buttons = []

    class FakeBox(FakeWidget):
        def __init__(self, children):
            FakeWidget.__init__(self)
            self.children = children

    class FakeTab(FakeBox):
        def set_title(self, _idx, _title):
            pass

    fake_widgets = types.ModuleType('ipywidgets.widgets')
    fake_widgets.Controller = FakeController
    fake_widgets.Image = FakeWidget
    fake_widgets.Output = FakeOutput
    fake_widgets.HTML = FakeWidget
    fake_widgets.Text = FakeWidget
    fake_widgets.BoundedIntText = FakeWidget
    fake_widgets.BoundedFloatText = FakeWidget
    fake_widgets.Checkbox = FakeWidget
    fake_widgets.Dropdown = FakeWidget
    fake_widgets.IntSlider = FakeWidget
    fake_widgets.FloatSlider = FakeWidget
    fake_widgets.Button = FakeButton
    fake_widgets.Layout = FakeLayout
    fake_widgets.HBox = FakeBox
    fake_widgets.VBox = FakeBox
    fake_widgets.Tab = FakeTab
    fake_package = types.ModuleType('ipywidgets')
    fake_package.widgets = fake_widgets

    class SlowOptionBEngine(object):
        def __init__(self, infer_delay_s=0.09):
            self.infer_delay_s = infer_delay_s
            self.closed = False
            self.calls = 0
            seg = np.full((1, 3, SEG_H, SEG_W), -5.0, dtype=np.float32)
            seg[0, 0, 10:64, 34:94] = 5.0
            seg[0, 1, 10:64, 34:38] = 5.0
            seg[0, 1, 10:64, 90:94] = 5.0
            seg[0, 2, 10:64, 62:66] = 5.0
            self._seg = seg
            self._reg = np.zeros((1, 3), dtype=np.float32)

        def infer(self, _x):
            time.sleep(self.infer_delay_s)
            self.calls += 1
            return {'seg': self._seg, 'reg': self._reg}

        def close(self):
            self.closed = True

    class RecordingDriver(object):
        def __init__(self):
            self.commands = []
            self.stopped = 0
            self.closed = False

        def set(self, steer, throttle):
            self.commands.append((float(steer), float(throttle)))

        def stop(self):
            self.stopped += 1

        def close(self):
            self.closed = True
            self.stop()

    old_package = sys.modules.get('ipywidgets')
    old_widgets = sys.modules.get('ipywidgets.widgets')
    old_build_driver = tuning_ui_mod.build_driver
    sys.modules['ipywidgets'] = fake_package
    sys.modules['ipywidgets.widgets'] = fake_widgets

    workdir = tempfile.mkdtemp(prefix='ui_jetson_start_')
    ui = None
    slow_eng = SlowOptionBEngine(infer_delay_s=0.09)
    created_drivers = []

    def slow_build_driver(_kind, _cfg):
        # Mo phong ServoKit/Blinka/I2C khoi tao cham (~0.35s) trong lan CHAY dau tien
        time.sleep(0.35)
        drv = RecordingDriver()
        created_drivers.append(drv)
        return drv

    try:
        tuning_ui_mod.build_driver = slow_build_driver
        ui = LaneTuningUI(
            config_path=CONFIG, source_kind='synthetic',
            driver_kind='nvidia', soft_start_s=0.2,
            record_dir=workdir, data_root=workdir,
            control_hz=30.0, simple=True)
        ui.log_dir = workdir
        cfg_opt = load_config(
            CONFIG, [os.path.join(ROOT, 'configs', 'cnn_option_b.yaml')])
        cfg_opt.set('lane.cnn.engine', '')
        cfg_opt.set('lane.cnn.check_reg', False)
        det = CnnLaneDetector(cfg_opt)
        det.engine = slow_eng
        ui.engine.cfg = cfg_opt
        ui.engine.lane = det
        ui.engine._dirty = False
        ui.engine.rebuild()

        ui._on_open()
        # Cho du frame de qua cong mat duong on dinh
        deadline = time.time() + 4.0
        while time.time() < deadline:
            summary = ui.engine.stats.summary()
            if summary is not None and summary['n'] >= 8:
                break
            time.sleep(0.02)
        assert ui._grabber is not None and not ui._stop_event.is_set()
        calls_before_run = slow_eng.calls

        # Bam CHAY khi build_driver mat 0.35s va moi frame CNN mat ~0.09s
        ui._on_run()
        assert ui._armed and ui.mode == MODE_AUTO
        assert len(created_drivers) == 1
        drv = created_drivers[0]
        # Trong luc build_driver() chay, _process_loop KHONG bi block boi _lifecycle_lock
        assert slow_eng.calls > calls_before_run, (
            '_process_loop bi treo trong luc build_driver() khoi tao')

        # Cho xe nhan nhieu lenh ga duong lien tiep qua nhieu frame (~100ms/frame)
        # ma KHONG bi false camera_stall hay ngat camera
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if (ui._run_meter.frames >= 4 and
                    any(th > 0.05 for _st, th in drv.commands)):
                break
            time.sleep(0.02)

        assert ui._armed, (
            'Xe bi tu dong DISARM ngay sau khi bam CHAY! status=%s'
            % getattr(ui.status, 'value', ''))
        assert ui._grabber is not None and not ui._stop_event.is_set(), (
            'Camera bi ngat ket noi sai ngay sau khi bam CHAY!')
        assert ui._run_meter.frames >= 4, (
            'Khong duy tri duoc nhieu frame sau khi CHAY: %d'
            % ui._run_meter.frames)
        assert any(th > 0.05 for _st, th in drv.commands), (
            'Xe khong nhan duoc lenh ga duong nao: %r' % (drv.commands[:10],))

        log_path = ui._logger.path
        ui._on_disarm()
        assert not ui._armed and ui.mode == MODE_STOP
        assert ui._grabber is not None and not ui._stop_event.is_set()
        with io.open(log_path, encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]['event'] == 'start'
        assert any(r['state'] == 'RUNNING' for r in rows)
        assert rows[-1]['event'] == 'stop' and rows[-1]['state'] == 'FINISHED'

        ui.close()
        ui = None
        assert slow_eng.closed, 'ui.close() phai goi engine/lane/TensorRT close()'
        assert drv.closed
    finally:
        if ui is not None:
            ui.close()
        tuning_ui_mod.build_driver = old_build_driver
        shutil.rmtree(workdir, ignore_errors=True)
        if old_package is None:
            sys.modules.pop('ipywidgets', None)
        else:
            sys.modules['ipywidgets'] = old_package
        if old_widgets is None:
            sys.modules.pop('ipywidgets.widgets', None)
        else:
            sys.modules['ipywidgets.widgets'] = old_widgets


def test_control_loop_tolerates_jitter_using_completion_time_but_stops_on_true_stall():
    """Do stall theo moc HOAN TAT lenh gan nhat + startup grace; stall that van cat ga."""
    commands = []
    reasons = []

    class Driver(object):
        def set(self, steer, throttle):
            commands.append((steer, throttle))
            ui._control_stop.set()

        def stop(self):
            commands.append(('stop', 0.0))
            ui._control_stop.set()

    class Engine(object):
        def stop_run(self):
            pass

    class Shaper(object):
        def reset(self):
            pass

    class Button(object):
        description = ''

    def _make_ui():
        inst = LaneTuningUI.__new__(LaneTuningUI)
        inst._lifecycle_lock = threading.RLock()
        inst._engine_lock = threading.RLock()
        inst._driver_lock = threading.Lock()
        inst._state_lock = threading.Lock()
        inst._control_stop = threading.Event()
        inst._camera_stopping = threading.Event()
        inst._stop_event = threading.Event()
        inst.control_period = 0.001
        inst.mode = MODE_AUTO
        inst._armed = True
        inst._driver = Driver()
        inst._cv_steer = 0.25
        inst._cv_throttle = 0.20
        inst._fps_stale_after = 0.25
        inst._run_startup_grace_s = 1.0
        inst._camera_stop_reason = None
        inst._camera_stop_detail = None
        inst.engine = Engine()
        inst.shaper = Shaper()
        inst.btn_manual = Button()
        inst.status = type('Status', (object,), {'value': ''})()
        inst.metrics = type('Metrics', (object,), {'value': ''})()
        inst._sync_driver_limits = lambda: None
        inst._finish_run_log = lambda reason: reasons.append(reason)
        inst._log = lambda _m: None
        inst._set_status = lambda _m: None
        inst._metrics_html = lambda: ''
        inst._fps = 15.0
        return inst

    # (a) Frame bat dau tu 0.30s truoc nhung VUA hoan tat 0.05s truoc -> KHONG stall
    ui = _make_ui()
    now = time.monotonic()
    ui._run_started_mono = now - 1.0
    ui._cv_started_mono = now - 0.30
    ui._cv_updated_mono = now - 0.05
    ui._control_loop()
    assert commands == [(0.25, 0.20)], commands
    assert ui._armed and not ui._stop_event.is_set()

    # (b) Frame hoan tat cach day 0.40s (> 0.25s) -> stall that -> cat ga & dung cam
    commands[:] = []
    reasons[:] = []
    ui = _make_ui()
    now = time.monotonic()
    ui._run_started_mono = now - 1.5
    ui._cv_started_mono = now - 0.50
    ui._cv_updated_mono = now - 0.40
    ui._control_loop()
    assert commands == [('stop', 0.0)], commands
    assert reasons == ['camera_stall']
    assert ui._camera_stop_reason == 'camera_stall'
    assert ui._stop_event.is_set() and not ui._armed
    assert 'PIPELINE MAT FRAME' in ui.status.value

    # (c) Dang cho frame dau tien sau CHAY trong startup_grace (0.40s < 1.0s) -> gui neutral (0, 0)
    commands[:] = []
    reasons[:] = []
    ui = _make_ui()
    now = time.monotonic()
    ui._run_started_mono = now - 0.40
    ui._cv_started_mono = now - 0.45
    ui._cv_updated_mono = now - 0.42
    ui._control_loop()
    assert commands == [(0.0, 0.0)], commands
    assert ui._armed and not ui._stop_event.is_set()

    # (d) Qua startup_grace (1.20s > 1.0s) ma van khong co frame sau CHAY -> stall & cat ga
    commands[:] = []
    reasons[:] = []
    ui = _make_ui()
    now = time.monotonic()
    ui._run_started_mono = now - 1.20
    ui._cv_started_mono = None
    ui._cv_updated_mono = None
    ui._control_loop()
    assert commands == [('stop', 0.0)], commands
    assert reasons == ['camera_stall']
    assert ui._stop_event.is_set() and not ui._armed


def test_finalize_camera_loop_preserves_error_banner_and_releases_before_clearing_grabber():
    """Finalizer phai giu nguyen thong bao loi va nha camera truoc khi xoa _grabber."""
    events = []

    class FakeGrabber(object):
        def stop(self):
            events.append(('grabber_stop', ui._grabber is self))

    ui = LaneTuningUI.__new__(LaneTuningUI)
    ui._lifecycle_lock = threading.RLock()
    ui._record_lock = threading.Lock()
    ui._writer_lock = threading.Lock()
    ui._camera_stopping = threading.Event()
    ui._recorder = None
    ui._writer = None
    ui.status = type('Status', (object,), {'value': ''})()
    ui._log = lambda _m: None
    ui._disarm_locked = lambda reason: events.append(('disarm', reason))

    # (a) camera_stall -> phai giu ly do trong status, khong ghi de bang thong bao chung
    grabber = FakeGrabber()
    ui._grabber = grabber
    ui._camera_stop_reason = 'camera_stall'
    ui._camera_stop_detail = 'command qua han 0.80s > 0.75s'
    ui._finalize_camera_loop(grabber, 'camera_stall')
    assert events == [('disarm', 'camera_stall'), ('grabber_stop', True)], events
    assert ui._grabber is None
    assert 'PIPELINE MAT FRAME' in ui.status.value
    assert '0.80s' in ui.status.value

    # (b) camera_error -> phai hien loi camera goc trong status
    events[:] = []
    grabber2 = FakeGrabber()
    ui._grabber = grabber2
    ui._camera_stop_reason = 'camera_error'
    ui._camera_stop_detail = 'csi-gstreamer mat frame'
    ui._finalize_camera_loop(grabber2, 'camera_error')
    assert 'MAT KET NOI CAMERA' in ui.status.value
    assert 'csi-gstreamer mat frame' in ui.status.value

    # (c) camera_stop binh thuong -> hien trang thai thuong
    events[:] = []
    grabber3 = FakeGrabber()
    ui._grabber = grabber3
    ui._camera_stop_reason = None
    ui._camera_stop_detail = None
    ui._finalize_camera_loop(grabber3, 'camera_stop')
    assert 'camera da dung; bam MO CAMERA de mo lai' in ui.status.value


def test_csi_source_recovers_transient_runtime_drop_and_fails_on_permanent_loss():
    """CSISource chiu duoc rot 1-2 frame tam thoi, nhung van bao loi khi mat cam that."""
    import jetracer_baseline.camera as camera_mod

    frame = np.full((12, 16, 3), 99, dtype=np.uint8)

    class FakeCapture(object):
        def __init__(self, reads):
            self.reads = list(reads)
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if not self.reads:
                return False, None
            return self.reads.pop(0)

        def release(self):
            self.released = True

    # Frame 1 = warmup OK; sau do 2 lan False tam thoi -> frame OK; cuoi cung mat han
    cap = FakeCapture([
        (True, frame),
        (False, None),
        (False, None),
        (True, frame),
        (False, None),
        (False, None),
        (False, None),
    ])
    original = camera_mod.cv2.VideoCapture
    camera_mod.cv2.VideoCapture = lambda *_args, **_kwargs: cap
    source = None
    try:
        source = CSISource(
            640, 480, 30, allow_usb_fallback=False,
            startup_attempts=1, startup_delay_s=0.0,
            csi_open_attempts=1, csi_retry_delay_s=0.0,
            runtime_read_retries=3, runtime_retry_delay_s=0.0)
        ok0, f0 = source.read()
        assert ok0 and f0 is not None

        # Rot 2 frame tam thoi -> tu phuc hoi trong read() thu 2
        ok1, f1 = source.read()
        assert ok1 and f1 is not None and source.last_error is None

        # Mat 3 frame lien tiep -> tra False va ghi last_error ro rang
        ok2, f2 = source.read()
        assert (not ok2) and f2 is None
        assert source.last_error and '3 lan doc lien tiep' in source.last_error
    finally:
        if source is not None:
            source.release()
        camera_mod.cv2.VideoCapture = original



def test_camera_dying_during_build_driver_aborts_run_and_preserves_error_detail():
    """Camera chet trong luc build_driver(): KHONG ARM, KHONG tao CSV, KHONG xoa ly do loi."""
    from jetracer_baseline import tuning_ui as tuning_ui_mod

    workdir = tempfile.mkdtemp(prefix='ui_cam_die_driver_')
    old_build = tuning_ui_mod.build_driver

    class DummyGrabber(object):
        def __init__(self):
            self.eof = False
            self.error = None

        def stop(self):
            pass

    class DummyDriver(object):
        def stop(self):
            pass

    class DummyButton(object):
        description = ''
        button_style = ''

    def _make_ui(grabber):
        eng = LaneTuningEngine(CONFIG)
        for _ in range(20):
            eng.stats.push(0.0, 0.0, True, 6)
        ui = LaneTuningUI.__new__(LaneTuningUI)
        ui._lifecycle_lock = threading.RLock()
        ui._engine_lock = threading.RLock()
        ui._driver_lock = threading.Lock()
        ui._state_lock = threading.Lock()
        ui._logger_lock = threading.Lock()
        ui._record_lock = threading.Lock()
        ui._writer_lock = threading.Lock()
        ui._camera_stopping = threading.Event()
        ui._grabber = grabber
        ui._driver = None
        ui.driver_kind = 'nvidia'
        ui.mode = MODE_STOP
        ui._armed = False
        ui.engine = eng
        ui.log_dir = workdir
        ui._logger = None
        ui._recorder = None
        ui._writer = None
        ui._run_mode = MODE_STOP
        ui._run_started_mono = None
        ui._run_meter = ProcessingFpsMeter(60)
        ui._fps_stale_after = 0.75
        ui._last_run_summary = None
        ui._camera_stop_reason = None
        ui._camera_stop_detail = None
        ui.btn_log = DummyButton()
        ui.btn_manual = DummyButton()
        ui.shaper = ControllerShaper(_cfg())
        ui.status = type('Status', (object,), {'value': ''})()
        ui._ensure_control_thread = lambda: None
        ui._controller_ready = lambda: True
        ui._read_sticks = lambda: (0.0, 0.0)
        ui._logs = []
        ui._log = lambda m: ui._logs.append(m)
        return ui

    try:
        # (a) _on_run_locked: camera loi giua luc build_driver() dang chay
        grab1 = DummyGrabber()
        ui1 = _make_ui(grab1)

        def dying_driver_run(_kind, _cfg_obj):
            grab1.eof = True
            grab1.error = IOError('nvargus died during driver init')
            ui1._camera_stop_reason = 'camera_error'
            ui1._camera_stop_detail = 'nvargus died during driver init'
            ui1._camera_stopping.set()
            return DummyDriver()

        tuning_ui_mod.build_driver = dying_driver_run
        ui1._on_run_locked()
        assert not ui1._armed and ui1.mode == MODE_STOP
        assert ui1._logger is None
        assert ui1._camera_stop_reason == 'camera_error'
        assert ui1._camera_stop_detail == 'nvargus died during driver init'
        assert os.listdir(workdir) == [], 'Khong duoc tao file CSV khi camera da chet'

        # Finalizer sau do van giu nguyen thong bao loi goc tren status
        ui1._finalize_camera_loop(grab1, 'camera_error')
        assert 'MAT KET NOI CAMERA' in ui1.status.value
        assert 'nvargus died during driver init' in ui1.status.value

        # (b) _enter_mode_locked (MODE_MANUAL): cung phai huy va khong tao CSV
        grab2 = DummyGrabber()
        ui2 = _make_ui(grab2)

        def dying_driver_manual(_kind, _cfg_obj):
            grab2.eof = True
            grab2.error = IOError('csi died in manual init')
            ui2._camera_stop_reason = 'camera_error'
            ui2._camera_stop_detail = 'csi died in manual init'
            ui2._camera_stopping.set()
            return DummyDriver()

        tuning_ui_mod.build_driver = dying_driver_manual
        ok = ui2._enter_mode_locked(MODE_MANUAL)
        assert ok is False
        assert not ui2._armed and ui2.mode == MODE_STOP
        assert ui2._logger is None
        assert ui2._camera_stop_detail == 'csi died in manual init'
        assert os.listdir(workdir) == []
    finally:
        tuning_ui_mod.build_driver = old_build
        shutil.rmtree(workdir, ignore_errors=True)


def test_tensorrt_engine_init_pops_and_detaches_context_on_keyboard_interrupt():
    """KeyboardInterrupt trong TensorRTEngine.__init__ phai pop + detach primary context."""
    from jetracer_baseline.perception.lane_cnn import TensorRTEngine

    ops = []

    class FakeContext(object):
        def push(self):
            ops.append('push')

        def pop(self):
            ops.append('pop')

        def detach(self):
            ops.append('detach')

    class FakeDevice(object):
        def __init__(self, _idx):
            pass

        def retain_primary_context(self):
            ops.append('retain')
            return FakeContext()

    class FakeLogger(object):
        WARNING = 1

        def __init__(self, _level):
            pass

    class FakeRuntime(object):
        def __init__(self, _logger):
            pass

        def deserialize_cuda_engine(self, _buf):
            raise KeyboardInterrupt('simulated Ctrl-C during deserialize')

    fake_trt = types.ModuleType('tensorrt')
    fake_trt.Logger = FakeLogger
    fake_trt.Runtime = FakeRuntime

    fake_cuda_drv = types.ModuleType('pycuda.driver')
    fake_cuda_drv.init = lambda: ops.append('init')
    fake_cuda_drv.Device = FakeDevice
    fake_pycuda = types.ModuleType('pycuda')
    fake_pycuda.driver = fake_cuda_drv

    old_trt = sys.modules.get('tensorrt')
    old_pycuda = sys.modules.get('pycuda')
    old_pycuda_drv = sys.modules.get('pycuda.driver')
    sys.modules['tensorrt'] = fake_trt
    sys.modules['pycuda'] = fake_pycuda
    sys.modules['pycuda.driver'] = fake_cuda_drv

    fd, tmp_engine = tempfile.mkstemp(prefix='fake_trt_', suffix='.engine')
    os.write(fd, b'fake-engine-bytes')
    os.close(fd)
    try:
        raised = False
        try:
            TensorRTEngine(tmp_engine)
        except KeyboardInterrupt:
            raised = True
        assert raised, 'KeyboardInterrupt phai duoc truyen ra ngoai'
        # Stack context phai can bang (so push == so pop) va phai detach
        assert ops.count('push') == ops.count('pop'), ops
        assert 'detach' in ops, ops
        assert ops[-1] == 'detach', ops
    finally:
        if os.path.exists(tmp_engine):
            os.remove(tmp_engine)
        for name, old in (('tensorrt', old_trt), ('pycuda', old_pycuda),
                          ('pycuda.driver', old_pycuda_drv)):
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_write_log_row_waits_for_arm_once_logger_is_attached():
    """Khi _logger da gan (nhung _armed chua bat), _write_log_row phai cho, khong bo sot frame."""
    workdir = tempfile.mkdtemp(prefix='ui_write_log_race_')

    class Btn(object):
        description = ''
        button_style = ''

    class Lane(object):
        cte = 0.05
        cte_lookahead = 0.08
        curvature = 0.01
        found = True
        n_bands = 6

    ui = LaneTuningUI.__new__(LaneTuningUI)
    ui.log_dir = workdir
    ui._logger = None
    ui._logger_lock = threading.Lock()
    ui._lifecycle_lock = threading.RLock()
    ui._state_lock = threading.Lock()
    ui._run_mode = MODE_STOP
    ui._run_started_mono = None
    ui._run_meter = ProcessingFpsMeter(60)
    ui._fps_stale_after = 0.75
    ui._last_run_summary = None
    ui.btn_log = Btn()
    ui._armed = False
    ui._log = lambda _m: None

    result = {
        'lane': Lane(), 'mode': 'THANG', 'steer': 0.1, 'throttle': 0.2,
        'servo_clipped': False,
    }
    try:
        # (a) Chua ARM va chua co _logger -> fast-path KHONG bi chan boi _lifecycle_lock
        done_unarmed = threading.Event()
        with ui._lifecycle_lock:
            def call_unarmed():
                ui._write_log_row(result, 1, time.time(), time.monotonic(), 5.0,
                                  pipeline_started_mono=time.monotonic())
                done_unarmed.set()
            t1 = threading.Thread(target=call_unarmed)
            t1.start()
            assert done_unarmed.wait(0.5), '_write_log_row khong duoc block khi chua mo logger'
            t1.join(0.5)

        # (b) _start_run_log da gan _logger trong _lifecycle_lock nhung _armed chua True
        #     -> _write_log_row phai doi _lifecycle_lock va ghi du frame dau tien
        done_first = threading.Event()
        with ui._lifecycle_lock:
            assert ui._start_run_log(MODE_AUTO)
            started = ui._run_started_mono
            assert ui._logger is not None and not ui._armed

            def call_first_frame():
                ui._write_log_row(result, 42, time.time(), time.monotonic(), 5.0,
                                  pipeline_started_mono=started + 0.001)
                done_first.set()

            t2 = threading.Thread(target=call_first_frame)
            t2.start()
            time.sleep(0.03)
            assert not done_first.is_set(), (
                '_write_log_row bo sot frame dau tien truoc khi _armed=True')
            ui._armed = True
        assert done_first.wait(1.0)
        t2.join(1.0)
        assert ui._run_meter.frames == 1
        summary = ui._finish_run_log('stop')
        with io.open(summary['path'], encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        assert any(r['frame_id'] == '42' and r['state'] == 'RUNNING' for r in rows), rows
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_csi_source_release_nulls_cap_before_releasing():
    """CSISource.release() phai gan self.cap=None TRUOC khi goi cap.release()."""
    import jetracer_baseline.camera as camera_mod

    frame = np.full((12, 16, 3), 55, dtype=np.uint8)
    cap_was_none_during_release = []
    source_box = []

    class InspectingCapture(object):
        def __init__(self):
            self.read_calls = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.read_calls += 1
            if self.read_calls == 1:
                return True, frame
            return False, None

        def release(self):
            cap_was_none_during_release.append(source_box[0].cap is None)
            self.released = True

    cap = InspectingCapture()
    original = camera_mod.cv2.VideoCapture
    camera_mod.cv2.VideoCapture = lambda *_args, **_kwargs: cap
    source = None
    try:
        source = CSISource(
            640, 480, 30, allow_usb_fallback=False,
            startup_attempts=1, startup_delay_s=0.0,
            csi_open_attempts=1, csi_retry_delay_s=0.0,
            runtime_read_retries=3, runtime_retry_delay_s=0.0)
        source_box.append(source)
        ok, f = source.read()
        assert ok and f is not None

        source.release()
        assert cap.released
        assert cap_was_none_during_release == [True], cap_was_none_during_release
        # Doc sau khi release khong duoc goi cap.read() nua
        reads_before = cap.read_calls
        ok_after, f_after = source.read()
        assert (not ok_after) and f_after is None
        assert cap.read_calls == reads_before
    finally:
        if source is not None:
            source.release()
        camera_mod.cv2.VideoCapture = original


def test_cnn_fast_path_copies_only_road_and_center_preview():
    from jetracer_baseline.perception.lane_cnn import CnnLaneDetector, SEG_H, SEG_W

    cfg = load_config(CONFIG, [os.path.join(ROOT, 'configs', 'cnn_option_b_fast.yaml')])
    cfg.set('lane.cnn.engine', '')
    cfg.set('lane.cnn.check_reg', False)
    engine = LaneTuningEngine(CONFIG)
    engine.cfg = cfg
    det = CnnLaneDetector(cfg)

    class _FakeEng(object):
        def infer(self, _x):
            s = np.full((1, 3, SEG_H, SEG_W), -5.0, dtype=np.float32)
            s[0, 0, 10:64, 34:94] = 5.0
            s[0, 1, 10:64, 34:38] = 5.0
            s[0, 1, 10:64, 90:94] = 5.0
            s[0, 2, 10:64, 62:66] = 5.0
            return {'seg': s, 'reg': np.zeros((1, 3), dtype=np.float32)}

    det.engine = _FakeEng()
    engine.lane = det
    engine._dirty = False
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    r_no = engine.process(frame, 1.0 / 30.0, copy_preview=False, need_aux=False)
    assert r_no['lane'].found and r_no['lane'].debug is None
    assert '_preview_mask' not in r_no and '_preview_road' not in r_no
    r_yes = engine.process(frame, 1.0 / 30.0, copy_preview=True, need_aux=True)
    assert r_yes['lane'].found and r_yes['lane'].debug is not None
    assert r_yes['_preview_mask'].any() and r_yes['_preview_road'].any()
    assert '_preview_white' not in r_yes
    # Kiem tra fit LUT tren duong nong khop voi ket qua khi bat BEV/aux
    assert abs(r_no['lane'].cte - r_yes['lane'].cte) < 1e-6


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS  ' + name)
            except Exception as exc:
                failures += 1
                print('FAIL  %s : %s' % (name, exc))
    print('')
    print('%d loi' % failures)
    sys.exit(1 if failures else 0)
