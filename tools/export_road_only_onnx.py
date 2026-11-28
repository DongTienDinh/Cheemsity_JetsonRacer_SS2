# -*- coding: utf-8 -*-
"""Cat tia (prune) checkpoint Option B (.pt) thanh ONNX 1 kenh Road-Only (Ch0) sieu nhe cho Jetson Nano.

Khong can train lai! Cat truc tiep tu trong so da train:
  - Giu duy nhat kenh Ch0 (drivable_road) cua `seg_head` (Conv 1x1 tu 3 kenh -> 1 kenh).
  - Loai bo hoan toan `reg_head` (bo GlobalAvgPool + Linear 96->32->3 + Tanh).
  - Giam 3x luong du lieu copy GPU -> CPU moi frame (24,579 -> 8,192 float32).
  - Tuong thich tuyet doi TensorRT 7.1.3 (opset 11, ir_version <= 6, batch=1 tinh).

Cach chay:
  .\\.venv_cuda\\Scripts\\python.exe tools\\export_road_only_onnx.py
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TOOLS_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from train_lane_option_b import (  # noqa: E402
    IN_H,
    IN_W,
    MODEL_REGISTRY,
)


class RoadOnlyWrapper(nn.Module):
    """Boc model Option B: chi giu nhanh encoder/decoder -> 1 kenh Ch0 (1, 1, 64, 128)."""

    def __init__(self, base_model):
        super(RoadOnlyWrapper, self).__init__()
        self.base = base_model
        # Thay seg_head (c_in -> 3) bang (c_in -> 1) giu nguyen trong so Ch0!
        old_head = self.base.seg_head
        new_head = nn.Conv2d(old_head.in_channels, 1, kernel_size=1, bias=(old_head.bias is not None))
        with torch.no_grad():
            new_head.weight.copy_(old_head.weight[0:1])
            if old_head.bias is not None:
                new_head.bias.copy_(old_head.bias[0:1])
        self.base.seg_head = new_head
        # Xoa reg_head de graph ONNX khong con nhanh hoi quy thua
        self.base.reg_head = nn.Identity()

    def forward(self, x):
        seg, _ = self.base(x)
        return seg


def export_road_only(model_key, pt_path, out_onnx):
    base = MODEL_REGISTRY[model_key]()
    state = torch.load(pt_path, map_location='cpu')
    base.load_state_dict(state)
    base.eval()

    # Do sai khac bit-exact giua Ch0 goc va ban cat tia
    rng = np.random.RandomState(0)
    x_test = torch.from_numpy(rng.uniform(0.0, 255.0, (1, 3, IN_H, IN_W)).astype(np.float32))
    with torch.no_grad():
        orig_seg, _ = base(x_test)
        orig_ch0 = orig_seg[:, 0:1].numpy()

    wrapped = RoadOnlyWrapper(base).eval()
    with torch.no_grad():
        pruned_ch0 = wrapped(x_test).numpy()
    max_diff = float(np.max(np.abs(orig_ch0 - pruned_ch0)))

    n_params = sum(p.numel() for p in wrapped.parameters())
    dummy = torch.zeros((1, 3, IN_H, IN_W), dtype=torch.float32)
    kwargs = {
        'input_names': ['input'],
        'output_names': ['seg'],
        'opset_version': 11,
        'do_constant_folding': True,
    }
    try:
        torch.onnx.export(wrapped, dummy, out_onnx, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapped, dummy, out_onnx, **kwargs)

    import onnx
    om = onnx.load(out_onnx)
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
    onnx.save(om, out_onnx)

    # Kiem tra ONNXRuntime khop bit-exact voi ban goc
    import onnxruntime as ort
    sess = ort.InferenceSession(out_onnx, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: x_test.numpy()})[0]
    ort_diff = float(np.max(np.abs(orig_ch0 - ort_out)))
    size_kb = os.path.getsize(out_onnx) / 1024.0
    print('[OK] %s -> %s' % (os.path.basename(pt_path), os.path.basename(out_onnx)))
    print('     params=%d | nodes=%d | size=%.1f KB | out=%s | max_diff=%.2e' % (
        n_params, len(om.graph.node), size_kb, list(ort_out.shape), max(max_diff, ort_diff)))
    assert ort_diff < 1e-4, 'Sai lech qua lon: %s' % ort_diff


def main():
    ap = argparse.ArgumentParser(description='Export 1-channel road-only ONNX from Option B .pt')
    ap.add_argument('--pt-dir', default=os.path.join(REPO_DIR, 'models', 'option_b_retrained'))
    ap.add_argument('--out-dir', default=os.path.join(REPO_DIR, 'models'))
    args = ap.parse_args()

    pairs = [
        ('lanetiny_fast',  'lanetiny_fast.pt',  'lane_road_only_fast.onnx'),
        ('lanetiny_resse', 'lanetiny_resse.pt', 'lane_road_only_resse.onnx'),
    ]
    for mkey, pt_fn, onnx_fn in pairs:
        pt_p = os.path.join(args.pt_dir, pt_fn)
        if not os.path.isfile(pt_p):
            print('[BO QUA] Khong thay:', pt_p)
            continue
        export_road_only(mkey, pt_p, os.path.join(args.out_dir, onnx_fn))


if __name__ == '__main__':
    main()
