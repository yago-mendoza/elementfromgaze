"""Export the generic base (models/generic_base.pkl) for the in-browser demo.

Writes <out>/gaze-net.bin (float32, little endian) and <out>/gaze-net.json (where each tensor sits).
BatchNorm is folded into the convolution before it, so the page runs conv + bias + ReLU + max-pool.
Also writes <out>/gaze-net.check.json: a few eye patches and the network's outputs for them, which
the page's test compares against (same numbers to ~1e-4, or the port is wrong).

    python docs/export_web.py <out dir>
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elementfromgaze import generic  # noqa: E402
from elementfromgaze.appearance import IN_H, IN_W, PATCH_H, PATCH_W, _norm  # noqa: E402


def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64)
    g, b = bn.weight.detach().numpy(), bn.bias.detach().numpy()
    mu, var = bn.running_mean.numpy(), bn.running_var.numpy()
    s = g / np.sqrt(var + bn.eps)
    return (w * s[:, None, None, None]).astype(np.float32), (b - mu * s).astype(np.float32)


def main(out: Path) -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    base = pickle.loads(generic.GENERIC_BASE.read_bytes())
    members = getattr(base, "members", [base])
    blobs, index, off = [], {}, 0

    def put(name, a):
        nonlocal off
        a = np.ascontiguousarray(a, np.float32)
        index[name] = {"offset": off, "shape": list(a.shape)}
        blobs.append(a.tobytes())
        off += a.size

    for i, m in enumerate(members):
        net = m.net
        tw = net.towers[0]
        for j in range(3):
            w, b = fold(tw.c[j][0], tw.c[j][1])
            put(f"m{i}.conv{j}.w", w)
            put(f"m{i}.conv{j}.b", b)
        put(f"m{i}.fc.w", tw.fc[2].weight.detach().numpy())
        put(f"m{i}.fc.b", tw.fc[2].bias.detach().numpy())
        put(f"m{i}.head0.w", net.head[0].weight.detach().numpy())
        put(f"m{i}.head0.b", net.head[0].bias.detach().numpy())
        put(f"m{i}.head2.w", net.head[2].weight.detach().numpy())
        put(f"m{i}.head2.b", net.head[2].bias.detach().numpy())
    (out / "gaze-net.bin").write_bytes(b"".join(blobs))
    meta = {
        "members": len(members), "tensors": index, "patch": [PATCH_H, PATCH_W], "input": [IN_H, IN_W],
        "corner_dist": 36.0,
        "license": "CC BY-NC-SA 4.0 (trained on MPIIFaceGaze); github.com/yago-mendoza/elementfromgaze",
    }
    (out / "gaze-net.json").write_text(json.dumps(meta))

    # reference outputs: random-ish but eye-like patches are not needed, any input checks the arithmetic
    rng = np.random.default_rng(0)
    P = rng.integers(0, 256, (3, 2, PATCH_H, PATCH_W)).astype(np.uint8)
    oy, ox = (PATCH_H - IN_H) // 2, (PATCH_W - IN_W) // 2
    x = torch.from_numpy(np.ascontiguousarray(_norm(P)[..., oy:oy + IN_H, ox:ox + IN_W]))
    ref = []
    with torch.inference_mode():
        for m in members:
            net = m.net
            f = net.towers[0](x.reshape(6, 1, IN_H, IN_W)).reshape(3, 128)
            ref.append(net.head(f).numpy().tolist())
    (out / "gaze-net.check.json").write_text(json.dumps({"patches": P.tolist(), "angles": ref}))
    print(f"{len(members)} redes, {off} números ({off * 4 / 1e6:.2f} MB) -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
