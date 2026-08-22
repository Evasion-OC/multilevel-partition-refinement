"""Write fake-quantized copies of a shipped checkpoint for the task-level benchmark.

Quantizes the ENCODER's 2-D weight tensors (what the artifact studied); actor/critic heads stay
fp32. The files load through the normal path (refiner.load_spectral_actor_critic), so
src/benchmark.py runs them unchanged:

    python quantization/export_quantized.py --ckpt models/spectral_refiner_k16_eps_0_03.pt
    python src/benchmark.py --model models/quantized/spectral_refiner_k16_eps_0_03_int8_pc.pt ...
"""
import argparse, copy, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from quantize import fake_quantize

SCHEMES = [(8, False), (8, True), (4, False), (4, True)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/spectral_refiner_k16_eps_0_03.pt")
    ap.add_argument("--out-dir", default="models/quantized")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    for bits, pc in SCHEMES:
        q = copy.deepcopy(ck); n = 0
        for k, v in q["model_state"].items():
            if k.startswith("encoder.") and k.endswith("weight") and v.dim() >= 2:
                q["model_state"][k] = fake_quantize(v.float(), bits=bits, per_channel=pc).to(v.dtype); n += 1
        q["quantization"] = {"bits": bits, "per_channel": pc, "tensors": n, "scope": "encoder 2-D weights"}
        path = os.path.join(args.out_dir, f"{stem}_int{bits}_{'pc' if pc else 'pt'}.pt")
        torch.save(q, path)
        print(f"{path}: {n} encoder weight tensors fake-quantized to int{bits} {'per-channel' if pc else 'per-tensor'}")

if __name__ == "__main__":
    main()
