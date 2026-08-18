# Weight quantization of the deployed refiner: error model, measurement, ablation

Post-training, weight-only, symmetric fake quantization of the shipped
`spectral_refiner_*` checkpoints, written by hand (~20 lines of scheme code in
`quantize.py`) so the scheme is fully controlled and the error model is legible.

## The error model, stated before measuring

For symmetric b-bit quantization the rounding error is ~U[-s/2, s/2] with
s = max|W|/(2^(b-1)-1), giving

    SQNR ~= 12 (2^(b-1)-1)^2 / numel * ||W||_F^2 / max|W|^2

Three predictions follow: (1) ~6 dB lost per bit removed, (2) sensitivity is
governed by dynamic range max|W|/RMS(W), not tensor size or spectrum,
(3) per-channel scales recover most of what outlier channels cost.

## What was measured (all four checkpoints, k = 4/8/16/32)

`report.py`, per weight tensor: measured SQNR under {int8, int4} x {per-tensor,
per-channel}, the closed-form predicted SQNR, and dynamic range. The
predictions hold: int8 per-tensor sits ~40 dB, int4 ~15 dB (the 6 dB/bit rule),
per-channel adds 4-5 dB, and the worst layers are exactly the highest-dynamic-
range ones. Weights: 0.72 MB fp32 -> 0.18 MB int8 -> 0.09 MB int4.

`ablate.py`, encoder output drift on fixed synthetic probe graphs (proxy metric,
runnable without the benchmark suite), k16 checkpoint:

| scheme | whole-model drift |
|---|---|
| int8 per-tensor  | 0.52% |
| int8 per-channel | 0.34% |
| int4 per-tensor  | 8.7%  |
| int4 per-channel | 5.8%  |

Single-layer ablation (int4 pt) ranks `in_proj` as the most sensitive tensor
(6.3% drift on its own), then the first layer's local/FFN blocks: early-layer
error compounds through depth, late layers are cheap to quantize.

**Invariance is preserved under quantization, measured rather than assumed:**
vertex-permutation and degenerate-eigenspace-rotation invariance hold to
~1e-7 on the quantized model under every scheme. Quantization commutes with
the model's symmetry structure, as it should (the schemes act on weights,
the invariances are architectural), and now there is a number saying so.

## Honest scope notes

- Drift on synthetic probes is a functional proxy, not task accuracy. The
  task-level ablation (cut ratio vs METIS through the real pipeline) plugs in
  via `eval_fn` and needs the benchmark graphs: a longer, offline run.
- One tie-break difference per ~8k elements against
  `torch.fake_quantize_per_tensor_affine` (round-half cases), max one
  quantization step: found by the sanity check, bounded, documented.
- Weight-only. Activation quantization needs calibration data and is where
  outliers genuinely bite; out of scope here, noted as future work.

## Files

- `quantize.py`   - scheme + SQNR/dynamic-range stats + torch-reference sanity
- `report.py`     - per-layer tables for a checkpoint (CSV in results/)
- `ablate.py`     - whole-model + per-layer drift, invariance checks (CSV)

## Status

- [x] quantizer verified against the torch reference (tie-break-aware)
- [x] per-layer SQNR/range report, all four shipped checkpoints
- [x] drift ablation + invariance preservation, k16
- [ ] drift ablation on the other three checkpoints (one command each)
- [ ] task-level ablation through the benchmark pipeline (offline run)
