# Runbook: the GPU runs, university cluster (A100)

Everything below runs from the repo root on the `kernel-work` branch. Fill the
SLURM placeholders (partition, account, module names) from the cluster docs;
everything else is ready.

## Environment

torch >= 2.4 with CUDA build ships its own matching triton on Linux; no separate
triton install is needed. Sanity once, on a GPU node:

    python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"

If the cluster's torch is older or CPU-only, a venv with
`pip install torch --index-url https://download.pytorch.org/whl/cu121` (or the
cluster's blessed wheel) is enough; the code needs only torch + triton + scipy.

## The three runs

1. Correctness gate (minutes):

       python kernels/test_correctness.py

   Expect all four gates to pass. Note: on A100 the triton-vs-rewrite gate is
   1e-3, not 1e-5 — both cuBLAS and tl.dot use TF32 for fp32 on Ampere and
   round differently. That is expected and documented, not a bug.

2. The sweep, both dtypes (about 10 minutes each):

       python kernels/bench.py --out kernels/results/sweep_fp32.csv
       python kernels/bench.py --out kernels/results/sweep_bf16.csv --dtype bfloat16

   bf16 is the arm that matters on A100 (native, and it is what the training
   pipeline uses under autocast). Meta JSON records device + TF32 state.

3. One profile, for the README (optional but worth it):

       python -c "
       import sys, torch; sys.path.insert(0,'src'); sys.path.insert(0,'kernels')
       from graph_transformer import LanczosSpectralMix
       from spectral_mix_opt import mix_eager, mix_rewrite
       mod = LanczosSpectralMix(d_model=128, num_filters=8).cuda().eval()
       h = torch.randn(50_000,128,device='cuda'); ev=torch.sort(torch.rand(16,device='cuda')).values
       V,_ = torch.linalg.qr(torch.randn(50_000,16,device='cuda')); V=V.double()
       from torch.profiler import profile, ProfilerActivity
       for fn,name in ((mix_eager,'eager'),(mix_rewrite,'rewrite')):
           with torch.no_grad():
               for _ in range(5): fn(h,ev,V,mod.phi,mod.proj)
               with profile(activities=[ProfilerActivity.CUDA]) as pr:
                   for _ in range(20): fn(h,ev,V,mod.phi,mod.proj)
           print('===',name); print(pr.key_averages().table(sort_by='cuda_time_total', row_limit=8))
       " | tee kernels/results/profile.txt

## SLURM template

    #!/bin/bash
    #SBATCH --job-name=specmix-bench
    #SBATCH --partition=<GPU_PARTITION>        # fill in
    #SBATCH --gres=gpu:a100:1
    #SBATCH --cpus-per-task=8
    #SBATCH --time=00:30:00
    # module load <cluster pytorch module>     # or activate the venv
    cd $SLURM_SUBMIT_DIR
    python kernels/test_correctness.py
    python kernels/bench.py --out kernels/results/sweep_fp32.csv
    python kernels/bench.py --out kernels/results/sweep_bf16.csv --dtype bfloat16

## What A100 changes about the predictions

- HBM at ~1.6-2 TB/s (vs the ~0.5 TB/s the spec's arithmetic assumed): every
  bandwidth number shrinks ~3-4x while launch overhead does not, so the
  launch-bound region extends to LARGER n. Expect the rewrite's win to be
  dominated by eliminating the (n,1024) materialisation, and expect the
  hand-written kernel's margin over torch.compile to be thinner than on a T4.
  If compile matches the kernel, that is the reportable result.
- TF32 is ON by default for fp32 matmuls. The bench records the flag; leave it
  on (it is the realistic deployment default) and say so in the README numbers.

## Bring back

- kernels/results/sweep_fp32.csv, sweep_bf16.csv, *_meta.json, profile.txt
- tick the two open boxes in kernels/README.md, fill the measured numbers in,
  then merge kernel-work -> main and push (the repo is linked from the CV).

## Also quick on the cluster (CPU-fine, no GPU needed)

    python quantization/ablate.py --ckpt models/spectral_refiner_k4_eps_0_03.pt
    python quantization/ablate.py --ckpt models/spectral_refiner_k8_eps_0_03.pt
    python quantization/ablate.py --ckpt models/spectral_refiner_k32_eps_0_03.pt

closes the "other three checkpoints" box in quantization/README.md.
