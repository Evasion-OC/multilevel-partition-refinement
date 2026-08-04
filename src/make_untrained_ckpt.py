"""Write an UNTRAINED control checkpoint for the search/learning decomposition of Appendix H.

Loads a trained checkpoint, re-instantiates SpectralActorCritic from that checkpoint's OWN
stored constructor arguments, and replaces only `model_state` with the fresh random
initialisation. Every other key (k, n_eigs, coarsen_min, ml_refine_max_n, epsilon, ...) is
copied unchanged, so `lev1.py --model <out>` runs the identical scaffold, size gate and
guards. The policy weights are the only factor that changes -- which is the control condition
Appendix H requires.

Usage:
  PYTHONPATH=src python src/make_untrained_ckpt.py \
      --model models/spectral_refiner_k4_eps_0_03.pt \
      --out   models/untrained_ctrl_seed0.pt --seed 0
"""
import argparse
import torch

from refiner import SpectralActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained checkpoint to mirror")
    ap.add_argument("--out", required=True, help="destination for the untrained control")
    ap.add_argument("--seed", type=int, default=0, help="init seed (recorded in the checkpoint)")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)

    torch.manual_seed(args.seed)
    model = SpectralActorCritic(
        k=ck["k"], in_dim=ck["in_dim"], d_model=ck["d_model"], n_heads=ck["n_heads"],
        n_layers=ck["n_layers"], pe_dim=ck["pe_dim"],
        num_filters=ck.get("num_filters", 8), pe_mode=ck.get("pe_mode", "diag"),
        use_local=ck.get("use_local", True),
        global_kind=ck.get("global_kind", "attn"), pe_kind=ck.get("pe_kind", "stable"),
    )

    trained_state = ck["model_state"]
    fresh_state = model.state_dict()
    assert set(fresh_state) == set(trained_state), "architecture mismatch: key sets differ"
    for key in fresh_state:
        assert fresh_state[key].shape == trained_state[key].shape, f"shape mismatch at {key}"

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == ck.get("param_count", n_params), "parameter count differs from trained"

    max_delta = max((fresh_state[k] - trained_state[k]).abs().max().item()
                    for k in fresh_state if fresh_state[k].is_floating_point())
    assert max_delta > 0.0, "fresh weights are identical to the trained ones"

    out_ck = dict(ck)
    out_ck["model_state"] = fresh_state
    out_ck["untrained_control"] = True
    out_ck["untrained_seed"] = args.seed
    out_ck["mirrors_checkpoint"] = args.model
    torch.save(out_ck, args.out)

    print(f"wrote {args.out}")
    print(f"  mirrors        {args.model}")
    print(f"  k={out_ck['k']} d_model={out_ck['d_model']} n_eigs={out_ck.get('n_eigs')} "
          f"coarsen_min={out_ck.get('coarsen_min')} ml_refine_max_n={out_ck.get('ml_refine_max_n')}")
    print(f"  params         {n_params} (matches trained)")
    print(f"  init seed      {args.seed}")
    print(f"  max |w_new - w_trained|  {max_delta:.4f}")


if __name__ == "__main__":
    main()
