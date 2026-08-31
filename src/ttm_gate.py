#!/usr/bin/env python
"""Gate for the TTM (TinyTimeMixer) backbone — install, download, forward-pass shape.

IDEAS.md §I29 ranks TTM the most architecturally orthogonal candidate on the shortlist: an
MLP-Mixer with NO ATTENTION AT ALL, against MOMENT's T5 encoder (e0915) and Chronos-Bolt's
T5 encoder-decoder (e0919). Both incumbents are transformers; both land at r 0.96-0.98 with the
champion. The e0391 bar table says the CHEAP axis is decorrelation, not strength -- the bar falls
0.6553 -> 0.6441 -> 0.6324 as r goes 0.98 -> 0.96 -> 0.94 -- and e0392 showed LoRA moves r the
WRONG way (0.979 -> 0.98266 while rho_partial fell). A no-attention backbone is the one lever
aimed at the axis that is actually cheap.

This script installs nothing by itself except through the caller and DECIDES NOTHING about the
model -- it only answers "can this run at all", in seconds, before a GPU job is queued. The
session already lost hours to three fresh-env failures that a gate would have caught in 25s.

Run on `compute` / `computeshort`: apini has no outbound internet, so weights must be pre-cached
here into $HF_HOME, after which the GPU job runs with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import sys
import traceback


def line(m):
    print(m, flush=True)


def main() -> int:
    bad = 0

    line("=== 1. imports ===")
    try:
        import torch
        line(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception as e:
        line(f"  torch MISSING: {e}"); return 1

    have_tsfm = False
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction  # noqa: F401
        import tsfm_public
        line(f"  tsfm_public {getattr(tsfm_public, '__version__', '?')}  (native TTM loader)")
        have_tsfm = True
    except Exception as e:
        line(f"  tsfm_public MISSING ({type(e).__name__}) -- will try the transformers fallback")

    try:
        from transformers import PatchTSMixerModel  # noqa: F401
        import transformers
        line(f"  transformers {transformers.__version__}  PatchTSMixerModel OK")
    except Exception as e:
        line(f"  transformers/PatchTSMixerModel MISSING: {e}"); bad += 1

    line("\n=== 2. fetch weights (needs internet -- run on compute) ===")
    REPO = "ibm-granite/granite-timeseries-ttm-r2"
    got = None
    for rev in (None, "512-96-r2", "main"):
        try:
            if have_tsfm:
                from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction as M
                kw = {"revision": rev} if rev else {}
                mdl = M.from_pretrained(REPO, **kw)
            else:
                from transformers import PatchTSMixerModel as M
                kw = {"revision": rev} if rev else {}
                mdl = M.from_pretrained(REPO, **kw)
            got = (rev, mdl)
            line(f"  LOADED {REPO}  revision={rev!r}")
            break
        except Exception as e:
            line(f"  revision={rev!r} failed: {type(e).__name__}: {str(e)[:160]}")
    if got is None:
        line("  -> TTM could not be loaded. The direction is blocked on packaging, not on the idea.")
        return 2

    rev, mdl = got
    n = sum(p.numel() for p in mdl.parameters())
    line(f"  parameters {n:,}")
    cfg = getattr(mdl, "config", None)
    for k in ("context_length", "prediction_length", "patch_length", "d_model", "num_layers",
              "num_input_channels"):
        if cfg is not None and hasattr(cfg, k):
            line(f"    config.{k} = {getattr(cfg, k)}")

    line("\n=== 3. forward pass on a panel-shaped batch ===")
    try:
        import torch
        ctx = int(getattr(cfg, "context_length", 512))
        x = torch.randn(4, ctx, 1)                      # [B, T, C] -- TTM's layout
        mdl.eval()
        with torch.no_grad():
            try:
                out = mdl(past_values=x, output_hidden_states=True)
            except TypeError:
                out = mdl(x)
        keys = [k for k in ("last_hidden_state", "backbone_hidden_state", "hidden_states",
                            "prediction_outputs", "decoder_hidden_state") if hasattr(out, k)]
        line(f"  output fields present: {keys}")
        for k in keys:
            v = getattr(out, k)
            if hasattr(v, "shape"):
                line(f"    {k}: {tuple(v.shape)}")
            elif isinstance(v, (list, tuple)) and v and hasattr(v[0], "shape"):
                line(f"    {k}: {len(v)} tensors, last {tuple(v[-1].shape)}")
        line("  -> a per-PATCH hidden state is what the e0915 recipe needs "
             "([B, C, n_patches, d] or [B, n_patches, d]).")
    except Exception:
        line("  forward FAILED:"); traceback.print_exc(); bad += 1

    line(f"\n=== GATE {'PASS' if bad == 0 else 'FAIL'} (context length {ctx}, "
         f"SEQ_LEN in run_tsfm_gru.py is 512) ===")
    return 0 if bad == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
