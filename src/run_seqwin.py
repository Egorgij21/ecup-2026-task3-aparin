#!/usr/bin/env python
"""
USER-SPLIT 5-fold CV for a BIDIRECTIONAL windowed encoder.

    python src/run_seqwin.py --config configs/e0130_biwin_usercv.yaml

A separate entrypoint on purpose.  This is a different CV protocol and a different training
regime from src/run_seq.py, and keeping it out of that file is what stops the 18 logged
`nn_seq` experiments from silently changing (CLAUDE.md rule 3 / rule 8).

WHAT IS BEING MEASURED, AND WHAT IT IS NOT
  Folds are 5 disjoint random slices of the 250,000 users.  A model is fit on 4 slices and
  scored on the held-out slice, at the SAME anchors it trained on.  So the generalisation
  measured here is *to unseen users within a seen time period*.

  The leaderboard asks the opposite: the same 250k users, a period nobody has seen.  These
  scores are therefore NOT on the same scale as the frozen-fold numbers in experiments.csv and
  must never be compared to them -- an anchor whose calendar window the model trained on
  (through other users) is an easier problem.  Reported here as its own quantity.

DESIGN, as specified
  * bidirectional attention -- no causal mask.  Safe by construction: the window ENDS at the
    anchor, so nothing after it exists inside the window to attend to.  Asserted, not assumed.
  * one target per forward pass, at N anchors per user.  e0115 measured the cost of giving up
    dense supervision at +0.00012 (0.6 sigma), so this trade is free.
  * zero padding to a fixed context length L, left-padded, with an attention key-padding mask
    so the pad cannot be attended to.  L defaults to the full history the TEST anchor will
    have (409 days), exactly as the submission would see it.
  * positional embeddings are ANCHOR-RELATIVE: index L-1 is always the anchor day, so the
    encoding carries no absolute calendar position.  That preserves the one structural edge
    seq_transfer.py measured (+0.00065 RMSLE per 100 days of cut-off gap, vs the GBDT's
    +0.00428).
  * no fixed epoch count.  The held-out-user score is computed after EVERY epoch and the whole
    curve is reported, so the epoch budget is an observation rather than a setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import rmsle, score_all              # noqa: E402
from seqdata import HORIZON, build_seq_panel      # noqa: E402
from seqnet import DayEncoder                     # noqa: E402

MIN_HISTORY_DAYS = 90
LAST_CLEAN_ANCHOR = date(2025, 10, 16)     # A + 30 < 2025-11-16, the guaranteed-activity zone


def log(m: str) -> None:
    print(m, flush=True)


class BiWindow(nn.Module):
    """Bidirectional transformer over a fixed-length window ending at the anchor."""

    def __init__(self, c_in: int, mu, sd, L: int, d: int = 128, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.enc = DayEncoder(c_in, d, mu, sd)
        # Anchor-relative: position L-1 IS the anchor for every sample, so this embedding
        # encodes "how many days before the prediction date", never a calendar date.
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.enc(x).transpose(1, 2) + self.pos          # (B, L, d)
        h = self.tr(h, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, -1])).squeeze(-1)   # readout at the anchor day


def gather_windows(Xg: torch.Tensor, users: torch.Tensor, anchors: torch.Tensor,
                   L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, C, L) window ending at each anchor, left-zero-padded, plus the pad mask."""
    idx = anchors[:, None] - (L - 1) + torch.arange(L, device=Xg.device)[None, :]
    valid = idx >= 0
    Xu = Xg[users]                                            # (B, C, n_days)
    g = torch.gather(Xu, 2, idx.clamp(min=0)[:, None, :].expand(-1, Xu.shape[1], -1))
    return g * valid[:, None, :], ~valid


@torch.no_grad()
def assert_window_causal(anchors: torch.Tensor, L: int) -> None:
    """The window must end AT the anchor -- never one day past it.

    This is the only leakage surface a bidirectional model has: attention inside the window is
    unrestricted, so correctness rests entirely on the window's right edge.  Checked explicitly
    because it is a single off-by-one away from feeding the model its own target.
    """
    idx = anchors[:, None] - (L - 1) + torch.arange(L, device=anchors.device)[None, :]
    assert int(idx.max().item()) == int(anchors.max().item()), "window reaches past its anchor"
    assert torch.equal(idx[:, -1], anchors), "last window position is not the anchor day"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"]
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\n=== {exp_id} [USER-SPLIT CV] : {cfg['change']} ===")
    log(f"    bidirectional {cfg.get('n_layers', 4)}L x {cfg.get('n_heads', 4)}H d={cfg.get('d_model', 128)}  "
        f"epochs={cfg['epochs']} lr={cfg['lr']} seed={cfg['seed']}  device={device}")
    torch.backends.cudnn.benchmark = True

    sp = build_seq_panel(derived=bool(cfg.get("derived_channels", False)),
                         ranks=bool(cfg.get("rank_channels", False)))
    L = int(cfg.get("context_len", sp.n_days))
    Xg = torch.from_numpy(sp.X).to(device)
    if device == "cuda":
        log(f"    gpu resident: {torch.cuda.memory_allocated() / 1e9:.2f} GB  context L={L}")

    # anchor grid: >=90d history, and every target window clear of the guaranteed-activity zone
    stride = int(cfg.get("anchor_stride", 7))
    anchors = list(range(MIN_HISTORY_DAYS - 1, sp.idx(LAST_CLEAN_ANCHOR) + 1, stride))
    log(f"    {len(anchors)} anchors per user: {sp.day(anchors[0])} .. {sp.day(anchors[-1])} "
        f"(stride {stride}d, all target windows clean)")

    # (user, anchor) pairs, restricted to the frozen population rule at that anchor
    uu, aa, yy = [], [], []
    for t in anchors:
        m = sp.pop[:, t]
        uu.append(np.flatnonzero(m).astype(np.int32))
        aa.append(np.full(int(m.sum()), t, np.int32))
        yy.append(sp.Y[m, t])
    uu, aa, yy = np.concatenate(uu), np.concatenate(aa), np.concatenate(yy).astype(np.float32)
    log(f"    {uu.size:,} (user, anchor) pairs   mean target log1p {yy.mean():.4f}")

    rng = np.random.default_rng(int(cfg["seed"]))
    ufold = rng.integers(0, 5, sp.n_users)          # 5 disjoint random user slices
    pair_fold = ufold[uu]
    Uall = torch.from_numpy(uu.astype(np.int64)).to(device)
    Aall = torch.from_numpy(aa.astype(np.int64)).to(device)
    Yall = torch.from_numpy(yy).to(device)
    assert_window_causal(Aall[:4096], L)

    mu, sd = sp.norm_stats(sp.idx(LAST_CLEAN_ANCHOR))
    bs = int(cfg.get("batch_pairs", 512))
    per_fold, curves = [], []

    for k in args.folds:
        tr = np.flatnonzero(pair_fold != k)
        va = np.flatnonzero(pair_fold == k)
        tri = torch.from_numpy(tr.astype(np.int64)).to(device)
        vai = torch.from_numpy(va.astype(np.int64)).to(device)
        torch.manual_seed(int(cfg["seed"]) + k)

        model = BiWindow(sp.n_ch, torch.from_numpy(mu), torch.from_numpy(sd), L,
                         d=int(cfg.get("d_model", 128)), n_layers=int(cfg.get("n_layers", 4)),
                         n_heads=int(cfg.get("n_heads", 4)),
                         dropout=float(cfg.get("dropout", 0.1))).to(device)
        epochs = int(cfg["epochs"])
        steps = math.ceil(tr.size / bs)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                                weight_decay=float(cfg.get("weight_decay", 1e-4)))
        warm = max(1, int(0.03 * epochs * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, epochs * steps - warm))))
        log(f"\n    fold {k}: {tr.size:,} train pairs ({np.unique(uu[tr]).size:,} users)  |  "
            f"{va.size:,} held-out pairs ({np.unique(uu[va]).size:,} unseen users)  "
            f"{epochs} epochs x {steps} steps")

        def amp():
            return (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda"
                    else torch.autocast(device_type="cpu", enabled=False))

        g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]) + k)
        curve = []
        for ep in range(epochs):
            model.train()
            perm = tri[torch.randperm(tr.size, generator=g).to(device)]
            tot = 0.0
            for i in range(steps):
                b = perm[i * bs:(i + 1) * bs]
                xb, pad = gather_windows(Xg, Uall[b], Aall[b], L)
                with amp():
                    out = model(xb.float(), pad)
                loss = ((out.float() - Yall[b]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += float(loss.item()) * b.numel()
            # held-out users, scored after EVERY epoch -- the epoch budget is observed, not set
            model.eval()
            pr = torch.empty(va.size, device=device)
            with torch.no_grad():
                for i in range(0, va.size, 2048):
                    b = vai[i:i + 2048]
                    xb, pad = gather_windows(Xg, Uall[b], Aall[b], L)
                    with amp():
                        pr[i:i + b.numel()] = model(xb.float(), pad).float()
            v = float(torch.sqrt(((pr - Yall[vai]) ** 2).mean()).item())
            curve.append(v)
            log(f"      epoch {ep + 1:>3d}/{epochs}  train {math.sqrt(tot / tr.size):.5f}  "
                f"HELD-OUT USERS {v:.5f}  lr {sched.get_last_lr()[0]:.2e}  "
                f"[{(time.time() - t0) / 60:.1f}m]")

        best_ep = int(np.argmin(curve)) + 1
        per_fold.append(min(curve)); curves.append(curve)
        log(f"    fold {k}: best {min(curve):.5f} at epoch {best_ep}  (last {curve[-1]:.5f})")

        if k == args.folds[-1]:
            yv = Yall[vai].cpu().numpy(); pv = pr.cpu().numpy(); av = aa[va]
            log(f"\n    per-anchor breakdown on unseen users (fold {k}, final epoch):")
            for t in anchors[::4]:
                m = av == t
                log(f"      {sp.day(t)}  n={int(m.sum()):>7,}  rmsle={rmsle(np.expm1(yv[m]), np.expm1(pv[m])):.5f}")

    pf = np.array(per_fold)
    runtime = (time.time() - t0) / 60
    log(f"\n  USER-SPLIT CV (unseen users) = {pf.mean():.5f} +/- {pf.std():.5f}")
    log(f"  folds {np.round(pf, 5).tolist()}")
    log(f"  best epoch per fold: {[int(np.argmin(c)) + 1 for c in curves]} of {cfg['epochs']}")
    log(f"  runtime {runtime:.1f} min")
    log("\n  NOT COMPARABLE to experiments.csv: those score the same users at a FUTURE anchor;")
    log("  this scores unseen users at anchors whose calendar window the model trained on.")

    out = ROOT / "reports" / "eda" / f"{exp_id}_usercv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"exp_id": exp_id, "config": args.config,
                               "date": datetime.now().isoformat(timespec="seconds"),
                               "protocol": "5-fold split by user_id; unseen-user RMSLE",
                               "folds": pf.tolist(), "mean": float(pf.mean()),
                               "curves": curves, "anchors": anchors,
                               "runtime_min": round(runtime, 1)}, indent=2))
    log(f"  wrote reports/eda/{exp_id}_usercv.json")


if __name__ == "__main__":
    main()
