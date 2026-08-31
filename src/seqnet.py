#!/usr/bin/env python
"""
Causal sequence models over the daily panel, for the `seq` approach.

Every architecture here obeys one contract:

    forward(x) -> (B, T) tensor whose value at position t is the model's estimate of
    log1p(sum gmv over [t+1, t+30]) using ONLY days <= t.

That contract is what makes dense supervision legal, and it is machine-checked by
`assert_causal` rather than argued for -- this repo has already lost a day to a look-ahead
that nobody could see by reading the code (DATA.md §10), so the guard runs on every job.

Three backbones, one per experiment:

  tcn         dilated causal convolutions.  Receptive field 511 days at 8 blocks, covers the
              full history.  Cheapest, and a convolution IS a windowed aggregate -- the same
              primitive the feature set is built from, but learned and composable.
  gru         recurrent, unbounded receptive field.  EBES and RQ-Reg independently land on
              GRU/LSTM at this data scale (reports/papers_part3_sequence.md), so this is the
              literature's default rather than ours.
  xformer     causal self-attention with ALiBi.  Deliberately NO absolute positional
              embedding: an absolute encoding would let the model key on calendar position,
              and the test anchor sits 120 days past the last clean training anchor, which is
              exactly where an absolute encoding extrapolates badly.  ALiBi's distance-only
              bias keeps the whole model translation-invariant in time.

Normalisation is by FIXED per-channel constants passed in from the caller (fit on training
days only).  BatchNorm would be a leak here in a way that is easy to miss: its batch statistics
are pooled over the time axis, so the value at position t would be normalised using moments
that include positions > t of the same user.  LayerNorm over the channel axis at each position
has no such path and is what every block below uses.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# --------------------------------------------------------------------------- input encoding
class DayEncoder(nn.Module):
    """(B, C, T) raw channels -> (B, d, T) embeddings.

    Calendar channels are optional and OFF by default.  Day-of-week is the only calendar
    signal that is safe by construction (it repeats every 7 days, so the test window is not an
    extrapolation); anything indexed on absolute time or day-of-year is a separate, riskier
    experiment -- DATA.md §5.4 measures the test window's seasonality as real, but the model
    has never seen February in target position, so a learned day-of-year term would be pure
    extrapolation.
    """

    def __init__(self, c_in: int, d: int, mu: Tensor, sd: Tensor, dow: bool = False):
        super().__init__()
        self.register_buffer("mu", mu.view(1, -1, 1))
        self.register_buffer("sd", sd.view(1, -1, 1))
        self.dow = dow
        self.proj = nn.Conv1d(c_in + (2 if dow else 0), d, 1)

    def forward(self, x: Tensor, dow_feat: Tensor | None = None) -> Tensor:
        x = (x - self.mu) / self.sd
        if self.dow:
            assert dow_feat is not None, "dow=True but no calendar tensor passed"
            x = torch.cat([x, dow_feat.expand(x.shape[0], -1, -1)], dim=1)
        return self.proj(x)


class ChanNorm(nn.Module):
    """LayerNorm over the channel axis at each position -- no mixing along time."""

    def __init__(self, d: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:          # (B, d, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


# ------------------------------------------------------------------------------- backbones
class TCNBlock(nn.Module):
    def __init__(self, d: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = dilation * (k - 1)                # left-pad only == causal
        self.norm = ChanNorm(d)
        self.conv = nn.Conv1d(d, d, k, dilation=dilation)
        self.pw = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.conv(F.pad(h, (self.pad, 0)))
        h = self.pw(F.gelu(h))
        return x + self.drop(h)


class TCNBackbone(nn.Module):
    def __init__(self, d: int, n_blocks: int = 8, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TCNBlock(d, k, 2 ** i, dropout) for i in range(n_blocks)])
        self.out_norm = ChanNorm(d)
        self.receptive_field = 1 + (k - 1) * (2 ** n_blocks - 1)

    def forward(self, x: Tensor) -> Tensor:
        for b in self.blocks:
            x = b(x)
        return self.out_norm(x)


class RecurrentBackbone(nn.Module):
    """GRU / LSTM / vanilla RNN behind one interface.

    All three are causal by construction (forward-only recurrence). The vanilla RNN is included
    as a deliberate weak control: if it lands near the GRU, the gating is not doing the work.
    """

    KIND = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}

    def __init__(self, d: int, kind: str = "gru", n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        kw = dict(num_layers=n_layers, batch_first=True,
                  dropout=dropout if n_layers > 1 else 0.0)
        if kind == "rnn":
            kw["nonlinearity"] = "tanh"
        self.rnn = self.KIND[kind](d, d, **kw)
        self.out_norm = ChanNorm(d)
        self.receptive_field = 10 ** 6

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.rnn(x.transpose(1, 2))
        return self.out_norm(h.transpose(1, 2))


def _rope(q: Tensor, k: Tensor, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """Rotary position embedding: relative by construction, so still translation-invariant."""
    T, D = q.shape[-2], q.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=q.device).float() / D))
    fr = torch.outer(torch.arange(T, device=q.device).float(), inv)
    cos, sin = fr.cos()[None, None], fr.sin()[None, None]

    def rot(x: Tensor) -> Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    return rot(q), rot(k)


class GRUBackbone(nn.Module):
    def __init__(self, d: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(d, d, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.out_norm = ChanNorm(d)
        self.receptive_field = 10 ** 6

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.rnn(x.transpose(1, 2))           # (B, T, d), causal by construction
        return self.out_norm(h.transpose(1, 2))


class CNNGRUBackbone(nn.Module):
    """Dilated causal convolutions first, then a GRU over their output.

    The two winners so far fail differently: the TCN (e0100, 1.76775) has a fixed 511-day
    receptive field built from local patterns, the GRU (e0101, 1.76458) has unbounded reach but
    must squeeze everything through one hidden state.  Their log-predictions correlate at
    0.9955 -- the lowest pair inside the seq family -- which says they really are computing
    different things.  Stacking them gives the recurrence pre-summarised local structure
    instead of raw daily noise, so the hidden state carries pattern rather than level.

    Causality is preserved by construction (left-padded convolutions, then a forward-only GRU)
    and checked by `assert_causal` like every other backbone here.
    """

    def __init__(self, d: int, n_conv: int = 4, n_gru: int = 2, k: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = nn.ModuleList([TCNBlock(d, k, 2 ** i, dropout) for i in range(n_conv)])
        self.rnn = nn.GRU(d, d, num_layers=n_gru, batch_first=True,
                          dropout=dropout if n_gru > 1 else 0.0)
        self.out_norm = ChanNorm(d)
        self.receptive_field = 10 ** 6

    def forward(self, x: Tensor) -> Tensor:
        for b in self.conv:
            x = b(x)
        h, _ = self.rnn(x.transpose(1, 2))
        return self.out_norm(h.transpose(1, 2))


def _alibi_slopes(n_heads: int) -> Tensor:
    """Standard ALiBi geometric slope schedule (Press et al.)."""
    def pow2_slopes(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(pow2_slopes(n_heads))
    closest = 2 ** math.floor(math.log2(n_heads))
    s = pow2_slopes(closest) + pow2_slopes(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(s)


class XformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads, self.dh = n_heads, d // n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, bias: Tensor, rope: bool = False) -> Tensor:
        B, T, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.n_heads, self.dh).permute(2, 0, 3, 1, 4)
        if rope:
            q, k = _rope(q, k)
        # bias already encodes the causal mask (-inf above the diagonal) plus ALiBi distance
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[:, :T, :T])
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, d)))
        return x + self.drop(self.mlp(self.norm2(x)))


class XformerBackbone(nn.Module):
    def __init__(self, d: int, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1,
                 max_len: int = 512, pos: str = "alibi"):
        super().__init__()
        self.pos = pos
        # "learned" is the vanilla transformer: absolute positional embeddings, which key on
        # calendar position. The test anchor sits 120 days past the last clean training anchor,
        # so this is the one variant with a real extrapolation risk -- included precisely to
        # measure whether that risk is worth anything against ALiBi and RoPE.
        self.pemb = nn.Parameter(torch.zeros(1, max_len, d)) if pos == "learned" else None
        if self.pemb is not None:
            nn.init.normal_(self.pemb, std=0.02)
        self.blocks = nn.ModuleList([XformerBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.receptive_field = 10 ** 6
        t = torch.arange(max_len)
        dist = (t[None, :] - t[:, None]).float()                     # j - i
        causal = torch.where(dist <= 0, 0.0, float("-inf"))
        slope = _alibi_slopes(n_heads)[:, None, None] if pos == "alibi" else 0.0
        self.register_buffer("bias", causal[None] + slope * dist, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)
        if self.pemb is not None:
            h = h + self.pemb[:, : h.shape[1]]
        for b in self.blocks:
            h = b(h, self.bias, rope=(self.pos == "rope"))
        return self.out_norm(h).transpose(1, 2)


# ----------------------------------------------------------------------------- full model
class ChannelIndependentGRU(nn.Module):
    """One recurrence PER CHANNEL, fused only after the sequence is consumed.

    `BACKLOG.md` states the rule this is built to satisfy: useful diversity needs "a different
    INDUCTIVE BIAS over the SAME information", not less information.  Every other seq model here
    projects the 13 channels into one d-dim vector per day (`DayEncoder`) and then runs a single
    recurrence, so channels are mixed BEFORE any temporal processing.  This inverts that: each
    channel is carried through its own recurrence and they may not interact until the fusion
    head.  Same inputs, same causality, different bias.

    Contrast with e0400-e0402, which split the INFORMATION and failed for a measured reason --
    behavioural-only barely beat the naive baseline (1.912 vs 1.927) because the behavioural
    channels are a modifier on the monetary signal rather than a standalone predictor.  Here the
    information stays whole and only the processing is split, which is the version that argument
    does not touch.

    Implementation note: the per-channel GRUs SHARE weights and are distinguished by a learned
    channel embedding, so all C recurrences run as ONE cuDNN call at batch B*C rather than C
    sequential calls.  That is the channel-independent design the TS foundation models use
    (MOMENT and Chronos are both channel-independent), and it keeps this ~1 GRU in cost instead
    of 13.
    """

    def __init__(self, c_in: int, d: int, n_layers: int = 2, dropout: float = 0.1,
                 dc: int = 16, emb: int = 4):
        super().__init__()
        self.c_in, self.dc = c_in, dc
        self.chan = nn.Parameter(torch.randn(c_in, emb) * 0.02)
        self.rnn = nn.GRU(1 + emb, dc, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.fuse = nn.Sequential(nn.Conv1d(c_in * dc, d, 1), nn.GELU())
        self.out_norm = ChanNorm(d)
        self.receptive_field = 10 ** 6

    def forward(self, x: Tensor) -> Tensor:                # x (B, C, T) -> (B, d, T)
        B, C, T = x.shape
        xi = x.reshape(B * C, T, 1)
        e = self.chan.unsqueeze(0).expand(B, C, -1).reshape(B * C, 1, -1).expand(-1, T, -1)
        h, _ = self.rnn(torch.cat([xi, e], dim=2))          # (B*C, T, dc)
        h = h.reshape(B, C * self.dc, T) if False else \
            h.reshape(B, C, T, self.dc).permute(0, 1, 3, 2).reshape(B, C * self.dc, T)
        return self.out_norm(self.fuse(h))


class SeqModel(nn.Module):
    def __init__(self, c_in: int, mu: Tensor, sd: Tensor, arch: str = "tcn", d: int = 128,
                 n_blocks: int = 8, dropout: float = 0.1, dow: bool = False, **kw):
        super().__init__()
        self.enc = DayEncoder(c_in, d, mu, sd, dow=dow)
        if arch == "tcn":
            self.backbone = TCNBackbone(d, n_blocks=n_blocks, dropout=dropout,
                                        k=int(kw.get("kernel", 3)))
        elif arch in ("gru", "lstm", "rnn"):
            self.backbone = RecurrentBackbone(d, kind=arch, n_layers=n_blocks, dropout=dropout)
        elif arch == "cigru":
            # channels must NOT be mixed before the recurrence, so this backbone consumes the
            # normalised raw channels directly and DayEncoder's projection is bypassed below.
            self.backbone = ChannelIndependentGRU(c_in, d, n_layers=n_blocks, dropout=dropout,
                                                  dc=int(kw.get("ci_width", 16)),
                                                  emb=int(kw.get("ci_emb", 4)))
        elif arch == "cnngru":
            self.backbone = CNNGRUBackbone(d, n_conv=int(kw.get("n_conv", 4)), n_gru=n_blocks,
                                           k=int(kw.get("kernel", 3)), dropout=dropout)
        elif arch.startswith("xformer"):
            pos = {"xformer": "alibi", "xformer_alibi": "alibi",
                   "xformer_learned": "learned", "xformer_rope": "rope"}[arch]
            self.backbone = XformerBackbone(d, n_layers=n_blocks, dropout=dropout,
                                            n_heads=int(kw.get("n_heads", 4)), pos=pos)
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.head = nn.Conv1d(d, 1, 1)
        self.arch = arch

    def forward(self, x: Tensor, dow_feat: Tensor | None = None) -> Tensor:
        if self.arch == "cigru":
            h = (x - self.enc.mu) / self.enc.sd            # normalise, do NOT project/mix
        else:
            h = self.enc(x, dow_feat)
        return self.head(self.backbone(h)).squeeze(1)      # (B, T)


# ------------------------------------------------------------------------------ the guard
@torch.no_grad()
def assert_causal(model: nn.Module, c_in: int, n_days: int, device: str,
                  probes: tuple[int, ...] = (95, 168, 288, 378), tol: float = 1e-4,
                  dow_feat: Tensor | None = None) -> None:
    """Perturb the future and require the past not to move.

    The analogue of run.py's `assert_no_lookahead`, and the same reasoning: causality here is
    a property of padding, masks and slopes that a one-character edit can silently break, so
    it is tested rather than reviewed.  For each probe position t we resample everything after
    t and require the output at every position <= t to be bit-comparable.
    """
    was_training = model.training
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.rand(4, c_in, n_days, generator=g).to(device)
    base = model(x, dow_feat).float()
    for t in probes:
        x2 = x.clone()
        x2[:, :, t + 1:] = torch.rand(4, c_in, n_days - t - 1, generator=g).to(device)
        out = model(x2, dow_feat).float()
        dev = (out[:, : t + 1] - base[:, : t + 1]).abs().max().item()
        if dev > tol:
            raise AssertionError(
                f"LOOK-AHEAD in {getattr(model, 'arch', type(model).__name__)}: perturbing "
                f"days > {t} moved outputs at days <= {t} by {dev:.3e}")
        after = (out[:, t + 1:] - base[:, t + 1:]).abs().max().item()
        if after < tol:
            raise AssertionError(
                f"probe {t} is vacuous: perturbing the future changed NOTHING downstream "
                f"either ({after:.3e}) -- the guard is not exercising the model")
    if was_training:
        model.train()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_days, c_in = 409, 13
    mu, sd = torch.zeros(c_in), torch.ones(c_in)
    for arch in ("tcn", "gru", "lstm", "rnn", "cnngru",
                 "xformer_alibi", "xformer_learned", "xformer_rope"):
        m = SeqModel(c_in, mu, sd, arch=arch, d=64, n_blocks=4).to(dev).float()
        assert_causal(m, c_in, n_days, dev)
        n = sum(p.numel() for p in m.parameters())
        rf = m.backbone.receptive_field
        print(f"  {arch:8s} ok  params {n:>9,}  receptive field {min(rf, n_days)}")
    print("src/seqnet.py: causality guard passed for all architectures")
