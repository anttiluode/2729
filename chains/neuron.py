"""Chain-neuron instruments: build pinned chains, write them, read their spikes.

A "phase spike" here is not an added rule. A chain's output is the sum of its beads' emissions,
s(t) = sum_i g_i cos(theta_i). Letter frequencies are multiples of 0.01 (A .03, C .05, G .09, T .11) and the
pump locks each complement pair's SUM phase, so s(t) is a carrier locked to the pump times an envelope that
recurs with period T = 2 pi / 0.02 ~ 314 steps. The envelope peaks (all beads in phase) are the spikes. Their
timing relative to the pump clock is set by the pairs' free DIFFERENCE phases, i.e. by what was written.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.signal import find_peaks, hilbert

from chains import Config, World
from chains.fork import fork_world
from chains.probing import source_kernel

T_SPIKE = 2 * math.pi / 0.02          # envelope period, steps


def build(N=64, chains=(("ACGT", (32.0, 32.0)),), spacing=3.6, warm=600, seed=0, phase_noise=0.0):
    """chains: list of (sequence, (x_center, y)). Beads pinned, no ligation, no noise during warm-up."""
    cfg = Config(N=N, seed=seed, device="cpu", p_lig=0.0, kT=0.0, phase_noise=0.0, chain_boost=0.0)
    w = World(cfg)
    groups = []
    for seq, (cx, cy) in chains:
        start = w.P
        xs = [cx - (len(seq) - 1) * spacing / 2 + spacing * i for i in range(len(seq))]
        w.add_monomers([w.letters.index(c) for c in seq], pos=[(x, cy) for x in xs])
        groups.append(list(range(start, w.P)))
    w.pinned[:] = True
    for _ in range(warm):
        w.step()
    w.cfg.phase_noise = phase_noise
    return w, groups


def write(w, src_xy, bits, amp=0.2, steps=600, freqs=(0.03, 0.09)):
    """BPSK: bit k multiplies a carrier at freqs[k]; carrier phase referenced to the world (pump) clock."""
    k = source_kernel(w, src_xy, 1.0)
    for _ in range(steps):
        t = int(w.t)
        a = sum(b * math.cos(f * t) for b, f in zip(bits, freqs)) * amp
        w.step(external_source=k * float(a))
    return w


def record(w, groups, steps):
    """Per-chain summed emission s_c(t) = sum g cos(theta) over the chain's beads, with world times."""
    out = np.zeros((len(groups), steps))
    ts = np.zeros(steps, dtype=np.int64)
    for i in range(steps):
        w.step()
        c = torch.cos(w.theta).numpy()
        for j, g in enumerate(groups):
            out[j, i] = c[g].sum()
        ts[i] = w.t
    return out, ts


def spikes(sig, ts, min_frac=0.75):
    """Envelope peaks above min_frac of the chain's maximum possible height (number of beads ~ max |s|)."""
    env = np.abs(hilbert(sig - sig.mean()))
    p, _ = find_peaks(env, distance=int(0.6 * T_SPIKE))
    h = env[p]
    keep = h >= min_frac * max(env.max(), 1e-9)
    return ts[p][keep], h[keep], env


def spike_phase(times):
    """Circular mean spike phase relative to the pump clock, in rad, and its resultant length."""
    if len(times) == 0:
        return float("nan"), 0.0
    z = np.exp(1j * 2 * math.pi * (np.asarray(times) % T_SPIKE) / T_SPIKE).mean()
    return float(np.angle(z)), float(abs(z))
