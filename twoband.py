"""2729: a cavity whose walls are phase pairs.

Oct 27 (Kavitaatio, WallMemoryLaw) gave a medium that READS: a ping crosses a damped
gapless cavity and its echo at one receiver carries the wall state. But its walls were
written by fiat.  Oct 29 (ChainNeuronsWithPhaseSpike, section T) gave an element that
WRITES ITSELF: a near-complement pair whose stored difference phase D gates whether an
input fires it, and whose firing moves D.  But it lives in a sub-gap medium where nothing
propagates.  This file puts them together without changing either physics.

Two bands
  * cavity band (propagating): wml/cavity.py's ellipse, wave speed 0.65, dt 0.5,
    damping 0.006, Neumann walls; a written patch adds a Robin spring.  Unchanged update.
  * pair band (private, sub-gap): one near-complement pair (A 0.03 + X 0.11+delta,
    delta 0.015) per writable patch, each in its own chains.World field (N 48, m2 0.02,
    pumped ear at 2*Omega = 0.14), exactly as experiments/theta_neuron.py builds it.

Two declared couplings (the only new rules)
  1. cavity -> pair (HEAR).  Every step, the cavity displacement averaged over patch k's
     writable wall cells, times g_in, is injected into pair k's private field through the
     same source kernel theta_neuron.py uses for its drive (2.5 cells from the pair).
  2. pair -> cavity (WALL).  Patch k's Robin spring is
        kappa_k = KAPPA * (1 - cos(D_k - D_k0)) / 2,
     D_k - D_k0 = pair k's theta_A - theta_X minus that of an unforced twin on the same clock.
     An unwritten pair leaves a Neumann wall; a pair whose D moved by pi gives the full
     Kavitaatio spring 0.28.  This transducer is a modelling choice, not derived physics.

The pairs never emit into the cavity; the cavity never sees the pairs except through
kappa.  So everything a ping reads is what the pairs wrote, and everything a pair writes
is something the cavity delivered to it.
"""
import math

import numpy as np
import torch

from chains import Config, World
from chains.fork import fork_world
from chains.probing import source_kernel
from wml.cavity import Cavity, C, DT, DAMP, KAPPA, SIGMA

torch.set_num_threads(1)

PAIR_N = 48
PAIR_DIST = 2.4
PAIR_XY = (24.0, 24.0)
DRIVE_XY = (24.0, 26.5)        # theta_neuron.py's drive position
DELTA = 0.015                  # just below the locking edge (theta_neuron T1: edge ~0.0157)
CARRIER = 0.11                 # A's complement to the pump: the frequency that pushes theta_A
PATCHES = (0, 2, 4, 6)         # ResonanceAsControlledMethod's four writable sectors of 8


def make_pair(delta=DELTA, warm=3000):
    alph = {"A": 0.03, "C": 0.05, "G": 0.09, "T": 0.11, "X0": 0.11 + delta}
    cfg = Config(N=PAIR_N, seed=0, device="cpu", p_lig=0.0, kT=0.0, phase_noise=0.0,
                 chain_boost=0.0, alphabet=alph)
    w = World(cfg)
    x, y = PAIR_XY
    w.add_monomers([0, w.letters.index("X0")], pos=[(x - PAIR_DIST / 2, y), (x + PAIR_DIST / 2, y)])
    w.pinned[:] = True
    for _ in range(warm):
        w.step()
    return w


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class TwoBand:
    """One cavity, K pair-walls, one clock."""

    def __init__(self, g_in=1.0, delta=DELTA, phase_noise=0.0, couple_wall=True, couple_hear=True,
                 patches=PATCHES, warm=3000, _proto=None):
        self.cav = Cavity(K=8, fraction=1.0, shape="ellipse")
        self.patches = list(patches)
        self.g_in, self.couple_wall, self.couple_hear = g_in, couple_wall, couple_hear
        proto = _proto if _proto is not None else make_pair(delta, warm)
        self.pairs = [fork_world(proto) for _ in self.patches]
        for k, p in enumerate(self.pairs):
            p.cfg.phase_noise = phase_noise
            p.gen.manual_seed(1000 + k)
        self.kern = source_kernel(self.pairs[0], DRIVE_XY, 1.0)
        # an unforced, noise-free twin on the same clock: the locked pair's difference
        # theta_A - theta_X rotates at the locked difference frequency, so "retained D" is
        # always measured against this shadow (what an unwritten pair would read now)
        self.shadow = fork_world(proto)
        self.shadow.cfg.phase_noise = 0.0
        self.shadow_thA = float(proto.theta[0])          # unwrapped theta_A of the unwritten twin
        # unwrapped phase bookkeeping (to count slips exactly)
        self._prev = np.stack([p.theta.numpy().copy() for p in self.pairs])
        self.unwrapped = np.zeros_like(self._prev)
        self.t = 0
        self.u = np.zeros((96, 96))
        self.prev = np.zeros((96, 96))
        cav = self.cav
        self.masks = [cav.counts[k] > 0 for k in self.patches]          # wall cells per patch
        self.ins = cav.inside.astype(float)
        self.neigh = cav.neigh.astype(float)
        self.c2 = (C * DT) ** 2
        self.a = 1 + DAMP * DT / 2
        self.b = 1 - DAMP * DT / 2
        self.yy, self.xx = cav.y, cav.x

    # ------------------------------------------------------------------ state
    def D(self):
        """Signed retained difference phase per pair (rad), relative to the unwritten shadow."""
        th = np.stack([p.theta.numpy() for p in self.pairs])
        sh = self.shadow.theta.numpy()
        return wrap((th[:, 0] - th[:, 1]) - (sh[0] - sh[1]))

    def slips(self):
        """Integer 2pi windings of each pair's sum phase against the pump since construction.
        A locked pair's two phases advance by 0.14 rad/step in total; a slip adds 2pi."""
        return np.round(self.sum_phase_excess() / (2 * math.pi)).astype(int)

    kappa_override = None          # fixed per-patch springs (Gate 0; round-2 feedback branches)
    drive_log = None               # if a list: per step, the patch displacements each pair hears

    wall_mode = "local"            # "pooled": every patch gets the mean of the pairs' springs
                                   # (round-2 control: same total wall, no spatial pattern)

    def kappa(self):
        if self.kappa_override is not None:
            return np.asarray(self.kappa_override, float)
        if not self.couple_wall:
            return np.zeros(len(self.patches))
        k = KAPPA * (1 - np.cos(self.D())) / 2
        if self.wall_mode == "pooled":
            k = np.full_like(k, k.mean())
        elif self.wall_mode == "scaled":           # sensitivity floor: live springs x wall_scale
            k = k * self.wall_scale
        return k

    def robin(self):
        r = np.zeros((96, 96))
        for k, kap in zip(self.patches, self.kappa()):
            r += kap * self.cav.counts[k]
        return r

    def pulse(self, source):
        sy, sx = self.cav.sources[source]
        return np.exp(-((self.xx - sx) ** 2 + (self.yy - sy) ** 2) / (2 * SIGMA ** 2)) * self.cav.inside

    # ------------------------------------------------------------------ dynamics
    def step(self, src_field=None, record=None):
        robin = self.robin()
        u = self.u
        lap = -(self.neigh + robin) * u
        for ax, s, nb in self.cav._nbr:
            lap += np.roll(u, s, axis=ax) * nb
        nxt = (2 * u - self.b * self.prev + self.c2 * lap) / self.a
        if src_field is not None:
            nxt = nxt + src_field
        nxt *= self.ins
        self.prev, self.u = u, nxt
        # pairs hear their patch (cavity value after this step)
        if self.drive_log is not None:
            self.drive_log.append([float(self.u[m].mean()) for m in self.masks])
        for k, p in enumerate(self.pairs):
            hear = float(self.u[self.masks[k]].mean()) if self.couple_hear else 0.0
            p.step(external_source=self.kern * (self.g_in * hear) if hear != 0.0 else None)
        a0 = float(self.shadow.theta[0])
        self.shadow.step()
        self.shadow_thA += (float(self.shadow.theta[0]) - a0 + math.pi) % (2 * math.pi) - math.pi
        th = np.stack([p.theta.numpy() for p in self.pairs])
        self.unwrapped += (th - self._prev + math.pi) % (2 * math.pi) - math.pi
        self._prev = th.copy()
        self.t += 1
        if record is not None:
            record.append(self.u[self.cav.receiver])

    def sum_phase_excess(self):
        """Sum phase (rad) advanced beyond the pump's 0.14 rad/step since construction."""
        return self.unwrapped.sum(1) - 0.14 * self.t

    def run(self, steps, src=None):
        for _ in range(steps):
            f = src(self.t) if src is not None else None
            self.step(f)

    # ------------------------------------------------------------------ actions
    def ping(self, source=0, listen=350):
        """Kavitaatio read: add the Gaussian pulse to the current field, record the receiver."""
        self.u = self.u + self.pulse(source)
        rec = []
        for _ in range(listen):
            self.step(record=rec)
        return np.array(rec)

    def echo(self, source=0, listen=350):
        """A ping's own echo: receiver trace with the ping minus an identical un-pinged twin.
        After a write the cavity still rings (and its Neumann zero mode only decays at the
        springs), so this background subtraction is what makes a ping read the wall and not the
        leftover write.  The twin is a fork of the live system, an idealised background."""
        twin = self.fork()
        rec_bg = []
        for _ in range(listen):
            twin.step(record=rec_bg)
        return self.ping(source, listen) - np.array(rec_bg)

    def write(self, source=0, amp=1.0, phase=0.0, dur=80, after=1500, ref="pump"):
        """Carrier at A's complement from one cavity source, then a quiet ringdown.
        ref="pump": cos(0.11 t + phase), phase-locked to the pump clock (theta_neuron.py's drive).
        ref="start": cos(0.11 (t - t_start) + phase): an action defined relative to its own start;
           with align_clock() before it, the whole action is repeatable on an unwritten wall.
        ref="pair": cos(0.14 t - thA_twin(t) + phase), phase-locked to the unwritten pair's own
           clock, so the pump x carrier beat meets theta_A at a fixed phase whatever the time."""
        P = self.pulse(source)
        start = self.t

        def src(t):
            if t - start < dur:
                if ref == "pump":
                    arg = CARRIER * t + phase
                elif ref == "start":
                    arg = CARRIER * (t - start) + phase
                else:
                    arg = 0.14 * t - self.shadow_thA + phase
                return P * (amp * math.cos(arg))
            return None
        self.run(dur + after, src)

    def clock(self):
        """The two clocks of an unwritten system: pump phase and the twin pair's theta_A."""
        return (0.14 * self.t) % (2 * math.pi), self.shadow_thA % (2 * math.pi)

    def align_clock(self, ref, eps=0.08, max_steps=200000):
        """Idle until BOTH clocks are within eps of ref.  An unwritten wall is the same state only
        when pump phase and pair phase return together (locked theta_A runs at ~0.016 rad/step,
        incommensurate with the 0.14 pump), so a repeatable action must start on matched clocks."""
        for i in range(max_steps):
            p, a = self.clock()
            if abs(wrap(p - ref[0])) < eps and abs(wrap(a - ref[1])) < eps:
                return i
            self.step()
        raise RuntimeError("clock did not realign")

    def settle(self, steps=1500):
        self.run(steps)

    def fork(self):
        tw = TwoBand.__new__(TwoBand)
        tw.__dict__.update(self.__dict__)
        tw.pairs = [fork_world(p) for p in self.pairs]
        tw.shadow = fork_world(self.shadow)
        tw.shadow_thA = self.shadow_thA
        for name in ("u", "prev", "_prev", "unwrapped"):
            setattr(tw, name, getattr(self, name).copy())
        return tw
