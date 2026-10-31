"""Elliptical cavity with writable wall patches (the Kavitaatio physics, K-configurable).

Same discretisation as anttiluode/Kavitaatio experiment.py (Sol, Oct 27 2026):
96x96 grid, ellipse semi-axes 36 x 27, wave speed 0.65, dt 0.5, damping 0.006,
Neumann walls (missing neighbour faces carry zero flux) and, on written patches, a
Robin spring of 0.28 per missing face.  A Gaussian pulse (sigma 1.55) starts at one
of three source cells near the first focus; the receiver is the second focus.

Differences from Kavitaatio, all optional: K patches instead of 8, a patch may cover
only the central `fraction` of its angular sector (gaps between memory cells), full
traces are recorded at every step, and many wall states run in one batched pass.
With K=8, fraction=1 it reproduces Kavitaatio's simulate() to rounding (Gate 0).
"""
import numpy as np

N = 96
C = 0.65
DT = 0.5
A_AX, B_AX = 36.0, 27.0
DAMP = 0.006
KAPPA = 0.28
SIGMA = 1.55
CENTER = (N - 1) / 2
STADIUM = (18.0, 22.0)    # straight half-length, cap radius (new-geometry test)
STEP = C * DT            # cells travelled per step


def _shift_mask(arr, axis, sign):
    out = np.roll(arr, sign, axis=axis)
    if axis == 0:
        out[0 if sign == 1 else -1, :] = False
    else:
        out[:, 0 if sign == 1 else -1] = False
    return out


class Cavity:
    def __init__(self, K=8, fraction=1.0, shape="ellipse"):
        self.K, self.fraction, self.shape = K, fraction, shape
        yy, xx = np.mgrid[:N, :N]
        X, Y = xx - CENTER, yy - CENTER
        if shape == "ellipse":
            self.inside = (X / A_AX) ** 2 + (Y / B_AX) ** 2 <= 1.0
            f = np.sqrt(A_AX ** 2 - B_AX ** 2)
            self.sources = [(round(CENTER), round(CENTER - f)), (round(CENTER + 8), round(CENTER - f)),
                            (round(CENTER - 8), round(CENTER - f))]
            self.receiver = (round(CENTER), round(CENTER + f))
            angle = np.arctan2(Y / B_AX, X / A_AX)
        elif shape == "stadium":
            # Bunimovich stadium: straight half-length L, cap radius R (chaotic ray dynamics,
            # no foci).  Source and receiver deliberately off every symmetry axis.
            L, R = STADIUM
            self.inside = ((np.abs(X) <= L) & (np.abs(Y) <= R)) | ((np.abs(X) - L) ** 2 + Y ** 2 <= R ** 2)
            self.sources = [(round(CENTER + 3), round(CENTER - 30)), (round(CENTER + 11), round(CENTER - 30)),
                            (round(CENTER - 5), round(CENTER - 30))]
            self.receiver = (round(CENTER + 6), round(CENTER + 29))
            angle = np.arctan2(Y / R, X / (L + R))
        else:
            raise ValueError(shape)
        pos = ((angle + np.pi) / (2 * np.pi)) * K               # sector coordinate in [0, K)
        self.sector = np.floor(pos).astype(int) % K
        frac_in = pos - np.floor(pos)                           # position inside the sector
        in_patch = np.abs(frac_in - 0.5) <= fraction / 2 + 1e-12
        counts = np.zeros((K, N, N))
        self.neigh = np.zeros((N, N))
        self._nbr = []
        for ax in (0, 1):
            for s in (-1, 1):
                nb = self.inside & _shift_mask(self.inside, ax, s)
                self.neigh += nb
                self._nbr.append((ax, s, nb.astype(float)))
                missing = self.inside & ~_shift_mask(self.inside, ax, s)
                for k in range(K):
                    counts[k] += missing & (self.sector == k) & in_patch
        self.counts = counts
        self.wall = counts.sum(0) > 0                           # writable wall cells
        self.y, self.x = yy, xx

    def patch_points(self, k):
        """(row, col) of the writable wall cells of patch k."""
        return np.argwhere(self.counts[k] > 0).astype(float)

    def run(self, bits, source=0, steps=350, probes=None, dtype=np.float64):
        """Simulate a batch of wall states.  bits: (B, K) 0/1.  Returns traces
        (B, steps, P) at the probe cells (default: the receiver), recorded after
        every step, so trace[:, t] equals Kavitaatio's u after step t."""
        bits = np.atleast_2d(np.asarray(bits, dtype=float))
        Bn = bits.shape[0]
        robin = np.einsum('bk,kij->bij', bits * KAPPA, self.counts).astype(dtype)
        sy, sx = self.sources[source]
        pulse = np.exp(-((self.x - sx) ** 2 + (self.y - sy) ** 2) / (2 * SIGMA ** 2)) * self.inside
        u = np.broadcast_to(pulse, (Bn, N, N)).astype(dtype).copy()
        prev = np.zeros_like(u)
        probes = probes or [self.receiver]
        pr = np.array(probes)
        out = np.zeros((Bn, steps, len(probes)), dtype=dtype)
        ins = self.inside.astype(dtype)
        c2 = (C * DT) ** 2
        a = 1 + DAMP * DT / 2
        b = 1 - DAMP * DT / 2
        neigh = self.neigh.astype(dtype)
        for t in range(steps):
            lap = -(neigh + robin) * u
            for ax, s, nb in self._nbr:
                sh = np.roll(u, s, axis=ax + 1)
                lap += sh * nb
            nxt = (2 * u - b * prev + c2 * lap) / a
            nxt *= ins
            prev, u = u, nxt
            out[:, t, :] = u[:, pr[:, 0], pr[:, 1]]
        return out
