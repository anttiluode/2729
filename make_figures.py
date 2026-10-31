"""Figures from the saved receipts in results/ (no simulation)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = "results"


def load(n):
    return json.load(open(os.path.join(R, n)))


def fig_write():
    g = load("g2_write.json")
    rows = g["rows"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for j, amp in enumerate((0.5, 1.0, 2.0)):
        M = np.full((3 * 16, 4), np.nan)
        for r in rows:
            if r["amp"] != amp:
                continue
            i = r["source"] * 16 + int(round(r["phase"] / (2 * np.pi / 16)))
            M[i] = [1.0 if s else np.tanh(abs(d) / 0.5) * 0.4 for s, d in zip(r["slips"], r["D"])]
        ax[j].imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax[j].set_title(f"carrier amplitude {amp}")
        ax[j].set_xticks(range(4), [f"patch {k}" for k in range(4)])
        ax[j].set_yticks([8, 24, 40], ["source 0", "source 1", "source 2"])
        for y in (15.5, 31.5):
            ax[j].axhline(y, color="w", lw=0.6)
    fig.suptitle("G2: which wall pairs one carrier fires (bright = slip; dim = D moved without firing).  "
                 "Rows: 16 carrier phases per source")
    fig.savefig(os.path.join(R, "g2_write.png"), dpi=120)


def fig_decode():
    g = load("g3_decode.json")["analog"]
    e = np.array(g["est_kappa_over_KAPPA"]).ravel()
    t = np.array(g["true_kappa_over_KAPPA"]).ravel()
    fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="0.7", lw=1)
    ax.scatter(t, e, s=22)
    ax.set_xlabel("true spring / KAPPA  (from the pair's D)")
    ax.set_ylabel("spring read from the echo (6-level grid)")
    ax.set_title(f"G3b: echo reads the pairs' D   r = {g['corr']:.2f}")
    fig.savefig(os.path.join(R, "g3_analog.png"), dpi=120)


if __name__ == "__main__":
    fig_write()
    if os.path.exists(os.path.join(R, "g3_decode.json")):
        fig_decode()
