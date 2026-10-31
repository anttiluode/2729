"""Round 2 (Astra's review, 31 Oct): does feedback THROUGH THE CAVITY add anything, or is all the
adaptation local to each pair?  Everything below was frozen before it ran.

feedback  After a write, fork the system (same pair states, clocks, cavity field, noise stream) and
          apply the same next action in three branches:
            live     springs follow the pairs' D (the 2729 system)
            neumann  springs held at the unwritten value (kappa = 0); pairs still hear and update.
                     NOT a fair no-memory control: a perfectly Neumann cavity keeps an undamped
                     zero mode, and the uniform offset a write deposits then drives continuous slips
            frozen   springs frozen at their written values for the whole second write
            permuted the written springs frozen but shifted one patch along (same values, wrong place)
            pooled   every patch follows the mean of the pairs' springs (same total, no pattern)
          live vs pooled / permuted = what the wall's SPATIAL pattern contributes (fair attackers).
          live vs frozen  = what springs changing DURING the write contribute.
          live vs neumann = the zero-mode effect of having any spring at all.
          First application (fresh wall): live vs neumann vs pooled.
          Measured: firing per patch, and the drive each patch hears (RMS difference relative to
          the live drive).  The 12 actions are G4's.

order     One small job: after A then B, respond to a later probe differently than after B then A.
            A  = source 0, phase pi,     amp 1   (G2: fires patch 0 alone from a fresh wall)
            B  = source 0, phase 1.178,  amp 1   (G2: fires patch 2 alone from a fresh wall)
            P  = source 0, phase 2.356,  amp 1   (probe; state-sensitive in G4)
          A starts at t=0 (aligned clock); the second input starts `gap` steps later (shorter gaps
          overlap the first ringdown); after a full ringdown the clock is realigned and P is
          applied.  Pair phase noise on (independent seed per trial).
          Conditions: full (each patch's spring follows its own pair's D) vs pooled (every patch
          follows the MEAN of the four pairs' springs: same total wall, same zero-mode drain, but
          the wall cannot hold WHICH patch was written; the pairs' local memories are intact).
          [A first version used couple_wall off (springs held Neumann) as the baseline. Holding the
          wall perfectly Neumann keeps the cavity's zero mode undamped, which drives continuous
          slipping, so it is not a fair 'no wall memory' control; see r2_feedback_v1.json.]
          Train: gaps 1500 and 6080, noise 0.003, 5 trials per order and gap.  Held out: gap 3000 at noise
          0.003, and gap 1500 at noise 0.01 (unseen timing, unseen noise).
          Read-outs (nearest centroid): probe firing pattern; stored D before the probe (cos, sin);
          and the 3-source echo before the probe.
          Floor: any commutative memory (a sum of per-input effects) is at chance by construction.
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twoband import TwoBand, make_pair, KAPPA  # noqa: E402

R = "results"
RINGDOWN = 6000
_P = None
A = (0, math.pi, 1.0)
B = (0, 1.1780972450961724, 1.0)
PROBE = (0, 2.356194490192345, 1.0)


def proto():
    global _P
    if _P is None:
        _P = make_pair()
    return _P


def gi():
    return json.load(open(os.path.join(R, "calib.json")))["g_in"]


def save(name, obj):
    json.dump(obj, open(os.path.join(R, name), "w"), indent=1)


def fired(tw, before):
    return (tw.slips() - before).tolist()


# ------------------------------------------------------------------------------------ feedback
def _branch(base, action, mode):
    tw = base.fork()
    if mode == "neumann":
        tw.kappa_override = np.zeros(len(tw.patches))
    elif mode == "frozen":
        tw.kappa_override = tw.kappa().copy()
    elif mode == "permuted":
        tw.kappa_override = np.roll(tw.kappa(), 1)
    elif mode == "pooled":
        tw.wall_mode = "pooled"
    elif mode.startswith("scaled"):
        tw.wall_mode = "scaled"
        tw.wall_scale = float(mode[6:])
    tw.drive_log = []
    b = tw.slips().copy()
    tw.write(action["source"], action["amp"], action["phase"], dur=80, after=RINGDOWN, ref="start")
    return fired(tw, b), np.array(tw.drive_log)


def _rel(a, b):
    return float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))


def _feedback_one(action):
    fresh = TwoBand(g_in=gi(), _proto=proto())
    ref = fresh.clock()
    first = {}
    for mode in ("live", "neumann", "pooled"):
        f, d = _branch(fresh, action, mode)
        first[mode] = {"fired": f, "drive": d}
    # the real system after its first (live) application
    tw = fresh.fork()
    tw.write(action["source"], action["amp"], action["phase"], dur=80, after=RINGDOWN, ref="start")
    tw.align_clock(ref)
    kap = (tw.kappa() / KAPPA).tolist()
    second = {}
    for mode in ("live", "neumann", "frozen", "permuted", "pooled"):
        f, d = _branch(tw, action, mode)
        second[mode] = {"fired": f, "drive": d}
    return {
        "action": action,
        "first": {m: v["fired"] for m, v in first.items()},
        "first_drive_rel_diff": {m: _rel(first[m]["drive"], first["live"]["drive"]) for m in ("neumann", "pooled")},
        "kappa_over_KAPPA_before_second": kap,
        "second": {m: v["fired"] for m, v in second.items()},
        "second_drive_rel_diff": {m: _rel(second[m]["drive"], second["live"]["drive"])
                                  for m in ("neumann", "frozen", "permuted", "pooled")},
    }


def feedback():
    acts = [r["action"] for r in json.load(open(os.path.join(R, "g4_repeat.json")))["rows"]]
    with Pool(2) as pool:
        rows = pool.map(_feedback_one, acts)
    n = len(rows)
    diff = {"first": {m: sum(r["first"]["live"] != r["first"][m] for r in rows) for m in ("neumann", "pooled")},
            "second": {m: sum(r["second"]["live"] != r["second"][m] for r in rows)
                       for m in ("neumann", "frozen", "permuted", "pooled")}}
    med = {"first": {m: float(np.median([r["first_drive_rel_diff"][m] for r in rows])) for m in ("neumann", "pooled")},
           "second": {m: float(np.median([r["second_drive_rel_diff"][m] for r in rows]))
                      for m in ("neumann", "frozen", "permuted", "pooled")}}
    save("r2_feedback.json", {"rows": rows, "n": n, "firing_differs_from_live": diff, "drive_rel_diff_median": med})
    for r in rows:
        a = r["action"]
        print(f"src {a['source']} ph {a['phase']:.2f} amp {a['amp']}: k/K {np.round(r['kappa_over_KAPPA_before_second'], 2).tolist()}"
              f" | 1st {r['first']} | 2nd {r['second']}")
    print(f"feedback: firing differs from live (of {n}): {diff}")
    print(f"feedback: median relative drive difference from live: {med}")


# ------------------------------------------------------------------------------------ order
def _order_one(args):
    cond, order, gap, noise, seed = args
    tw = TwoBand(g_in=gi(), _proto=proto(), phase_noise=noise)
    tw.wall_mode = "local" if cond == "full" else "pooled"
    for k, p in enumerate(tw.pairs):
        p.gen.manual_seed(100000 * seed + 10 * k + (1 if order == "BA" else 0) + (2 if cond == "full" else 0))
    ref = tw.clock()
    x, y = (A, B) if order == "AB" else (B, A)
    s0 = tw.slips().copy()
    tw.write(x[0], x[2], x[1], dur=80, after=gap - 80, ref="start")
    s1 = tw.slips().copy()
    tw.write(y[0], y[2], y[1], dur=80, after=RINGDOWN, ref="start")
    s2 = tw.slips().copy()
    tw.align_clock(ref)
    D = tw.D().tolist()
    echo = []
    if True:
        e = tw.fork()
        for s in range(3):
            echo.append(e.echo(s, 350).tolist())
            e.settle(1500)
    b = tw.slips().copy()
    tw.write(PROBE[0], PROBE[2], PROBE[1], dur=80, after=RINGDOWN, ref="start")
    return {"cond": cond, "order": order, "gap": gap, "noise": noise, "seed": seed,
            "fired_first": (s1 - s0).tolist(), "fired_second": (s2 - s1).tolist(),
            "D_before_probe": D, "probe_fired": fired(tw, b), "echo": echo}


def _feat(r, kind):
    if kind == "probe":
        return np.array(r["probe_fired"], float)
    if kind == "D":
        d = np.array(r["D_before_probe"])
        return np.concatenate([np.cos(d), np.sin(d)])
    return np.array(r["echo"]).ravel()


def _nc_fit(rows, kind):
    X = np.stack([_feat(r, kind) for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    cents = {o: ((np.stack([_feat(r, kind) for r in rows if r["order"] == o]) - mu) / sd).mean(0)
             for o in ("AB", "BA")}
    return mu, sd, cents


def _nc_pred(model, r, kind):
    mu, sd, c = model
    z = (_feat(r, kind) - mu) / sd
    return min(c, key=lambda o: ((z - c[o]) ** 2).sum())


def _score(train, test, kind):
    if test is None:                                      # leave-one-out on train
        ok = sum(_nc_pred(_nc_fit(train[:i] + train[i + 1:], kind), r, kind) == r["order"]
                 for i, r in enumerate(train))
        return ok, len(train)
    m = _nc_fit(train, kind)
    return sum(_nc_pred(m, r, kind) == r["order"] for r in test), len(test)


def order(trials=5):
    jobs = []
    for cond in ("full", "pooled"):
        for o in ("AB", "BA"):
            for t in range(trials):
                for gap, noise, split in ((1500, 0.003, "train"), (6080, 0.003, "train"),
                                          (3000, 0.003, "heldout_gap"), (1500, 0.01, "heldout_noise")):
                    jobs.append((cond, o, gap, noise, t + (0 if split == "train" else 50)))
    with Pool(2) as pool:
        rows = pool.map(_order_one, jobs)
    save("r2_order_raw.json", rows)
    res = {}
    for cond in ("full", "pooled"):
        R_ = [r for r in rows if r["cond"] == cond]
        tr = [r for r in R_ if r["seed"] < 50]
        hg = [r for r in R_ if r["seed"] >= 50 and r["gap"] == 3000]
        hn = [r for r in R_ if r["seed"] >= 50 and r["noise"] == 0.01]
        kinds = ["probe", "D", "echo"]
        res[cond] = {k: {"train_loo": _score(tr, None, k), "heldout_gap": _score(tr, hg, k),
                         "heldout_noise": _score(tr, hn, k)} for k in kinds}
        # how often does the probe's firing pattern differ between orders at all?
        res[cond]["probe_patterns"] = {o: sorted({str(r["probe_fired"]) for r in R_ if r["order"] == o})
                                       for o in ("AB", "BA")}
        res[cond]["first_second_fired"] = {o: sorted({str((r["fired_first"], r["fired_second"])) for r in R_
                                                      if r["order"] == o}) for o in ("AB", "BA")}
    save("r2_order.json", res)
    for cond, v in res.items():
        for k in ("probe", "D", "echo"):
            if k in v:
                print(f"order {cond:5s} read-out {k:5s}: train LOO {v[k]['train_loo']}, held-out gap 3000 "
                      f"{v[k]['heldout_gap']}, held-out noise 0.01 {v[k]['heldout_noise']}")
        print(f"   probe patterns AB {v['probe_patterns']['AB']} | BA {v['probe_patterns']['BA']}")


def _floor_one(action):
    fresh = TwoBand(g_in=gi(), _proto=proto())
    ref = fresh.clock()
    tw = fresh.fork()
    tw.write(action["source"], action["amp"], action["phase"], dur=80, after=RINGDOWN, ref="start")
    tw.align_clock(ref)
    out = {"action": action}
    for mode in ("live", "scaled1.02", "scaled0.98"):
        f, d = _branch(tw, action, mode)
        out[mode] = {"fired": f, "drive": d}
    return {"action": action, "fired": {m: out[m]["fired"] for m in ("live", "scaled1.02", "scaled0.98")},
            "drive_rel_diff": {m: _rel(out[m]["drive"], out["live"]["drive"]) for m in ("scaled1.02", "scaled0.98")}}


def floor():
    """Sensitivity floor for the feedback branches: same second application with the live springs
    scaled by 1.02 or 0.98 (meaningless perturbations).  If these flip firing as often as pooled /
    permuted do, those flips say nothing about spatial memory."""
    acts = [r["action"] for r in json.load(open(os.path.join(R, "g4_repeat.json")))["rows"]]
    with Pool(2) as pool:
        rows = pool.map(_floor_one, acts)
    flips = {m: sum(r["fired"]["live"] != r["fired"][m] for r in rows) for m in ("scaled1.02", "scaled0.98")}
    med = {m: float(np.median([r["drive_rel_diff"][m] for r in rows])) for m in ("scaled1.02", "scaled0.98")}
    either = sum((r["fired"]["live"] != r["fired"]["scaled1.02"]) or (r["fired"]["live"] != r["fired"]["scaled0.98"])
                 for r in rows)
    save("r2_floor.json", {"rows": rows, "flips": flips, "either": either, "drive_rel_diff_median": med})
    print(f"floor: firing differs from live with springs x1.02 in {flips['scaled1.02']}/12, x0.98 in "
          f"{flips['scaled0.98']}/12, either {either}/12; median drive diff {med}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("parts", nargs="+", choices=["feedback", "order", "floor"])
    for part in p.parse_args().parts:
        globals()[part]()
