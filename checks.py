"""Receipts for the README's mechanism claims (each was first seen in a debug run).

  ringdown   who writes: cavity left as is / silenced at carrier end / only its mean removed
  readwrit   on written walls: read cycles vs quiet twin vs silenced-cavity twin
  wobble     a fired pair alone (cavity silenced): period, depth and cycle-mean stability of D
  clock      is an action repeatable on a fresh wall?  absolute carrier, pair-referenced carrier,
             and start-timed carrier on matched clocks (pump phase + twin theta_A within 0.08)
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twoband import TwoBand, make_pair  # noqa: E402

R = "results"
RINGDOWN = 6000
P = None


def proto():
    global P
    if P is None:
        P = make_pair()
    return P


def gi():
    return json.load(open(os.path.join(R, "calib.json")))["g_in"]


def g2_action(fired):
    rows = json.load(open(os.path.join(R, "g2_write.json")))["rows"]
    return [r for r in rows if tuple(int(x != 0) for x in r["slips"]) == fired][0]


def save(name, obj):
    json.dump(obj, open(os.path.join(R, name), "w"), indent=1)


def _ringdown_one(fired):
    r = g2_action(fired)
    tw = TwoBand(g_in=gi(), _proto=proto())
    tw.write(r["source"], r["amp"], r["phase"], dur=80, after=0)
    out = {"action": {k: r[k] for k in ("source", "phase", "amp")}, "slips_at_carrier_end": tw.slips().tolist()}
    arms = {"as_is": tw.fork(), "silenced": tw.fork(), "mean_removed": tw.fork()}
    arms["silenced"].u[:] = 0
    arms["silenced"].prev[:] = 0
    ins = tw.cav.inside
    arms["mean_removed"].u[ins] -= arms["mean_removed"].u[ins].mean()
    arms["mean_removed"].prev[ins] -= arms["mean_removed"].prev[ins].mean()
    for name, a in arms.items():
        a.run(RINGDOWN)
        out[name] = a.slips().tolist()
    return out


def ringdown():
    sets = [(1, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0)]
    with Pool(2) as pool:
        rows = pool.map(_ringdown_one, sets)
    save("check_ringdown.json", rows)
    for r in rows:
        print(f"ringdown {r['action']}: at carrier end {r['slips_at_carrier_end']}; as is {r['as_is']}, "
              f"silenced {r['silenced']}, mean removed {r['mean_removed']}")


def _readwrit_one(fired):
    r = g2_action(fired)
    tw = TwoBand(g_in=gi(), _proto=proto())
    tw.write(r["source"], r["amp"], r["phase"], dur=80, after=RINGDOWN)
    read, quiet, silent = tw.fork(), tw.fork(), tw.fork()
    silent.u[:] = 0
    silent.prev[:] = 0
    cyc = []
    for i in range(3):
        read.echo(i, 350)
        read.settle(1500)
        quiet.settle(350 + 1500)    # echo() advances the read arm 350 steps (its twin is a fork)
        silent.settle(350 + 1500)
        cyc.append({"read": read.D().tolist(), "quiet": quiet.D().tolist(), "silenced": silent.D().tolist()})
    return {"action": {k: r[k] for k in ("source", "phase", "amp")}, "D_after_write": tw.D().tolist(),
            "cycles": cyc, "slips": {"read": read.slips().tolist(), "quiet": quiet.slips().tolist()}}


def readwrit():
    sets = [(1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 0)]
    with Pool(2) as pool:
        rows = pool.map(_readwrit_one, sets)
    dq = max(abs(math.remainder(a - b, 2 * math.pi)) for r in rows for c in r["cycles"]
             for a, b in zip(c["read"], c["quiet"]))
    ds = max(abs(math.remainder(a - b, 2 * math.pi)) for r in rows for c in r["cycles"]
             for a, b in zip(c["quiet"], c["silenced"]))
    save("check_readwrit.json", {"rows": rows, "max_read_vs_quiet": dq, "max_quiet_vs_silenced": ds})
    print(f"readwrit: {len(rows)} written walls x 3 read cycles: max |D_read - D_quiet| {dq:.4f} rad; "
          f"max |D_quiet - D_silenced| {ds:.4f}; slips read {[r['slips']['read'] for r in rows]} "
          f"quiet {[r['slips']['quiet'] for r in rows]}")


def wobble():
    r = g2_action((1, 1, 0, 1))
    tw = TwoBand(g_in=gi(), _proto=proto())
    tw.write(r["source"], r["amp"], r["phase"], dur=80, after=RINGDOWN)
    tw.u[:] = 0
    tw.prev[:] = 0
    Ds = []
    for i in range(16000):
        tw.step()
        if i % 5 == 0:
            Ds.append(tw.D().copy())
    Ds = np.unwrap(np.array(Ds), axis=0)
    rows = []
    for k in range(4):
        x = Ds[:, k] - Ds[:, k].mean()
        f = np.fft.rfftfreq(len(x), 5)
        j = 1 + int(np.argmax(np.abs(np.fft.rfft(x))[1:]))
        h = len(x) // 2
        rows.append({"patch": k, "mean": float(Ds[:, k].mean()), "range": float(np.ptp(Ds[:, k])),
                     "period_steps": float(1 / f[j]), "first_half_mean": float(Ds[:h, k].mean()),
                     "second_half_mean": float(Ds[h:, k].mean())})
        print(f"wobble patch {k}: mean {rows[-1]['mean']:+.3f} range {rows[-1]['range']:.3f} period "
              f"{rows[-1]['period_steps']:.0f}  halves {rows[-1]['first_half_mean']:+.3f} / "
              f"{rows[-1]['second_half_mean']:+.3f}")
    save("check_wobble.json", {"action": {k: r[k] for k in ("source", "phase", "amp")}, "patches": rows})


_NOW = _LATE = None


def _clock_one(a):
    s, ph, lag, ref = a
    tw = (_LATE if lag else _NOW).fork()
    b = tw.slips().copy()
    tw.write(s, 2.0, ph, dur=80, after=RINGDOWN, ref=ref)
    return (tw.slips() - b).tolist()


def clock():
    global _NOW, _LATE
    _NOW = TwoBand(g_in=gi(), _proto=proto())
    ref = _NOW.clock()
    t0 = _NOW.shadow_thA
    probe = _NOW.fork()
    probe.run(3000)
    rate = (probe.shadow_thA - t0) / 3000
    _LATE = TwoBand(g_in=gi(), _proto=proto())
    _LATE.run(80 + RINGDOWN)
    waited = _LATE.align_clock(ref)
    out = {"locked_thetaA_rate": rate, "aligned_after_extra_steps": waited, "late_clock": _LATE.clock(),
           "ref_clock": ref}
    phases = [2 * math.pi * i / 8 for i in range(8)]
    for refname in ("pump", "pair", "start"):
        jobs = [(s, ph, lag, refname) for s in (0, 1) for ph in phases for lag in (0, 1)]
        with Pool(2) as pool:
            res = pool.map(_clock_one, jobs)
        same = sum(res[i] == res[i + 1] for i in range(0, len(res), 2))
        out[refname] = {"same": same, "total": len(res) // 2,
                        "pairs": [[jobs[i][:2], res[i], res[i + 1]] for i in range(0, len(res), 2)]}
        print(f"clock carrier ref={refname}: fresh wall now vs fresh wall later (clocks matched) "
              f"agree {same}/{len(res) // 2}")
    print(f"locked theta_A rate {rate:.5f} rad/step")
    save("check_clock.json", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checks", nargs="+", choices=["ringdown", "readwrit", "wobble", "clock"])
    for c in p.parse_args().checks:
        globals()[c]()
