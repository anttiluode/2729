"""The 2729 gates.  Every number here was frozen before the gate it belongs to ran.

  calib  freeze g_in: the strongest read ping, at any patch, delivers 1/10 of the direct-drive
         firing threshold (0.05) as an equivalent 80-step carrier at 0.11 rad/step.
  read   G1: 30 Kavitaatio pings (sources cycled) on a fresh system.  PASS if no pair slips
         and no patch's spring rises above 1% of KAPPA.
  write  G2: one 80-step carrier from one cavity source, then a 6000-step ringdown (the cavity
         keeps delivering the carrier long after it stops; firing happens there); 3 sources x 16
         phases x 3 amps from a fresh system.  Which patches fire?  Addressability: for each patch, is there an
         action that fires it and nothing else?
  decode G3: for every distinct fired set seen in G2, ping the written system (3 sources) and
         decode the 4-bit word by nearest template.  Templates are computed from the cavity
         alone with every written patch at the median one-slip spring measured in G2, so
         the decoder never sees a written system.  Plus detector noise.
  repeat2 G4: the same action applied four times to the same wall, every application started on
         matched clocks, with a never-written control wall at the same clock.  (theta_neuron
         T4/T5 predict state dependence: a slip moves D, and D gates firing.)
  repeat, clock  (v1, kept for the record) the same without clock matching; the control showed
         that run measured mostly the clock -> results/*_v1*.json
"""
import argparse
import itertools
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twoband import TwoBand, make_pair, CARRIER, KAPPA  # noqa: E402
from wml.cavity import Cavity  # noqa: E402

RES = "results"
THRESH = 0.05            # theta_neuron T2: 0.04 relaxes, 0.06 fires (80-step pulse)
PHASES = [2 * math.pi * i / 16 for i in range(16)]
AMPS = (0.5, 1.0, 2.0)
_PROTO = None


def proto():
    global _PROTO
    if _PROTO is None:
        _PROTO = make_pair()
    return _PROTO


def load(name):
    return json.load(open(os.path.join(RES, name)))


def save(name, obj):
    os.makedirs(RES, exist_ok=True)
    json.dump(obj, open(os.path.join(RES, name), "w"), indent=1)


def carrier_eq(trace):
    """Equivalent 80-step carrier amplitude at 0.11 rad/step: 2|sum u e^{-i w t}|/80."""
    t = np.arange(len(trace))
    return 2 * np.abs((trace * np.exp(-1j * CARRIER * t)[:, None]).sum(0)) / 80


# ---------------------------------------------------------------------------- calib
def calib():
    eq = []
    for s in range(3):
        tw = TwoBand(couple_hear=False, _proto=proto())
        tw.u = tw.u + tw.pulse(s)
        rec = []
        for _ in range(1200):
            tw.step()
            rec.append([tw.u[m].mean() for m in tw.masks])
        eq.append(carrier_eq(np.array(rec)).tolist())
    worst = float(np.max(eq))
    g_in = 0.1 * THRESH / worst
    save("calib.json", {"ping_equivalent_amp_per_source_patch": eq, "worst": worst, "g_in": g_in,
                        "rule": "strongest read ping delivers THRESH/10 at its worst patch"})
    print(f"calib: worst ping equivalent {worst:.3f}  ->  g_in = {g_in:.6f}")


def g_in():
    return load("calib.json")["g_in"]


# ---------------------------------------------------------------------------- G1 read
def gate_read(n=30, listen=350, quiet=400):
    tw = TwoBand(g_in=g_in(), _proto=proto())
    rows = []
    for i in range(n):
        D0 = tw.D().copy()
        tw.ping(i % 3, listen)
        tw.settle(quiet)
        rows.append({"ping": i, "source": i % 3, "dD": (tw.D() - D0).tolist(), "slips": tw.slips().tolist()})
    D = tw.D()
    kap = KAPPA * (1 - np.cos(D)) / 2
    ok = bool(np.all(tw.slips() == 0) and np.all(kap < 0.01 * KAPPA))
    out = {"rows": rows, "final_D": D.tolist(), "final_kappa": kap.tolist(), "final_slips": tw.slips().tolist(),
           "max_abs_dD_per_ping": float(max(np.abs(r["dD"]).max() for r in rows)), "pass": ok}
    save("g1_read.json", out)
    print(f"G1 read: {n} pings, slips {tw.slips().tolist()}, final D {np.round(D, 4).tolist()}, "
          f"kappa/KAPPA max {kap.max() / KAPPA:.2e}, max |dD| per ping {out['max_abs_dD_per_ping']:.2e} -> "
          f"{'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------- G2 write
RINGDOWN = 6000          # cavity amplitude ~ exp(-0.0015 t): 6000 steps -> ~1e-4 of peak


def _write_one(args):
    s, ph, amp, gi = args
    tw = TwoBand(g_in=gi, _proto=proto())
    tw.write(s, amp, ph, dur=80, after=0)
    fire_t = [[] for _ in tw.patches]
    last = tw.slips().copy()
    for i in range(RINGDOWN):
        tw.step()
        if i % 10 == 9:
            now = tw.slips()
            for k in np.nonzero(now != last)[0]:
                fire_t[k].append(80 + i + 1)
            last = now.copy()
    return {"source": s, "phase": ph, "amp": amp, "slips": tw.slips().tolist(), "D": tw.D().tolist(),
            "fire_steps_after_carrier_start": fire_t, "cavity_rms_end": float(np.sqrt((tw.u ** 2).mean()))}


def gate_write():
    gi = g_in()
    jobs = [(s, ph, a, gi) for s in range(3) for a in AMPS for ph in PHASES]
    with Pool(2) as pool:
        rows = pool.map(_write_one, jobs)
    fired = [tuple(int(x != 0) for x in r["slips"]) for r in rows]
    sets = sorted(set(fired))
    solo = {}
    for k in range(4):
        hits = [r for r, f in zip(rows, fired) if f == tuple(int(i == k) for i in range(4))]
        solo[k] = {"n_actions": len(hits), "example": hits[0] if hits else None}
    one_slip_D = [d for r in rows for d, s in zip(r["D"], r["slips"]) if s == 1]
    multi = sum(1 for r in rows for s in r["slips"] if abs(s) > 1)
    out = {"rows": rows, "distinct_fired_sets": [list(x) for x in sets], "solo": solo,
           "one_slip_D": {"n": len(one_slip_D), "median": float(np.median(one_slip_D)) if one_slip_D else None,
                          "min": float(min(one_slip_D)) if one_slip_D else None,
                          "max": float(max(one_slip_D)) if one_slip_D else None},
           "nonfiring_abs_dD_max": float(max(abs(d) for r in rows for d, s in zip(r["D"], r["slips"]) if s == 0)),
           "multi_slip_events": multi}
    save("g2_write.json", out)
    print(f"G2 write: {len(rows)} actions, {len(sets)} distinct fired sets: {sets}")
    for k in range(4):
        print(f"   patch {k}: fired ALONE by {solo[k]['n_actions']} actions")
    ft = [t for r in rows for ts in r["fire_steps_after_carrier_start"] for t in ts]
    out["fire_step_quantiles"] = np.quantile(ft, [0, .5, .9, 1]).tolist() if ft else None
    out["cavity_rms_end_max"] = max(r["cavity_rms_end"] for r in rows)
    save("g2_write.json", out)
    print(f"   firing steps after carrier start (min/median/90%/max): {out['fire_step_quantiles']}; "
          f"cavity rms at end max {out['cavity_rms_end_max']:.2e}")
    print(f"   one-slip D: {out['one_slip_D']}; non-firing |dD| max {out['nonfiring_abs_dD_max']:.3f}; "
          f"multi-slip events {multi}")


# ---------------------------------------------------------------------------- G3 decode
def templates(kap1, listen=350):
    """Cavity-only echoes for all 16 words, every written patch at spring kap1."""
    cav = Cavity(K=8)
    words = list(itertools.product((0, 1), repeat=4))
    bits = np.zeros((16, 8))
    for i, w in enumerate(words):
        for b, k in zip(w, (0, 2, 4, 6)):
            bits[i, k] = b * kap1 / KAPPA          # cavity.run multiplies bits by KAPPA
    return words, np.stack([cav.run(bits, source=s, steps=listen)[:, :, 0] for s in range(3)], 1)


def _decode_one(args):
    r, gi = args
    tw = TwoBand(g_in=gi, _proto=proto())
    tw.write(r["source"], r["amp"], r["phase"], dur=80, after=RINGDOWN)
    traces = []
    for s in range(3):
        traces.append(tw.echo(s, 350))
        tw.settle(1500)            # let the cavity ring down between pings
    return {"action": {k: r[k] for k in ("source", "phase", "amp")}, "slips": tw.slips().tolist(),
            "D": tw.D().tolist(), "traces": np.array(traces).tolist()}


def gate_decode(noise_fracs=(0.0, 0.04, 0.16), reps=50, seed=2729):
    g2 = load("g2_write.json")
    gi = g_in()
    kap1 = KAPPA * (1 - math.cos(g2["one_slip_D"]["median"])) / 2
    words, T = templates(kap1)                                  # (16, 3, 350)
    # one written system per distinct fired set (first action that produced it), plus the fresh system
    seen, picks = set(), []
    for r in g2["rows"]:
        f = tuple(int(x != 0) for x in r["slips"])
        if f not in seen:
            seen.add(f)
            picks.append(r)
    with Pool(2) as pool:
        sys_rows = pool.map(_decode_one, [(r, gi) for r in picks])
    ref_rms = float(np.median([np.sqrt(((T[i] - T[0]) ** 2).mean()) for i in range(1, 16)]))
    rng = np.random.default_rng(seed)
    res = {"kappa_one_slip_template": kap1, "template_word_rms_median": ref_rms, "systems": [], "noise": {}}
    for nf in noise_fracs:
        correct = 0
        total = 0
        for sr in sys_rows:
            truth = tuple(int(x != 0) for x in sr["slips"])
            y = np.array(sr["traces"])
            for _ in range(reps if nf > 0 else 1):
                z = y + rng.normal(0, nf * ref_rms, y.shape)
                d = ((T - z[None]) ** 2).sum((1, 2))
                guess = words[int(np.argmin(d))]
                correct += guess == truth
                total += 1
        res["noise"][str(nf)] = {"correct": correct, "total": total}
        print(f"G3 decode noise {nf:.2f} x template-rms: {correct}/{total}")
    for sr in sys_rows:
        truth = tuple(int(x != 0) for x in sr["slips"])
        y = np.array(sr["traces"])
        d = ((T - y[None]) ** 2).sum((1, 2))
        res["systems"].append({"action": sr["action"], "truth": truth, "guess": words[int(np.argmin(d))],
                               "D": sr["D"], "slips": sr["slips"],
                               "resid_vs_truth_template": float(np.sqrt(d[words.index(truth)] / y.size))})
    # G3b analog read: the wall stores D (through kappa), not "fired or not".  Nearest template on
    # a grid of 6 spring levels per patch (6^4 = 1296 words), noiseless echo, all 3 sources.
    lv_D = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.4])
    lv = (1 - np.cos(lv_D)) / 2                                 # kappa / KAPPA
    grid = np.array(list(itertools.product(range(6), repeat=4)))
    bits = np.zeros((len(grid), 8))
    bits[:, [0, 2, 4, 6]] = lv[grid]
    cav = Cavity(K=8)
    TG = np.stack([cav.run(bits, source=s, steps=350)[:, :, 0] for s in range(3)], 1)
    est, true = [], []
    for sr in sys_rows:
        y = np.array(sr["traces"])
        i = int(np.argmin(((TG - y[None]) ** 2).sum((1, 2))))
        est.append(lv[grid[i]].tolist())
        true.append(((1 - np.cos(np.array(sr["D"]))) / 2).tolist())
    est, true = np.array(est), np.array(true)
    res["analog"] = {"levels_D": lv_D.tolist(), "est_kappa_over_KAPPA": est.tolist(),
                     "true_kappa_over_KAPPA": true.tolist(),
                     "corr": float(np.corrcoef(est.ravel(), true.ravel())[0, 1]),
                     "mean_abs_err": float(np.abs(est - true).mean()),
                     "grid_step_mean": float(np.diff(lv).mean())}
    print(f"G3b analog: corr(est, true kappa) {res['analog']['corr']:.3f}, mean |err| "
          f"{res['analog']['mean_abs_err']:.3f} (grid step ~{res['analog']['grid_step_mean']:.3f}) x KAPPA")
    for sr, e, t in zip(sys_rows, est, true):
        print(f"   slips {sr['slips']}  true k/K {np.round(t, 3).tolist()}  read {np.round(e, 3).tolist()}")
    save("g3_decode.json", res)


# ---------------------------------------------------------------------------- G4 repeat
def _repeat_one(args):
    r, gi, n = args
    tw = TwoBand(g_in=gi, _proto=proto())
    steps = []
    for i in range(n):
        before = tw.slips().copy()
        tw.write(r["source"], r["amp"], r["phase"], dur=80, after=RINGDOWN)
        steps.append({"new_slips": (tw.slips() - before).tolist(), "D": tw.D().tolist()})
    return {"action": {k: r[k] for k in ("source", "phase", "amp")}, "applications": steps}


def gate_repeat(n=4):
    g2 = load("g2_write.json")
    gi = g_in()
    picks = [r for r in g2["rows"] if any(r["slips"])]
    rng = np.random.default_rng(29)
    picks = [picks[i] for i in sorted(rng.choice(len(picks), size=min(12, len(picks)), replace=False))]
    with Pool(2) as pool:
        rows = pool.map(_repeat_one, [(r, gi, n) for r in picks])
    same_second = sum(1 for r in rows if r["applications"][1]["new_slips"] == r["applications"][0]["new_slips"])
    save("g4_repeat_v1_confounded.json", {"rows": rows, "n_actions": len(rows),
                            "second_application_identical_to_first": same_second})
    for r in rows:
        a = r["action"]
        seq = " -> ".join(str([int(x) for x in s["new_slips"]]) for s in r["applications"])
        print(f"G4 src {a['source']} phase {a['phase']:.2f} amp {a['amp']}: {seq}")
    print(f"G4: second application identical to first in {same_second}/{len(rows)}")


# ---------------------------------------------------------------------------- G4 clock control
_IDLE = None


def _clock_one(args):
    r, k = args
    tw = _IDLE[k].fork()
    before = tw.slips().copy()
    tw.write(r["source"], r["amp"], r["phase"], dur=80, after=RINGDOWN)
    return (tw.slips() - before).tolist()


def gate_clock():
    """Attacker for G4: is 'the second application differs' just the pump clock?  Apply each G4
    action to a FRESH, never-written system at the same absolute time as its 2nd/3rd/4th
    application.  If fresh-but-late reproduces the repeated sequence, G4 measured the clock."""
    global _IDLE
    g4 = load("g4_repeat_v1_confounded.json")
    n = len(g4["rows"][0]["applications"])
    gi = g_in()
    tw = TwoBand(g_in=gi, _proto=proto())
    _IDLE = {}
    for k in range(1, n):
        tw.run(80 + RINGDOWN)
        _IDLE[k] = tw.fork()
    jobs = [(row["action"], k) for row in g4["rows"] for k in range(1, n)]
    with Pool(2) as pool:
        outs = pool.map(_clock_one, jobs)
    rows, i = [], 0
    agree_first, agree_repeat = 0, 0
    for row in g4["rows"]:
        first = row["applications"][0]["new_slips"]
        late = []
        for k in range(1, n):
            fresh = outs[i]
            i += 1
            rep = row["applications"][k]["new_slips"]
            late.append({"k": k, "fresh_late": fresh, "repeated": rep})
            agree_first += fresh == first
            agree_repeat += fresh == rep
        rows.append({"action": row["action"], "first": first, "late": late})
    total = len(jobs)
    save("g4_clock_v1.json", {"rows": rows, "fresh_late_equals_first": agree_first,
                           "fresh_late_equals_repeated": agree_repeat, "total": total})
    for r in rows:
        a = r["action"]
        print(f"clock src {a['source']} ph {a['phase']:.2f} amp {a['amp']}: first {r['first']}  fresh-late "
              f"{[x['fresh_late'] for x in r['late']]}  repeated {[x['repeated'] for x in r['late']]}")
    print(f"G4 clock control: fresh-but-late == first application in {agree_first}/{total}; "
          f"== the repeated system in {agree_repeat}/{total}")


# ---------------------------------------------------------------------------- G4 (v2, clock-matched)
def _repeat2_one(args):
    r, gi, n = args
    tw = TwoBand(g_in=gi, _proto=proto())
    ref = tw.clock()
    apps = []
    for i in range(n):
        waited = tw.align_clock(ref) if i else 0
        before = tw.slips().copy()
        tw.write(r["source"], r["amp"], r["phase"], dur=80, after=RINGDOWN, ref="start")
        apps.append({"waited": waited, "new_slips": (tw.slips() - before).tolist(), "D": tw.D().tolist()})
    # control: a never-written wall, idled the same time, realigned, same action
    f = TwoBand(g_in=gi, _proto=proto())
    f.run(80 + RINGDOWN)
    f.align_clock(ref)
    b = f.slips().copy()
    f.write(r["source"], r["amp"], r["phase"], dur=80, after=RINGDOWN, ref="start")
    return {"action": {k: r[k] for k in ("source", "phase", "amp")}, "applications": apps,
            "fresh_late_control": (f.slips() - b).tolist()}


def gate_repeat2(n=4):
    """Same action, same wall, n times, every application started on matched clocks (pump phase
    and unwritten-pair phase both within 0.08 rad of the first start; carrier timed from its own
    start).  Validated separately: on a never-written wall this makes an action repeat 16/16."""
    g4 = load("g4_repeat_v1_confounded.json")
    gi = g_in()
    picks = [{"source": r["action"]["source"], "phase": r["action"]["phase"], "amp": r["action"]["amp"]}
             for r in g4["rows"]]
    with Pool(2) as pool:
        rows = pool.map(_repeat2_one, [(r, gi, n) for r in picks])
    ctrl_ok = sum(r["fresh_late_control"] == r["applications"][0]["new_slips"] for r in rows)
    same2 = sum(r["applications"][1]["new_slips"] == r["applications"][0]["new_slips"] for r in rows)
    silent2 = sum(not any(r["applications"][1]["new_slips"]) for r in rows)
    save("g4_repeat.json", {"rows": rows, "n_actions": len(rows), "control_fresh_late_equals_first": ctrl_ok,
                            "second_identical_to_first": same2, "second_fires_nothing": silent2})
    for r in rows:
        a = r["action"]
        seq = " -> ".join(str([int(x) for x in s["new_slips"]]) for s in r["applications"])
        print(f"G4 src {a['source']} phase {a['phase']:.2f} amp {a['amp']}: {seq}   (fresh wall, late: "
              f"{r['fresh_late_control']})")
    print(f"G4: control fresh-late == first in {ctrl_ok}/{len(rows)}; on the SAME wall the 2nd application "
          f"== 1st in {same2}/{len(rows)}; 2nd fires nothing in {silent2}/{len(rows)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("gates", nargs="+", choices=["calib", "read", "write", "decode", "repeat", "clock", "repeat2"])
    a = p.parse_args()
    for gname in a.gates:
        {"calib": calib, "read": gate_read, "write": gate_write, "decode": gate_decode,
         "repeat": gate_repeat, "clock": gate_clock, "repeat2": gate_repeat2}[gname]()
